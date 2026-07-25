"""CLIENT-SIDE cast observer: per-agent remaining cast time from the native combat-event queue.

Guild Wars does not expose a live "remaining cast time" getter (``Agent.GetRemainingCastTime`` is a
stub that returns 0). The only source is the combat-event stream: watch each agent's SKILL_ACTIVATED
+ CASTTIME, and count down against the observed duration. Py4GW already has this logic in
``CombatEventQueue_src.helpers`` but it is dormant by design (its ``_callback_active`` query gate is
off, so the deque is never populated). So we drive the SAME method ourselves off the PUBLIC native
queue (``CombatEventQueue.PeekEvents`` -- non-destructive, not gated).

This is AGENT-observable data (the client watching casts happen in its own instance), so -- unlike the
shared-memory party region -- it is legitimate to encode and send over the socket. The client fills
``AgentView.remaining_cast_time_ms`` / ``casting_skill_id`` from here; the host reads it for interrupt
feasibility.

Threading: ``pump()`` reads the native queue and MUST run on a GW-legal thread (the client main
thread, once per frame). The getters read only local dict state and are called from the same thread
(the encoder runs in the client's map-gated update()).
"""
import Py4GW
from Py4GWCoreLib.CombatEvents import CombatEventQueue   # not re-exported at package top

from .protocol import CastStatus

# --- native-capture bring-up + instrumentation ---------------------------------------------------
# The higher-level CombatEvents stream is DEACTIVATED by design in this Py4GW build (Py4GW's own docs
# call it "the deactivated event stream"; HeroAI's interrupt sampler polls Agent.IsCasting instead of
# using it). We drive the RAW native queue directly (PeekEvents/GetAndClearEvents read native memory,
# ungated by the dormant _callback_active flag) -- but that only yields events if the native queue is
# actually INITIALIZED and capturing. The lazy init inside PeekEvents can fail silently (our pump used
# to swallow the exception), which looks identical to "no casts". So we now (re)initialize EXPLICITLY
# and LOG the queue state, so the in-game console tells us whether capture is live. Success also shows
# up on the server diagnostic: remain>0 and casts go non-zero once events flow.
_capture_ready = False
_peek_err_logged = False
_events_ever_seen = 0
_first_batch_logged = False


def _clog(msg: str, level: str = "Info") -> None:
    try:
        mt = getattr(Py4GW.Console.MessageType, level, Py4GW.Console.MessageType.Info)
        Py4GW.Console.Log("cc.cast_observer", msg, mt)
    except Exception:
        pass


def ensure_capture() -> None:
    """Force the native combat-event queue to initialize so PeekEvents actually captures, logging its
    state once. If it was already 'initialized' but capture is dead, force a Terminate+Initialize to
    re-arm the packet hook (nothing else consumes this queue in our client, so a reset is safe)."""
    global _capture_ready
    if _capture_ready:
        return
    _capture_ready = True   # one-shot regardless of outcome (don't spam the console every frame)
    try:
        was_init = bool(CombatEventQueue.IsInitialized())
        if not was_init:
            CombatEventQueue.Initialize()
        else:
            # already 'initialized' yet we saw nothing -> re-arm the native capture.
            try:
                CombatEventQueue.Terminate()
            except Exception:
                pass
            CombatEventQueue.Initialize()
        try:
            CombatEventQueue.SetMaxEvents(512)
        except Exception:
            pass
        _clog(f"combat queue bring-up: was_init={was_init} now_init={CombatEventQueue.IsInitialized()} "
              f"size={CombatEventQueue.GetQueueSize()}")
    except Exception as e:
        _clog(f"combat queue init FAILED: {e!r} -- native capture unavailable in this build", "Error")

# EventType ids (Py4GWCoreLib/enums_src/Event_enums.py -- mirrored in Py4GW/stubs/PyCombatEvents.pyi)
_ACTIVATED = (1, 2)            # SKILL_ACTIVATED, ATTACK_SKILL_ACTIVATED  (timed casts)
_INSTANT = 7                   # INSTANT_SKILL_ACTIVATED (no cast time) -> lands as an immediate success
_CASTTIME = 18                 # fval = duration in seconds
_END_SUCCESS = (4, 5)          # SKILL_FINISHED, ATTACK_SKILL_FINISHED
_END_FAIL = (3, 6, 8)          # SKILL_STOPPED, INTERRUPTED, ATTACK_SKILL_STOPPED (cancelled/interrupted)
_END = _END_SUCCESS + _END_FAIL

# agent_id -> [skill_id, target_id, start_ms, duration_ms]  (duration 0 until a CASTTIME arrives)
_casts: dict = {}
_watermark = 0                 # highest event timestamp folded in so far
_STALE_MS = 30000              # drop a dangling cast we never saw end (matches helpers' 30s window)

# --- recent-cast history (for the server world view) ---------------------------------------------
# agent_id -> list[[seq, skill_id, status, ts_ms]], most-recent LAST, capped at _HIST_MAX. Unlike
# _casts (in-progress only, for remaining time), this RETAINS the outcome (success/interrupt) of the
# last few casts so the host can render an authoritative per-agent cast bar. Outcome comes straight
# from the combat-event type -- no server-side heuristic. Keyed by a monotonic seq so ordering +
# repeated skills survive the wire delta/tombstone codec (see protocol.CastRecord).
_history: dict = {}
_seq = 0                       # monotonic cast sequence id (the wire key)
_HIST_MAX = 3                  # keep the last N casts per agent (the bar shows this many)
_HIST_STALE_MS = 60000         # forget an agent's history if it hasn't cast in this long (memory hygiene)


def _now() -> int:
    return int(Py4GW.Game.get_tick_count64())


def _hist_add(agent: int, skill: int, status: int, ts: int) -> None:
    """Append a new cast entry (fresh seq) to an agent's history, trimmed to the last _HIST_MAX."""
    global _seq
    if not skill:
        return
    _seq += 1
    lst = _history.setdefault(agent, [])
    lst.append([_seq, int(skill), int(status), ts])
    if len(lst) > _HIST_MAX:
        del lst[:-_HIST_MAX]


def _hist_finish(agent: int, status: int) -> None:
    """Resolve the agent's most-recent in-progress (Casting) entry to a terminal status. No-op if we
    never saw the start (e.g. it happened before we began pumping)."""
    lst = _history.get(agent)
    if not lst:
        return
    for entry in reversed(lst):
        if entry[2] == int(CastStatus.Casting):
            entry[2] = int(status)
            return


def pump() -> None:
    """Fold newly-arrived combat events into per-agent cast state. Non-destructive peek + a timestamp
    watermark, so we never starve another consumer of the native queue."""
    global _watermark, _events_ever_seen, _peek_err_logged, _first_batch_logged
    ensure_capture()
    try:
        events = CombatEventQueue.PeekEvents()
    except Exception as e:
        if not _peek_err_logged:
            _peek_err_logged = True
            _clog(f"PeekEvents raised (native queue unavailable): {e!r}", "Error")
        return
    if events:
        _events_ever_seen += len(events)
        if not _first_batch_logged:
            _first_batch_logged = True
            _clog(f"FIRST combat events captured ({len(events)}) -- native capture is LIVE")
    newest = _watermark
    for ev in events:
        try:
            ts = int(ev.timestamp)
        except Exception:
            continue
        if ts <= _watermark:
            continue
        if ts > newest:
            newest = ts
        etype = int(ev.event_type)
        agent = int(ev.agent_id)
        if etype in _ACTIVATED:
            _casts[agent] = [int(ev.value), int(ev.target_id), ts, 0]
            _hist_add(agent, int(ev.value), int(CastStatus.Casting), ts)
        elif etype == _INSTANT:
            _hist_add(agent, int(ev.value), int(CastStatus.Success), ts)
        elif etype == _CASTTIME:
            c = _casts.get(agent)
            if c is not None:
                c[3] = int(float(ev.float_value) * 1000.0)
        elif etype in _END:
            _casts.pop(agent, None)
            _hist_finish(agent, CastStatus.Success if etype in _END_SUCCESS else CastStatus.Interrupted)
    _watermark = newest
    _expire()


def _expire() -> None:
    """Drop in-progress casts that ran past their duration (finish event missed) or went stale, and
    forget history for agents that haven't cast in a while."""
    now = _now()
    dead = [a for a, c in _casts.items()
            if (now - c[2] > _STALE_MS) or (c[3] > 0 and now - c[2] > c[3])]
    for a in dead:
        _casts.pop(a, None)
    stale_hist = [a for a, lst in _history.items() if not lst or now - lst[-1][3] > _HIST_STALE_MS]
    for a in stale_hist:
        _history.pop(a, None)


def reset() -> None:
    """Clear all tracked casts + history (e.g. on zone). Safe to call any time."""
    global _watermark, _seq
    _casts.clear()
    _history.clear()
    _watermark = 0
    _seq = 0


# --------------------------------------------------------------------------- getters
def remaining_cast_time_ms(agent_id: int) -> int:
    c = _casts.get(int(agent_id))
    if c is None or c[3] <= 0:
        return 0
    return max(0, c[3] - (_now() - c[2]))


def casting_skill_id(agent_id: int) -> int:
    c = _casts.get(int(agent_id))
    return int(c[0]) if c is not None else 0


def casting_target_id(agent_id: int) -> int:
    c = _casts.get(int(agent_id))
    return int(c[1]) if c is not None else 0


def cast_state(agent_id: int):
    """(skill_id, target_id, remaining_ms) when the agent is mid-cast with a known remaining time,
    else None. The encoder prefers this over the living-struct casting flag so the shipped
    remaining_cast_time_ms and casting_skill_id stay mutually consistent."""
    c = _casts.get(int(agent_id))
    if c is None or c[3] <= 0:
        return None
    remaining = c[3] - (_now() - c[2])
    if remaining <= 0:
        return None
    return (int(c[0]), int(c[1]), int(remaining))


def cast_history(agent_id: int):
    """The agent's recent casts as ``[(seq, skill_id, status), ...]`` (most-recent last, <= _HIST_MAX),
    for the encoder to ship as ``AgentView.cast_history``. Empty when nothing observed. ``status`` is a
    ``CastStatus`` value (Casting / Success / Interrupted)."""
    lst = _history.get(int(agent_id))
    if not lst:
        return []
    return [(int(seq), int(skill), int(status)) for (seq, skill, status, _ts) in lst]


__all__ = ["pump", "reset", "remaining_cast_time_ms", "casting_skill_id",
           "casting_target_id", "cast_state", "cast_history"]
