"""Behavior-tree / HFSM framework primitives -- the composable execution layer.

Status, Node, Task and the concrete nodes (CastSkill, MoveTo, Sequence, Parallel,
Wait/WaitUntil, TaskManager, State, StateMachine). Pure framework: depends only on the
lower cc_lib leaves (transport/rpc/misc_helpers) + Py4GWCoreLib. The owning Behavior is
referenced only via quoted forward-ref annotations, so there is no import cycle."""
import time
import random
from enum import Enum, auto
from typing import Optional, Callable, List, Dict

import Py4GWCoreLib as GW
from Py4GWCoreLib import Routines

from cc_lib.transport import Client
from cc_lib.rpc import RPC
from cc_lib.misc_helpers import Vec2


class Status(Enum):
    RUNNING = auto()
    SUCCESS = auto()
    FAILURE = auto()


class Node:
    """Uniform unit of behavior. Everything composable implements ``tick(ctx) -> Status``.
    ``ctx`` is the owning Behavior, which exposes the blackboard, the task manager and
    role assignments. Decision nodes (State, StateMachine, a future UtilitySelector),
    composites (Sequence) and execution leaves (Task, Wait) all share this contract, so
    they nest freely -- a Sequence of Tasks is itself just a Node.

    ``owner`` / ``status`` live here (not just on Task) so the TaskManager can hold and
    scope-cancel *any* node it is handed, leaf or composite. ``enter()`` re-initializes a
    node for a (re)run -- composites call it on each child before ticking it."""
    owner = None              # set by TaskManager.add; enables owner-scoped cancellation
    status: "Status" = Status.RUNNING

    def enter(self, ctx: "Behavior") -> None: pass
    def tick(self, ctx: "Behavior") -> "Status": return Status.RUNNING
    def exit(self, ctx: "Behavior") -> None: pass

    def cancel(self) -> None:
        """Hook called when this node is torn down (e.g. its owning state exits)."""
        pass


class Task(Node):
    """Execution leaf: performs one interval-gated step per tick and reports a Status.

    - ``guard()``  -> if it returns False while active, the task FAILs (precondition lost).
    - ``done()``   -> success predicate; with ``recurring=True`` a satisfied done() just
                      pauses the action (returns RUNNING) instead of completing.
    - no ``done()`` and not recurring -> one-shot: fire once, then SUCCESS.

    Tasks are deliberately dumb: they never decide *what* to do, only carry a thing out."""
    def __init__(self, name: str, action: Callable, *,
                 guard: Optional[Callable] = None, done: Optional[Callable] = None,
                 interval: float = 0.0, recurring: bool = False):
        self.name = name
        self.action = action
        self.guard = guard
        self.done = done
        self.interval = interval
        self.recurring = recurring
        self.owner = None              # set by TaskManager.add; enables scoped cancellation
        self.status = Status.RUNNING
        self._last_run = 0.0

    def tick(self, ctx: "Behavior") -> Status:
        now = time.time()
        if self.done is not None and self.done():
            return Status.RUNNING if self.recurring else Status.SUCCESS
        if self.guard is not None and not self.guard():
            return Status.FAILURE
        if now - self._last_run >= self.interval:
            try:
                self.action()
                self._last_run = now
            except Exception as e:
                print(f"Task '{self.name}' crashed: {e}")
                return Status.FAILURE
        if self.done is None and not self.recurring:
            return Status.SUCCESS
        return Status.RUNNING

    def enter(self, ctx: "Behavior") -> None:
        # Reset per-run state so a composite (e.g. Sequence) can re-run this task cleanly.
        self._last_run = 0.0
        self.status = Status.RUNNING


class DoUntil(Task):
    """Repeat ``action`` every ``interval`` seconds until ``until()`` is true, then SUCCESS."""
    def __init__(self, name: str, until: Callable, action: Callable, interval: float = 0.0):
        super().__init__(name, action, done=until, interval=interval)


class CastWaitForEffect(Task):
    """Cast ``skill_id`` on a client (slot resolved from that client's synced skillbar)
    repeatedly until the matching effect is observed on the target. Honors recharge."""
    def __init__(self, client: 'Client', skill_id: int, target_id: Optional[int] = None,
                 interval: float = 0.5):
        sid = int(skill_id)

        def act():
            sd = client.game_client.get_skill_data(sid)
            if sd is None or sd.slot <= 0 or sd.get_recharge != 0:
                return
            tgt = client.game_client.agent_id if target_id is None else target_id
            client.transport.send(RPC.CMD.USE_SKILL, sd.slot, tgt)

        def done():
            eff = client.game_client.effects.get(sid)
            return eff is not None and eff.time_remaining > 0

        super().__init__(f"cast {sid}->{client.game_client.agent_id}",
                         act, done=done, interval=interval)


# Default aftercast tacked onto the cast watch-window (HeroAI uses 250ms; res-like skills want more).
CAST_AFTERCAST = 0.25


class CastSkill(Node):
    """Reusable 'fire a skill and watch it land' leaf -- the general cast primitive. A plain
    fire-and-forget cast returns immediately, so a UtilitySelector re-picks and re-fires every tick;
    that is exactly what makes a caster interrupt its own cast to start another. CastSkill instead
    stays RUNNING from the moment it fires until the cast is observed to resolve, so the selector
    COMMITS to it (it only preempts on a strictly higher tier). One cast per pick, no thrash.

    ``fire`` issues the cast (self USE_SKILL, CAST_TARGETED, whatever) -- kept as a callable so this
    works for any cast path. After firing it waits half the activation time (so the cast has a beat
    to actually start before we judge it -- avoids a lag race that reads 'not casting' too early),
    then each tick checks:
      * skill now on cooldown               -> SUCCESS  (the cast went off; caster is free again)
      * caster no longer casting/attacking  -> SUCCESS  (nothing in flight; stop waiting)
      * 2*casttime + aftercast elapsed       -> FAILURE  (safety fallback so it can never hang)
    Activation comes from the static skill table (host-side); caster state is read from the shared
    instance by agent id, same as the combat scan. Recharge is read from the synced blackboard."""
    def __init__(self, client: 'Client', skill_id: int, fire: Callable[[], None],
                 aftercast: float = CAST_AFTERCAST):
        self.client = client
        self.skill_id = int(skill_id)
        self.fire = fire
        self.aftercast = aftercast
        self.name = f"cast {self.skill_id}->{client.game_client.agent_id}"
        self._casttime = 0.0
        self._t0 = 0.0
        self._fired = False

    def enter(self, ctx: "Behavior") -> None:
        self._fired = False
        self._t0 = 0.0
        try:
            self._casttime = float(GW.GLOBAL_CACHE.Skill.Data.GetActivation(self.skill_id) or 0.0)
        except Exception:
            self._casttime = 0.0

    def _caster_active(self) -> bool:
        """Is our caster mid-cast or attacking? Read host-side by agent id (shared instance)."""
        aid = self.client.game_client.agent_id
        if not aid:
            return False
        casting = attacking = False
        try:
            casting = bool(GW.Agent.IsCasting(aid))
        except Exception:
            pass
        try:
            attacking = bool(Routines.Checks.Agents.IsAttacking(aid))
        except Exception:
            pass
        return casting or attacking

    def _on_cooldown(self) -> bool:
        sd = self.client.game_client.get_skill_data(self.skill_id)
        return sd is not None and sd.get_recharge != 0

    def tick(self, ctx: "Behavior") -> Status:
        now = time.time()
        if not self._fired:
            self.fire()
            self._t0 = now
            self._fired = True
            return Status.RUNNING
        # Phase 1: give the cast half its activation to actually start before we judge it.
        if now - self._t0 < 0.5 * self._casttime:
            return Status.RUNNING
        # Phase 2: watch for completion / abort.
        if self._on_cooldown():
            return Status.SUCCESS                       # cast landed -> caster free
        if not self._caster_active():
            return Status.SUCCESS                       # nothing in flight -> stop waiting
        if now - self._t0 >= 2.0 * self._casttime + self.aftercast:
            return Status.FAILURE                       # safety fallback -> never hang
        return Status.RUNNING


class MoveTo(Task):
    """Move a client toward a world position until within ``tolerance``. By default adds
    jitter and a periodic nudge to escape stuck spots, mirroring the original move_to_wait.
    Pass ``jitter=False, nudge=False`` for a clean single move that walks straight to the
    point and stops -- needed for the rit, which must come to rest precisely to hold its
    minion ball rather than drift around (the jitter/nudge read as it taking several steps
    and shedding aggro)."""
    def __init__(self, client: 'Client', target: 'Vec2', tolerance: float = 100.0,
                 interval: float = 0.1, jitter: bool = True, nudge: bool = True):
        self._n = 0

        def act():
            self._n += 1
            if nudge and self._n % 15 == 0:
                client.transport.send(RPC.CMD.RELATIVE_MOVE, 20, 20)
            if jitter:
                client.transport.send(RPC.CMD.MOVE,
                                      target.x + random.uniform(-20, 20),
                                      target.y + random.uniform(-20, 20))
            else:
                client.transport.send(RPC.CMD.MOVE, target.x, target.y)

        def done():
            pos = Vec2.from_tuple(GW.Agent.GetXY(client.game_client.agent_id))
            return (pos - target).magnitude() <= tolerance

        super().__init__(f"move {client.game_client.agent_id}", act, done=done, interval=interval)


class Sequence(Node):
    """Composite: run children in order. Advance to the next child when the current one
    returns SUCCESS; abort (FAILURE) if any child fails; complete SUCCESS after the last
    child (or wrap around when ``loop=True``). Children are re-``enter()``ed each time the
    sequence reaches them, so a looped sequence replays cleanly. This is how a multi-step
    action ("cast, wait, print, wait, res") is expressed without a hand-rolled step flag."""
    def __init__(self, name: str, children: List[Node], loop: bool = False):
        self.name = name
        self.children = children
        self.loop = loop
        self.i = 0

    def enter(self, ctx: "Behavior") -> None:
        self.i = 0
        if self.children:
            self.children[0].enter(ctx)

    def tick(self, ctx: "Behavior") -> Status:
        while self.i < len(self.children):
            st = self.children[self.i].tick(ctx)
            if st == Status.RUNNING:
                return Status.RUNNING
            if st == Status.FAILURE:
                return Status.FAILURE
            # SUCCESS: advance, entering the next child (or looping)
            self.i += 1
            if self.i < len(self.children):
                self.children[self.i].enter(ctx)
            elif self.loop:
                self.i = 0
                self.children[0].enter(ctx)
                return Status.RUNNING
        return Status.SUCCESS

    def cancel(self) -> None:
        if 0 <= self.i < len(self.children):
            self.children[self.i].cancel()


class Parallel(Node):
    """Composite: tick every child each frame; SUCCESS once they have ALL left RUNNING
    (FAILURE if any child fails). The single-Node equivalent of enqueuing a batch and
    waiting on ``group_done`` -- use it to run several leaves at once *inside* a Sequence
    (e.g. move both monks simultaneously as one step of a larger choreography)."""
    def __init__(self, name: str, children: List[Node]):
        self.name = name
        self.children = children
        self._status: Dict[int, Status] = {}

    def enter(self, ctx: "Behavior") -> None:
        self._status = {id(c): Status.RUNNING for c in self.children}
        for c in self.children:
            c.enter(ctx)

    def tick(self, ctx: "Behavior") -> Status:
        any_running = False
        for c in self.children:
            if self._status[id(c)] != Status.RUNNING:
                continue
            st = c.tick(ctx)
            self._status[id(c)] = st
            if st == Status.FAILURE:
                return Status.FAILURE
            if st == Status.RUNNING:
                any_running = True
        return Status.RUNNING if any_running else Status.SUCCESS

    def cancel(self) -> None:
        for c in self.children:
            c.cancel()


class Wait(Node):
    """Leaf: RUNNING until ``seconds`` have elapsed since entry, then SUCCESS. The clean
    way to space steps apart inside a Sequence."""
    def __init__(self, seconds: float):
        self.seconds = seconds
        self._end = 0.0

    def enter(self, ctx: "Behavior") -> None:
        self._end = time.time() + self.seconds

    def tick(self, ctx: "Behavior") -> Status:
        return Status.SUCCESS if time.time() >= self._end else Status.RUNNING


class WaitUntil(Node):
    """Leaf: RUNNING until ``predicate()`` is true, then SUCCESS. A passive 'catch' step
    (no action of its own) -- e.g. confirm a target revived before moving on."""
    def __init__(self, predicate: Callable[[], bool], name: str = "wait_until"):
        self.predicate = predicate
        self.name = name

    def tick(self, ctx: "Behavior") -> Status:
        return Status.SUCCESS if self.predicate() else Status.RUNNING


class TaskManager:
    """Parallel pool of in-flight Nodes (leaf Tasks or composites like Sequence). Each is
    tagged with the node that spawned it (its owner) so a state can have all of its work
    cancelled atomically when it exits."""
    def __init__(self):
        self.tasks: List[Node] = []

    def add(self, task: Node, owner=None) -> None:
        task.owner = owner
        task.status = Status.RUNNING
        self.tasks.append(task)

    def update(self, ctx: "Behavior") -> None:
        survivors: List[Node] = []
        for t in self.tasks:
            t.status = t.tick(ctx)
            if t.status == Status.RUNNING:
                survivors.append(t)
        self.tasks = survivors

    def cancel_owner(self, owner) -> None:
        kept = []
        for t in self.tasks:
            if t.owner is owner:
                t.cancel()
            else:
                kept.append(t)
        self.tasks = kept


class State(Node):
    """A mode in an HFSM. Override ``tick()`` to do work / enqueue tasks, and
    ``transitions()`` to return the next State when it is time to move on (None = stay)."""
    def transitions(self, ctx: "Behavior") -> Optional["State"]:
        return None


class StateMachine(Node):
    """Composite decision node driving one active State. HFSM via nesting: a State may
    itself hold or return a StateMachine. On every transition the outgoing state's tasks
    are cancelled, so no orphaned work survives a phase change."""
    def __init__(self, initial_factory: Callable[[], State]):
        self._initial_factory = initial_factory
        self.current: Optional[State] = None

    def _go(self, state: Optional[State], ctx: "Behavior") -> None:
        if self.current is not None:
            self.current.exit(ctx)
            ctx.tasks.cancel_owner(self.current)
        self.current = state
        if state is not None:
            state.enter(ctx)

    def enter(self, ctx: "Behavior") -> None:
        self._go(self._initial_factory(), ctx)

    def tick(self, ctx: "Behavior") -> Status:
        if self.current is None:
            return Status.SUCCESS
        nxt = self.current.transitions(ctx)
        if nxt is not None:
            self._go(nxt, ctx)
            if self.current is None:
                return Status.SUCCESS
        return self.current.tick(ctx)

    def exit(self, ctx: "Behavior") -> None:
        self._go(None, ctx)
