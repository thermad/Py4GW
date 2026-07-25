"""CLIENT-SIDE adapter: game memory -> ``protocol`` dataclasses. THE ONLY serialization module that
touches the game library.

This is the "thin client" half of the light-client/standalone-server split: it reads GW via
Py4GWCoreLib and produces pure ``protocol`` structs. Swap this file (and nothing else) to retarget a
different game-interface library in the future -- the server, wire, and decision layer never change.

MUST run on a GW-legal thread (main thread, or a map-gated behavior worker) per the CLAUDE.md
threading rules -- every getter here reads game memory.

Every read is wrapped so a single bad getter degrades one field to its default rather than dropping
the whole agent.

    SHARED MEMORY MUST NOT CROSS THE SOCKET. A client encodes only (1) its own PLAYER-observable
    state (``encode_self``: own effects/buffs/skillbar/attributes/vitals) and (2) AGENT-observable
    data for every agent it sees (``Agent.*`` living-struct reads). It NEVER reads the HeroAI
    shared-memory party region for other accounts -- so NO ``Routines.Checks.Agents.*`` (those prefer
    the shared party data for party members) and NO ``Effects.GetEffects/GetBuffs`` for anyone but
    self. A party member's effects reach the host only via THAT member's own SelfView; the host
    derives the party roster from the union of client self-views (see world_feed).

hp/energy convention: AgentView.hp/energy are 0..1 fractions. ``Agent.GetHealth`` is already a
fraction for non-party agents; ``encode_self`` normalizes the local player's raw hp by max. ``Agent.
GetEnergy`` is a fraction.
"""
import time

from Py4GWCoreLib import Agent, AgentArray
import Py4GW

from . import protocol as P
from . import cast_observer


def _try(fn, default):
    try:
        v = fn()
        return v if v is not None else default
    except Exception:
        return default


# --------------------------------------------------------------------------- struct-direct helpers
# Weapon-type id buckets (Py4GWCoreLib Weapon enum): 1=Bow 2=Axe 3=Hammer 4=Daggers 5=Scythe 6=Spear
# 7=Sword  8=Scepter 9=Scepter2 10=Wand 11=Staff1 12=Staff 13=Staff2 14=Staff3. Mirrors the name-set
# logic in Agent.IsMartial/IsMelee/IsCaster/IsRanged so wire output matches the facade.
_MARTIAL_WEAPONS = frozenset((1, 2, 3, 4, 5, 6, 7))
_MELEE_WEAPONS = frozenset((2, 3, 4, 5, 7))
_RANGED_WEAPONS = frozenset((1, 6))
_CASTER_WEAPONS = frozenset((8, 9, 10, 11, 12, 13, 14))


def _weapon_roles(weapon_type: int, is_pet: bool):
    """(is_martial, is_caster, is_melee, is_ranged) from weapon_type + pet status -- a struct-only
    replica of Agent.IsMartial/IsCaster/IsMelee/IsRanged. A pet is martial+melee (mirrors the facades'
    ``if IsPet: return True/False`` branch). NB: the facades' rare Illusionary_Weaponry override
    (a hexed martial reads as non-martial) is intentionally dropped -- it needs a per-agent effect
    lookup, and the only server consumer is enemy is_martial for cleave targeting; the divergence is
    limited to a Mesmer under one uncommon skill."""
    if is_pet:
        return True, False, True, False
    wt = int(weapon_type)
    if wt == 0:
        return False, False, False, False
    return (wt in _MARTIAL_WEAPONS, wt in _CASTER_WEAPONS,
            wt in _MELEE_WEAPONS, wt in _RANGED_WEAPONS)


# Map-static per-agent facts: name (async string decode), guild_id (tag pointer deref), is_fleshy
# (NPC-model table lookup). A name/guild/species never changes for an agent id within one instance,
# so decode ONCE and cache -- this removes the single genuinely expensive facade call (GetNameByID ->
# GetAgentEncName + decode) from the steady-state per-frame encode. The cache is cleared on map change
# (see _reset_static_if_zoned, driven from encode_snapshot).
_static_cache: dict = {}        # agent_id -> [name, guild_id, is_fleshy, resolved]
_static_map_id = None


def _static_fields(agent_id: int):
    """Return (name, guild_id, is_fleshy) for an agent from the map-static cache, decoding on first
    sighting. Name resolution is async client-side (may lag a few ticks), so keep retrying until it
    lands, then freeze the entry -- after that this is pure dict lookups, zero facade calls."""
    aid = int(agent_id)
    ent = _static_cache.get(aid)
    if ent is None:
        ent = ["", 0, False, False]
        _static_cache[aid] = ent
    if not ent[3]:
        if not ent[0]:
            ent[0] = str(_try(lambda: Agent.GetNameByID(aid), "") or "")
        ent[1] = int(_try(lambda: Agent.GetGuildID(aid), 0))
        ent[2] = bool(_try(lambda: Agent.IsFleshy(aid), False))
        if ent[0]:                      # consider resolved once the (async) name has landed
            ent[3] = True
    return ent[0], ent[1], ent[2]


def _reset_static_if_zoned(map_id: int) -> None:
    """Drop the map-static cache when the instance changes (agent ids are only stable within one
    instance)."""
    global _static_map_id
    if map_id != _static_map_id:
        _static_cache.clear()
        _static_map_id = map_id


# Self-name canary: the map-id zone check misses reloads that land back on the SAME instance id (a
# re-entered instance, a missed zone signal). When that happens the stale id->name cache survives, and
# GW reuses those agent ids for DIFFERENT agents in the new instance -> scrambled names. The client
# always knows its own true name, so poll it periodically: if the cached name for OUR OWN agent no
# longer matches what GW reports live, the whole cache is stale -> wipe it so every name re-decodes.
_SELF_NAME_CHECK_INTERVAL = 3.0     # seconds between self-name canary polls
_last_self_name_check = 0.0


def _verify_self_name(self_id: int) -> None:
    """Throttled staleness check keyed on our own agent's name. Clears the entire map-static cache when
    the cached self-name and the live self-name disagree. Guarded so an async/empty live read never
    trips a false renewal: both names must be non-empty to be compared."""
    global _last_self_name_check
    now = time.time()
    if now - _last_self_name_check < _SELF_NAME_CHECK_INTERVAL:
        return
    _last_self_name_check = now
    ent = _static_cache.get(int(self_id))
    if ent is None or not ent[0]:
        return                          # our name not resolved yet -> nothing trustworthy to compare
    live = str(_try(lambda: Agent.GetNameByID(int(self_id)), "") or "")
    if live and live != ent[0]:         # both sides real and disagree -> cache is stale, renew all
        _static_cache.clear()


# Pre-bucketed LIVING-type shared-memory arrays (never ItemArray/GadgetArray) -- the union is every
# living agent, so feeding only these to the living-struct encoder is type-safe by construction.
_LIVING_BUCKETS = ("GetAllyArray", "GetNeutralArray", "GetEnemyArray",
                   "GetSpiritPetArray", "GetMinionArray", "GetNPCMinipetArray")


def _living_agent_ids():
    """Union of the living-type bucket arrays (deduped by the caller). Dead agents remain in their
    allegiance bucket (GW keeps them until the corpse despawns), so corpses/exploitable bodies are
    still included."""
    ids = []
    for getter in _LIVING_BUCKETS:
        ids.extend(_try(lambda g=getter: list(getattr(AgentArray, g)()), []))
    return ids


# --------------------------------------------------------------------------- effects / skills
def encode_effects(agent_id: int):
    out = {}
    for e in _try(lambda: list(Effects_GetEffects(agent_id)), []):
        try:
            out[int(e.skill_id)] = P.EffectView(
                skill_id=int(e.skill_id), effect_id=int(e.effect_id),
                attribute_level=int(e.attribute_level), agent_id=int(e.agent_id),
                duration=float(e.duration), time_remaining_ms=int(e.time_remaining))
        except Exception:
            continue
    return out


def encode_buffs(agent_id: int):
    out = {}
    for b in _try(lambda: list(Effects_GetBuffs(agent_id)), []):
        try:
            out[int(b.skill_id)] = P.BuffView(
                skill_id=int(b.skill_id), buff_id=int(b.buff_id),
                target_agent_id=int(b.target_agent_id))
        except Exception:
            continue
    return out


def encode_attributes(agent_id: int):
    out = {}
    for a in _try(lambda: list(Agent.GetAttributes(agent_id)), []):
        try:
            lvl = int(a.level)
            if lvl > 0:
                out[int(a.attribute_id)] = lvl
        except Exception:
            continue
    return out


def encode_skillbar(agent_id: int):
    out = {}
    for slot in range(1, 9):
        try:
            from Py4GWCoreLib import GLOBAL_CACHE
            s = GLOBAL_CACHE.SkillBar.GetSkillData(slot)
            sid = int(s.id.id)
            out[sid] = P.SkillSlotView(
                slot=slot, skill_id=sid, get_recharge_ms=int(s.get_recharge),
                adrenaline_a=int(s.adrenaline_a), adrenaline_b=int(s.adrenaline_b),
                event=int(s.event))
        except Exception:
            continue
    return out


# Effects/Buffs live on the Effects facade; import lazily so a rename there can't break module load.
def Effects_GetEffects(agent_id):
    from Py4GWCoreLib import Effects
    return Effects.GetEffects(agent_id)


def Effects_GetBuffs(agent_id):
    from Py4GWCoreLib import Effects
    return Effects.GetBuffs(agent_id)


# --------------------------------------------------------------------------- agents
def encode_agent(agent_id: int) -> P.AgentView:
    """STRUCT-DIRECT observation of ONE living agent. Fetches the living struct ONCE
    (``Agent.GetLivingAgentByID`` -- a zero-copy ctypes overlay over GW's live memory) and reads every
    wire field straight off it, instead of the dozens of per-field ``Agent.GetXxx`` facade calls the
    old split path made (each re-fetched the same struct and re-derived its value). Same
    AGENT-observable data as before -- never the HeroAI shared-memory party region; a party member's
    own effects/attributes/bar travel only in ITS OWN SelfView (encode_self), never encoded here.

    TYPE SAFE: ``GetLivingAgentByID`` is guarded by ``is_living_type`` and returns None for
    items/gadgets, so a non-living pointer is never mis-cast to the living layout. Callers pass only
    living-type agents (see ``_living_agent_ids``); a None struct yields a defaulted view.

    The map-static facts (name/guild/fleshy) come from ``_static_fields`` (cached per instance) and the
    dead-stub facades in this Py4GW build (GetTarget/GetAttackTarget/GetAgetsTargeting/CanAct/
    GetKnockDownTimeRemaining -> constants) are NOT called -- their AgentView fields keep the dataclass
    defaults they carried before."""
    aid = int(agent_id)
    a = P.AgentView(agent_id=aid)
    living = _try(lambda: Agent.GetLivingAgentByID(aid), None)
    if living is None:
        return a   # non-living / momentarily unavailable -> defaults (caller feeds living agents only)

    # --- identity / classification (all direct struct reads) ---
    alleg = int(_try(lambda: int(living.allegiance), int(P.Allegiance.Unknown)))
    is_spawned = bool(_try(lambda: bool(living.is_spawned), False))
    a.allegiance = alleg
    a.is_spawned = is_spawned
    # is_spirit/is_pet/is_minion mirror Agent.IsSpirit/IsPet/IsMinion exactly (allegiance + spawned).
    a.is_spirit = (alleg == int(P.Allegiance.SpiritPet) and is_spawned)
    a.is_pet = (alleg == int(P.Allegiance.SpiritPet) and not is_spawned)
    a.is_minion = (alleg == int(P.Allegiance.Minion))
    a.player_number = int(_try(lambda: int(living.player_number), 0))
    a.model_id = a.player_number                       # Agent.GetModelID == living.player_number
    a.login_number = int(_try(lambda: int(living.login_number), 0))
    a.owner_id = int(_try(lambda: int(living.owner), 0))
    a.profession = int(_try(lambda: int(living.primary), 0))
    a.secondary_profession = int(_try(lambda: int(living.secondary), 0))
    a.level = int(_try(lambda: int(living.level), 0))
    a.team_id = int(_try(lambda: int(living.team_id), 0))
    a.is_player = bool(_try(lambda: bool(living.is_player), False))
    a.is_npc = bool(_try(lambda: bool(living.is_npc), False))
    a.has_boss_glow = bool(_try(lambda: bool(living.has_boss_glow), False))
    a.is_exploitable_corpse = bool(_try(lambda: bool(living.is_exploitable), False))
    # map-static, cached (expensive to derive): name / guild_id / is_fleshy.
    a.name, a.guild_id, a.is_fleshy = _static_fields(aid)

    # --- position / motion (base AgentStruct: pos GamePos{x,y,zplane}, z, rotation, velocity Vec2f) ---
    pos = _try(lambda: living.pos, None)
    if pos is not None:
        a.x = float(_try(lambda: float(pos.x), 0.0))
        a.y = float(_try(lambda: float(pos.y), 0.0))
        a.zplane = int(_try(lambda: int(pos.zplane), 0))
    a.z = float(_try(lambda: float(living.z), 0.0))
    a.rotation_angle = float(_try(lambda: float(living.rotation_angle), 0.0))
    vel = _try(lambda: living.velocity, None)
    if vel is not None:
        a.velocity_x = float(_try(lambda: float(vel.x), 0.0))
        a.velocity_y = float(_try(lambda: float(vel.y), 0.0))

    # --- vitals (living.hp/energy are already 0..1 fractions -- the AgentView convention) ---
    a.hp = float(_try(lambda: float(living.hp), 0.0))
    a.max_hp = int(_try(lambda: int(living.max_hp), 0))
    a.hp_regen = float(_try(lambda: float(living.hp_pips), 0.0))
    a.energy = float(_try(lambda: float(living.energy), 0.0))
    a.max_energy = int(_try(lambda: int(living.max_energy), 0))
    a.energy_regen = float(_try(lambda: float(living.energy_regen), 0.0))
    a.overcast = float(_try(lambda: float(living.h0128), 0.0))

    # --- combat state (struct properties) ---
    a.is_alive = bool(_try(lambda: bool(living.is_alive), True))
    a.is_dead = bool(_try(lambda: bool(living.is_dead), False))
    a.is_moving = bool(_try(lambda: bool(living.is_moving), False))
    a.is_attacking = bool(_try(lambda: bool(living.is_attacking), False))
    a.is_idle = bool(_try(lambda: bool(living.is_idle), False))
    a.is_knocked_down = bool(_try(lambda: bool(living.is_knocked_down), False))
    a.in_combat_stance = bool(_try(lambda: bool(living.is_in_combat_stance), False))
    a.dagger_status = int(_try(lambda: int(living.dagger_status), 0))
    a.is_holding_item = bool(_try(lambda: int(living.weapon_type) == 0, False))
    # Casting: prefer the cast observer (event-queue remaining time -- GW has no live getter); fall
    # back to the living-struct casting flag + current skill id when the cast start wasn't observed.
    _cs = cast_observer.cast_state(aid)
    if _cs is not None:
        a.is_casting = True
        a.casting_skill_id, a.casting_target_id, a.remaining_cast_time_ms = _cs
    else:
        a.is_casting = bool(_try(lambda: bool(living.is_casting), False))
        a.casting_skill_id = int(_try(lambda: int(living.skill), 0)) if a.is_casting else 0
        a.remaining_cast_time_ms = 0
    a.is_aggressive = bool(a.is_attacking or a.is_casting)   # mirrors Agent.IsAggressive
    # Recent-cast history (authoritative outcomes from the combat-event queue) -- keyed by the
    # observer's cast seq so the delta/tombstone codec preserves order + prunes old entries.
    _hist = cast_observer.cast_history(aid)
    if _hist:
        a.cast_history = {seq: P.CastRecord(skill_id=sk, status=st) for (seq, sk, st) in _hist}

    # --- conditions / hexes / enchants (struct properties) ---
    a.is_bleeding = bool(_try(lambda: bool(living.is_bleeding), False))
    a.is_conditioned = bool(_try(lambda: bool(living.is_conditioned), False))
    a.is_crippled = bool(_try(lambda: bool(living.is_crippled), False))
    a.is_deep_wounded = bool(_try(lambda: bool(living.is_deep_wounded), False))
    a.is_poisoned = bool(_try(lambda: bool(living.is_poisoned), False))
    a.is_enchanted = bool(_try(lambda: bool(living.is_enchanted), False))
    a.is_hexed = bool(_try(lambda: bool(living.is_hexed), False))
    a.is_degen_hexed = bool(_try(lambda: bool(living.is_degen_hexed), False))
    a.is_weapon_spelled = bool(_try(lambda: bool(living.is_weapon_spelled), False))

    # --- weapon / role (weapon_type off the struct; predicates derived to match the facades) ---
    a.weapon_type = int(_try(lambda: int(living.weapon_type), 0))
    a.is_martial, a.is_caster, a.is_melee, a.is_ranged = _weapon_roles(a.weapon_type, a.is_pet)

    # NB: target_id / attack_target_id / targeted_by / can_act / knockdown_time_remaining_ms are left
    # at their dataclass defaults -- their facades are dead stubs (constant returns) in this build, so
    # calling them only ever reproduced the default. NO effect/buff list, attributes, or
    # observed_skillbar here (party-region/self-only paths that must not cross the socket).
    return a


def encode_self(agent_id: int) -> P.SelfView:
    """The playing character = this client's PLAYER-OBSERVABLE data: its own effects/buffs, bar,
    attribute levels, ping, held item, vitals. This is the client's own state (not read via the
    shared party region of OTHER accounts), so it is legitimate to send. In the host merge this
    SelfView is overlaid as the authoritative view of this party member."""
    aid = int(agent_id)
    base = encode_agent(aid)
    me = P.SelfView(**{name: getattr(base, name)
                       for name, _f in P.iter_wire_fields(P.AgentView)})
    # Own effects/buffs -- the local player's own active effects (player-observable).
    me.effects = encode_effects(aid)
    me.buffs = encode_buffs(aid)
    me.skillbar = encode_skillbar(aid)
    me.attributes = encode_attributes(aid)
    me.ping_ms = int(_try(lambda: Py4GW.PingHandler().GetCurrentPing(), 0))
    # Own vitals as 0..1 FRACTIONS (AgentView.hp/energy convention). Agent.GetHealth and Agent.GetEnergy
    # BOTH already return 0..1 fractions (living.hp / living.energy), so take them directly -- do NOT
    # divide by max_hp again (that was a double-normalization bug: hp landed as fraction/max).
    me.hp = float(_try(lambda: Agent.GetHealth(aid), 0.0))
    me.energy = float(_try(lambda: Agent.GetEnergy(aid), 0.0))
    # Fast Casting attribute (id 0), surfaced explicitly for the interrupt-feasibility check.
    me.fast_casting = int(me.attributes.get(0, 0))
    # Party leader as a login_number -- LOCAL party topology (not the shared-memory effects window),
    # so it is legitimate to send. The host derives PartyView.leader_* from the union of these across
    # feeds; shipping the leader's login (stable across map loads) rather than its agent id lets the
    # host resolve the agent id against the roster it already builds.
    from Py4GWCoreLib import Party
    me.party_leader_login = int(_try(
        lambda: Party.Players.GetLoginNumberByAgentID(Party.GetPartyLeaderID()), 0))
    return me


# --------------------------------------------------------------------------- party roster
def encode_party() -> P.PartyView:
    """The LOCAL party roster (every player slot: login_number -> agent_id + name + leader flag).

    This is LOCAL party topology -- the same legitimacy class as ``party_leader_login`` in
    ``encode_self`` -- NOT the shared-memory effects/buffs window of other accounts (which must never
    cross the socket). Shipping the full roster lets the host offer ANY party player as a follow target
    (including a human main who is not running a CentralCommander client), and re-resolve a pinned
    leader by NAME after a map load, without the host ever touching a GW party getter.

    Keyed by login_number (stable across map loads); ``agent_id`` is the CURRENT instance id and is
    re-read here every push, so the host always has a live login->agent mapping."""
    from Py4GWCoreLib import Party
    pv = P.PartyView()
    leader_aid = int(_try(lambda: Party.GetPartyLeaderID(), 0))
    for player in (_try(lambda: Party.GetPlayers(), []) or []):
        login = int(_try(lambda: int(player.login_number), 0))
        if not login:
            continue
        aid = int(_try(lambda: Party.Players.GetAgentIDByLoginNumber(login), 0))
        name = _try(lambda: Party.Players.GetPlayerNameByLoginNumber(login), "") or ""
        # The agent THIS member is currently calling/pinging (PlayerPartyMember.called_target_id). It
        # is party-shared, so every client reads the same value -- including a non-client human's call.
        called = int(_try(lambda: int(player.called_target_id), 0))
        pv.members[login] = P.PartyMember(login_number=login, agent_id=aid, name=name,
                                          is_leader=bool(aid and aid == leader_aid),
                                          called_target_id=called)
    pv.leader_agent_id = leader_aid
    pv.leader_login_number = int(_try(lambda: Party.Players.GetLoginNumberByAgentID(leader_aid), 0))
    return pv


# --------------------------------------------------------------------------- map / snapshot
def encode_map() -> P.MapView:
    from Py4GWCoreLib import Map
    m = P.MapView()
    m.map_id = int(_try(lambda: Map.GetMapID(), 0))
    m.region = int(_try(lambda: Map.GetRegion()[0], 0))
    m.district = int(_try(lambda: Map.GetDistrict(), 0))
    m.language = int(_try(lambda: Map.GetLanguage()[0], 0))
    m.is_outpost = bool(_try(lambda: Map.IsOutpost(), False))
    return m


def encode_snapshot(self_agent_id: int, observed_agent_ids=None) -> P.Snapshot:
    """Build ONE client's upstream Snapshot = its PLAYER-observable self (``me``, with own effects)
    + AGENT-observable data for every LIVING agent it sees (``world``) + the LOCAL party roster
    (``party``, see encode_party) + map. It sends NO other-agent effects -- those would require the
    shared-memory party region, which must not cross the socket. The party ROSTER (login/agent/name)
    is local party topology and IS sent, so the host knows every party player (including non-clients).

    The observed set defaults to the union of the LIVING-type shared-memory buckets
    (``_living_agent_ids``) -- items/gadgets are never encoded, which also guarantees the living-struct
    encoder only ever sees living agents. Every agent gets the single struct-direct encode (no
    ally/enemy split). ``self`` is not re-encoded into ``world`` (it IS ``me``). ``observed_agent_ids``
    may be passed explicitly (tests) to bypass the live bucket scan."""
    self_id = int(self_agent_id)
    m = encode_map()
    _reset_static_if_zoned(m.map_id)                     # drop map-static cache on a zone
    _verify_self_name(self_id)                           # ...and on a self-name mismatch (missed zone)
    snap = P.Snapshot(me=encode_self(self_id), map=m, party=encode_party(), world={})
    ids = _living_agent_ids() if observed_agent_ids is None else observed_agent_ids
    seen = {self_id}
    for aid in ids:
        aid = int(aid)
        if aid and aid not in seen:
            seen.add(aid)
            snap.world[aid] = encode_agent(aid)
    return snap
