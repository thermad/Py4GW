"""The WIRE CONTRACT between thin clients and the (eventually standalone) decision server.

This module is the single source of truth for what a client observes and ships up, and what an
order looks like coming back down. It is deliberately **pure**: dataclasses + stdlib enums, nothing
else.

    CARDINAL INVARIANT -- this file MUST NOT import Py4GW / Py4GWCoreLib / HeroAI / anything that
    reaches game memory. The whole point of the light-client/standalone-server split is that the
    server depends on THIS schema, never on the memory-interface library. A future client can be
    rebuilt against whatever game-interface library exists then, as long as it fills these structs.
    The import ban is asserted at the bottom of the file and covered by a test.

Fields are game-semantic (agent ids, world positions, hp fractions, ms remaining), never
Py4GW-shaped. Units are baked into names (``*_ms``) per the CLAUDE.md units footgun.

RATE CLASSES
------------
Every field is tagged with how often it actually needs to travel, via ``rate(...)``. This is what
lets the schema be *comprehensive* without being *expensive*: the encoder/delta layer consults the
tag to decide what goes in the hot per-frame stream vs. the cold keyframe vs. what never travels at
all.

    HOT     changes continuously / reactively; goes in the every-frame delta stream. Continuous
            HOT fields (pos, hp) additionally get a deadband so sub-threshold jitter is culled.
    COLD    changes rarely (identity, professions, map, weapon); sent on keyframe + on-change only.
    STATIC  immutable global game data (everything derivable from a skill id or model id). NEVER
            travels -- the server holds it in a local table keyed by id. Present here only so the
            schema documents the full surface; encoders skip STATIC fields entirely.
"""
from dataclasses import dataclass, field, fields
from enum import IntEnum
from typing import Dict, List, Optional, Any


PROTOCOL_VERSION = 1


# --------------------------------------------------------------------------- rate-class tagging
HOT = "hot"        # per-frame delta stream (continuous ones also deadbanded)
COLD = "cold"      # keyframe + on-change
STATIC = "static"  # never on the wire; server-side table keyed by id


def rate(default: Any, cls: str, deadband: float = 0.0, doc: str = ""):
    """Declare a schema field with its rate class. ``deadband`` (HOT continuous fields only) is the
    minimum change below which the delta encoder treats the field as unchanged. ``default_factory``
    is used for mutable defaults (dict/list) by passing a zero-arg callable as ``default``."""
    meta = {"rate": cls, "deadband": deadband, "doc": doc}
    if callable(default):
        return field(default_factory=default, metadata=meta)
    return field(default=default, metadata=meta)


def field_rate(f) -> str:
    return f.metadata.get("rate", HOT)


def field_deadband(f) -> float:
    return float(f.metadata.get("deadband", 0.0))


# --------------------------------------------------------------------------- semantic enums
# Mirror the GW/Py4GW integer meanings so the SERVER can reason symbolically without importing
# Py4GW. The client adapter is responsible for mapping the game lib's values onto these.
class Allegiance(IntEnum):
    Unknown = 0
    Ally = 1
    Neutral = 2
    Enemy = 3
    SpiritPet = 4
    Minion = 5
    NpcMinipet = 6


class Profession(IntEnum):
    NoProfession = 0
    Warrior = 1
    Ranger = 2
    Monk = 3
    Necromancer = 4
    Mesmer = 5
    Elementalist = 6
    Assassin = 7
    Ritualist = 8
    Paragon = 9
    Dervish = 10


class MessageKind(IntEnum):
    KEYFRAME = 0   # full Snapshot; sent on connect + slow heartbeat + on server request
    DELTA = 1      # changed fields only, relative to the last acked snapshot


class CastStatus(IntEnum):
    """Outcome of an observed cast, straight from the native combat-event stream (see the client's
    cast_observer). The three states map 1:1 to the world-view border colours."""
    Casting = 0       # SKILL_ACTIVATED seen, no end event yet (in progress)
    Success = 1       # SKILL_FINISHED / ATTACK_SKILL_FINISHED (or an instant skill) -- completed
    Interrupted = 2   # SKILL_STOPPED / ATTACK_SKILL_STOPPED / INTERRUPTED -- cancelled or interrupted


# --------------------------------------------------------------------------- effects / skills
@dataclass
class EffectView:
    """An active effect on an agent, keyed upstream by skill_id. STATIC facts about the effect's
    skill (type, whether it's a hex/enchant, base duration) live in the server's skill table."""
    skill_id: int = rate(0, HOT)
    effect_id: int = rate(0, HOT)
    attribute_level: int = rate(0, HOT)
    agent_id: int = rate(0, HOT)
    duration: float = rate(0.0, HOT)
    time_remaining_ms: int = rate(0, HOT, deadband=250,
                                  doc="ms until expiry; deadbanded -- a 250ms drift isn't worth a resend")


@dataclass
class BuffView:
    """A maintained buff (party-window style), keyed upstream by skill_id."""
    skill_id: int = rate(0, HOT)
    buff_id: int = rate(0, HOT)
    target_agent_id: int = rate(0, HOT)


@dataclass
class SkillSlotView:
    """DYNAMIC per-slot bar state only. Everything about WHAT the skill does (recharge base,
    activation, energy cost, type, AoE range, all the Is* predicates) is STATIC -- the server looks
    it up by ``skill_id``. get_recharge is remaining cooldown in MILLISECONDS (0 == ready)."""
    slot: int = rate(0, COLD)
    skill_id: int = rate(0, COLD)
    get_recharge_ms: int = rate(0, HOT, deadband=200, doc="remaining cooldown; 0 == ready")
    adrenaline_a: int = rate(0, HOT)
    adrenaline_b: int = rate(0, HOT)
    event: int = rate(0, HOT)


@dataclass
class CastRecord:
    """One entry in an agent's recent-cast history (client-observed from the native combat-event
    queue). Carried upstream in ``AgentView.cast_history``, keyed by a client-assigned cast SEQUENCE
    number -- a keyed collection (not a list) so ordering + repeated skills survive the delta/tombstone
    codec (wire_delta replaces lists wholesale but merges keyed dicts). Defined BEFORE AgentView so the
    ``Dict[int, CastRecord]`` annotation resolves to a real type for the type-driven codec."""
    skill_id: int = rate(0, HOT)
    status: int = rate(int(CastStatus.Casting), HOT, doc="CastStatus enum")


# --------------------------------------------------------------------------- agents
@dataclass
class AgentView:
    """Everything the decision layer can observe about ONE agent in the shared instance. Filled by
    the client adapter from its game lib. The server reads ONLY this -- never a live game getter."""
    agent_id: int = rate(0, COLD)

    # --- identity / classification (COLD: fixed for the agent's lifetime) ---
    allegiance: int = rate(int(Allegiance.Unknown), COLD, doc="Allegiance enum")
    model_id: int = rate(0, COLD, doc="species/NPC discriminator; keys STATIC per-model data")
    player_number: int = rate(0, COLD)
    login_number: int = rate(0, COLD, doc="!=0 => a human player agent")
    owner_id: int = rate(0, COLD, doc="minion/pet owner; 0 if none")
    profession: int = rate(int(Profession.NoProfession), COLD)
    secondary_profession: int = rate(int(Profession.NoProfession), COLD)
    level: int = rate(0, COLD)
    name: str = rate("", COLD, doc="resolved client-side (async); may lag a few ticks")
    is_player: bool = rate(False, COLD)
    is_npc: bool = rate(False, COLD)
    is_spirit: bool = rate(False, COLD)
    is_pet: bool = rate(False, COLD)
    is_minion: bool = rate(False, COLD)
    is_fleshy: bool = rate(False, COLD, doc="corpse-exploitation eligibility")
    is_spawned: bool = rate(False, HOT, doc="distinguishes spawned spirits from pets; hex-on-spirit guard")
    is_exploitable_corpse: bool = rate(False, HOT, doc="dead + not yet exploited; minion/corpse skills")
    guild_id: int = rate(0, COLD)
    team_id: int = rate(0, COLD)
    has_boss_glow: bool = rate(False, COLD)

    # --- position / motion (HOT continuous: deadbanded) ---
    x: float = rate(0.0, HOT, deadband=15.0)
    y: float = rate(0.0, HOT, deadband=15.0)
    z: float = rate(0.0, HOT, deadband=15.0)
    zplane: int = rate(0, HOT)
    rotation_angle: float = rate(0.0, HOT, deadband=0.15)
    velocity_x: float = rate(0.0, HOT, deadband=5.0)
    velocity_y: float = rate(0.0, HOT, deadband=5.0)

    # --- vitals (HOT continuous: deadbanded fractions) ---
    hp: float = rate(0.0, HOT, deadband=0.02, doc="0..1 fraction")
    max_hp: int = rate(0, COLD)
    hp_regen: float = rate(0.0, HOT, deadband=0.05)
    energy: float = rate(0.0, HOT, deadband=0.02, doc="0..1 fraction")
    max_energy: int = rate(0, COLD)
    energy_regen: float = rate(0.0, HOT, deadband=0.05)
    overcast: float = rate(0.0, HOT, deadband=0.02)

    # --- combat state (HOT discrete: culled, no deadband) ---
    is_alive: bool = rate(True, HOT)
    is_dead: bool = rate(False, HOT)
    is_moving: bool = rate(False, HOT)
    is_attacking: bool = rate(False, HOT)
    is_casting: bool = rate(False, HOT)
    casting_skill_id: int = rate(0, HOT, doc="skill currently being cast, 0 = none")
    remaining_cast_time_ms: int = rate(0, HOT, deadband=100)
    is_idle: bool = rate(False, HOT)
    is_knocked_down: bool = rate(False, HOT)
    knockdown_time_remaining_ms: int = rate(0, HOT, deadband=100)
    in_combat_stance: bool = rate(False, HOT)
    is_aggressive: bool = rate(False, HOT)
    can_act: bool = rate(True, HOT)
    is_holding_item: bool = rate(False, HOT, doc="carrying a bundle/item; Clamor_of_Souls reads it per-agent")
    dagger_status: int = rate(0, HOT, doc="per-(attacker,target); only meaningful on the observer's target")

    # --- conditions / hexes / enchants (HOT discrete: culled) ---
    is_bleeding: bool = rate(False, HOT)
    is_conditioned: bool = rate(False, HOT)
    is_crippled: bool = rate(False, HOT)
    is_deep_wounded: bool = rate(False, HOT)
    is_poisoned: bool = rate(False, HOT)
    is_enchanted: bool = rate(False, HOT)
    is_hexed: bool = rate(False, HOT)
    is_degen_hexed: bool = rate(False, HOT)
    is_weapon_spelled: bool = rate(False, HOT)

    # --- targeting relationships (HOT) ---
    target_id: int = rate(0, HOT, doc="who this agent is targeting")
    attack_target_id: int = rate(0, HOT)
    casting_target_id: int = rate(0, HOT)
    targeted_by: List[int] = rate(list, HOT, doc="agent ids currently targeting this agent (threat)")

    # --- weapon (COLD semi-static) ---
    weapon_type: int = rate(0, COLD)
    is_martial: bool = rate(False, COLD)
    is_caster: bool = rate(False, COLD)
    is_melee: bool = rate(False, COLD)
    is_ranged: bool = rate(False, COLD)

    # --- active effects / attributes (HOT collections: delta by key + tombstones) ---
    effects: Dict[int, EffectView] = rate(dict, HOT, doc="keyed by skill_id")
    buffs: Dict[int, BuffView] = rate(dict, HOT, doc="keyed by skill_id")
    attributes: Dict[int, int] = rate(dict, HOT, doc="buffed level keyed by attribute id")
    observed_skillbar: List[int] = rate(list, COLD, doc="inferred enemy bar (skill ids), when known")
    cast_history: Dict[int, CastRecord] = rate(dict, HOT, doc="recent casts keyed by client cast-seq; "
                                               "client-observed outcomes (CastStatus). Drives the "
                                               "world-view per-agent cast bar.")


@dataclass
class SelfView(AgentView):
    """The playing character. Adds self-only state the client can read about ITSELF that isn't
    observable for arbitrary agents -- its own bar, its ping, its held item, its instance context."""
    skillbar: Dict[int, SkillSlotView] = rate(dict, HOT, doc="keyed by skill_id")
    skillbar_disabled: bool = rate(False, HOT)
    skillbar_casting: int = rate(0, HOT)
    ping_ms: int = rate(0, HOT, deadband=15)
    controlled_minion_count: int = rate(0, HOT, doc="printer-relevant minion roster size")
    fast_casting: int = rate(0, COLD, doc="Fast Casting attribute level; feeds interrupt feasibility")
    party_leader_login: int = rate(0, COLD, doc="login_number of THIS client's party leader; the "
                                   "host derives PartyView leader fields from the feeds' consensus")


# --------------------------------------------------------------------------- instance / party
@dataclass
class MapView:
    """Which instance the client is in. Identifies a unique instance for the converge-and-party
    flow. COLD: only changes on a zone."""
    map_id: int = rate(0, COLD)
    region: int = rate(0, COLD)
    district: int = rate(0, COLD)
    language: int = rate(0, COLD)
    is_outpost: bool = rate(False, COLD)
    party_number: int = rate(0, COLD)
    instance_uptime_ms: int = rate(0, COLD)


# --------------------------------------------------------------------------- envelope + messages
@dataclass
class PartyMember:
    """One slot in the party roster. The login->agent->name mapping the host reads via GW.Party.*
    (GetPlayers / GetPlayerNameByLoginNumber / GetAgentIDByLoginNumber) must travel so the server can
    resolve party topology without a game getter."""
    login_number: int = rate(0, COLD)
    agent_id: int = rate(0, COLD)
    name: str = rate("", COLD)
    is_leader: bool = rate(False, COLD)
    called_target_id: int = rate(0, HOT, doc="agent id this member is currently CALLING/pinging (0 = "
                                             "none); party-shared, so a client can read a non-client "
                                             "human's call. Drives the CalledTarget focus goal.")


@dataclass
class PartyView:
    """Party topology. Replaces the scattered GW.Party.* reads in behaviors.py."""
    leader_agent_id: int = rate(0, COLD)
    leader_login_number: int = rate(0, COLD)
    members: Dict[int, PartyMember] = rate(dict, COLD, doc="keyed by login_number")


@dataclass
class Snapshot:
    """The full upstream payload from one client: itself + everything it observes. A KEYFRAME carries
    a complete Snapshot; a DELTA carries the same shape with only changed fields present plus removal
    tombstones (see wire_delta)."""
    me: SelfView = rate(SelfView, COLD)
    world: Dict[int, AgentView] = rate(dict, HOT, doc="observed agents keyed by agent_id")
    map: MapView = rate(MapView, COLD)
    party: PartyView = rate(PartyView, COLD)


@dataclass
class Envelope:
    """Wraps every message on the wire. ``seq`` + ``t_ms`` let the server order, detect staleness,
    and reason about temporal skew across clients; ``kind`` selects full-vs-delta decoding."""
    protocol_version: int = PROTOCOL_VERSION
    client_id: str = ""          # stable client identity (NOT agent_id)
    seq: int = 0                 # monotonically increasing per client
    t_ms: int = 0                # client clock at capture
    kind: int = int(MessageKind.KEYFRAME)
    body: Optional[dict] = None  # encoded Snapshot (keyframe) or delta payload


# --------------------------------------------------------------------------- schema introspection
def iter_wire_fields(dc_type):
    """Yield (name, field) for every field of a schema dataclass that actually travels (rate !=
    STATIC). Encoders/delta use this so STATIC fields are structurally impossible to put on the
    wire."""
    for f in fields(dc_type):
        if field_rate(f) != STATIC:
            yield f.name, f


# --------------------------------------------------------------------------- import-ban guard
def _assert_no_game_imports():
    """Fail loudly if this module ever transitively pulls in the game-interface library. Keeps the
    contract honest -- see the CARDINAL INVARIANT at the top."""
    import sys
    banned = ("Py4GW", "Py4GWCoreLib", "PyEffects", "PySkill", "PySkillbar", "HeroAI")
    leaked = [m for b in banned for m in sys.modules
              if m == b or m.startswith(b + ".")]
    if leaked:
        raise ImportError(
            "protocol.py must stay game-lib-free but these modules are loaded in its process: "
            + ", ".join(sorted(set(leaked)))
            + ". A schema field or helper is reaching into game memory -- move it to the client "
              "encoder (encode.py) instead.")


__all__ = [
    "PROTOCOL_VERSION", "HOT", "COLD", "STATIC", "rate", "field_rate", "field_deadband",
    "Allegiance", "Profession", "MessageKind", "CastStatus",
    "EffectView", "BuffView", "SkillSlotView", "CastRecord", "AgentView", "SelfView", "MapView",
    "PartyMember", "PartyView", "Snapshot", "Envelope", "iter_wire_fields",
]
