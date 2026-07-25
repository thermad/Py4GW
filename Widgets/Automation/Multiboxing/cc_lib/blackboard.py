"""Typed blackboard: per-client game state mirrored from the wire -- hot-reloadable cc_lib leaf."""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SkillData:
    """get_recharge is the remaining cooldown in milliseconds"""
    id: int = 0
    slot: int = 0
    adrenaline_a: int = 0
    adrenaline_b: int = 0
    recharge: int = 0
    event: int = 0
    get_recharge: int = 0


@dataclass
class EffectData:
    skill_id: int = 0
    attribute_level: int = 0
    effect_id: int = 0
    agent_id: int = 0
    duration: float = 0
    timestamp: int = 0
    time_elapsed: int = 0
    time_remaining: int = 0


@dataclass
class BuffData:
    skill_id: int = 0
    buff_id: int = 0
    target: int = 0


class GameClient:
    def __init__(self):
        self.agent_id = 0
        self.name = ""           # character name, resolved client-side and sent over the wire
        self.target_id = 0
        # Dagger combo sequence on the client's current target (client-only state synced up so the
        # host can make the assassin lead->offhand->dual decision). 0=fresh/reset, 1=lead landed,
        # 2=offhand landed, 3=dual/chain complete.
        self.target_dagger_status = 0
        # Current map/instance, synced so the host can confirm a client arrived at its outpost
        # during the call-to-outpost+join flow. 0 until the first sync.
        self.map_id = 0
        self.map_region = 0
        self.map_district = 0
        self.map_language = 0
        self.hp = 0
        self.max_hp = 0
        self.energy = 0
        self.max_energy = 0
        # Caster attributes synced for host-side interrupt-feasibility evaluation.
        self.ping = 0
        self.fast_casting = 0
        # Current (buffed) attribute levels keyed by attribute id (e.g. Leadership == 40). Synced so
        # the host can read a remote client's attributes -- used by the Heroic Refrain bootstrap.
        self.attributes: Dict[int, int] = {}
        # Use dictionaries for these as they are dynamic sets of effects/skills
        self.skills: Dict[int, SkillData] = {}
        self.effects: Dict[int, EffectData] = {}
        self.buffs: Dict[int, BuffData] = {}

    def update_from_dict(self, data: Dict[str, Any]):
        """converts dictionary to object attributes"""
        self.agent_id = data.get("id", self.agent_id)
        # Keep the last non-empty name so a transient "" tick from the client doesn't blank the
        # host's label (which is what caused the row to flicker every frame).
        incoming_name = data.get("name", "")
        if incoming_name:
            self.name = incoming_name
        self.target_id = data.get("target_id", self.target_id)
        self.target_dagger_status = data.get("target_dagger_status", self.target_dagger_status)
        self.map_id = data.get("map_id", self.map_id)
        self.map_region = data.get("map_region", self.map_region)
        self.map_district = data.get("map_district", self.map_district)
        self.map_language = data.get("map_language", self.map_language)
        self.hp = data.get("hp", self.hp)
        self.max_hp = data.get("max_hp", self.max_hp)
        self.energy = data.get("energy", self.energy)
        self.max_energy = data.get("max_energy", self.max_energy)
        self.ping = data.get("ping", self.ping)
        self.fast_casting = data.get("fast_casting", self.fast_casting)
        # Update Attributes (clear+repopulate so a dropped buff's attribute boost can't linger as a
        # ghost -- same reasoning as effects/buffs below).
        self.attributes.clear()
        for a_id, a_val in data.get("attributes", {}).items():
            self.attributes[int(a_id)] = int(a_val)
        # print(f"""Skill update data {data.get("skilldata", {}).items()}""")
        # Update Skills
        self.skills.clear()
        for s_id, s_val in data.get("skilldata", {}).items():
            self.skills[int(s_id)] = SkillData(**s_val)
        # Update Effects
        self.effects.clear()
        for e_id, e_val in data.get("effects", {}).items():
            self.effects[int(e_id)] = EffectData(**e_val)
        # Update Buffs
        self.buffs.clear()
        for b_id, b_val in data.get("buffs", {}).items():
            self.buffs[int(b_id)] = BuffData(**b_val)

    def update_from_selfview(self, me, mp=None):
        """Populate this client's synced state from the wire ``SelfView`` (+ ``MapView``) -- the single
        upstream serialization path (protocol/encode), replacing the removed ``player_data`` dict. A
        straight pass-through so the decision layer sees the SAME values player_data used to sync. ``me``
        / ``mp`` are decoded protocol dataclasses (wire_delta.from_wire)."""
        self.agent_id = int(me.agent_id) or self.agent_id
        # Keep the last non-empty name so a transient "" tick doesn't blank the host's label.
        if me.name:
            self.name = me.name
        self.target_id = int(me.target_id)
        self.target_dagger_status = int(me.dagger_status)
        if mp is not None:
            self.map_id = int(mp.map_id)
            self.map_region = int(mp.region)
            self.map_district = int(mp.district)
            self.map_language = int(mp.language)
        # hp/energy are 0..1 fractions on the wire (SelfView convention; encode_self takes GetHealth/
        # GetEnergy directly). Store them straight -- the decision layer's `gc.hp/gc.max_hp` recovers
        # the same fraction it always did (this matches the value player_data synced).
        self.hp = me.hp
        self.max_hp = int(me.max_hp)
        self.energy = me.energy
        self.max_energy = int(me.max_energy)
        self.ping = int(me.ping_ms)
        self.fast_casting = int(me.fast_casting)
        # Rebuild the collections each update (clear+repopulate) so a dropped buff/effect/skill can't
        # linger as a ghost -- same reasoning as update_from_dict.
        self.attributes = {int(k): int(v) for k, v in me.attributes.items()}
        self.skills = {int(k): SkillData(id=int(v.skill_id), slot=int(v.slot),
                                         adrenaline_a=int(v.adrenaline_a), adrenaline_b=int(v.adrenaline_b),
                                         recharge=0, event=int(v.event),
                                         get_recharge=int(v.get_recharge_ms))
                       for k, v in me.skillbar.items()}
        self.effects = {int(k): EffectData(skill_id=int(v.skill_id), attribute_level=int(v.attribute_level),
                                           effect_id=int(v.effect_id), agent_id=int(v.agent_id),
                                           duration=v.duration, time_remaining=int(v.time_remaining_ms))
                        for k, v in me.effects.items()}
        self.buffs = {int(k): BuffData(skill_id=int(v.skill_id), buff_id=int(v.buff_id),
                                       target=int(v.target_agent_id))
                      for k, v in me.buffs.items()}

    def get_buff_data(self, b_id) -> BuffData:
        return self.buffs.get(b_id, None)

    def get_skill_data(self, s_id) -> SkillData:
        return self.skills.get(s_id, None)

    def get_attribute(self, attr_id) -> int:
        """Current level of an attribute (0 if absent/not synced). attr_id is the GW Attribute
        enum value, e.g. Leadership == 40."""
        return self.attributes.get(int(attr_id), 0)

