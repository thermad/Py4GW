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
        self.hp = 0
        self.max_hp = 0
        self.energy = 0
        self.max_energy = 0
        # Caster attributes synced for host-side interrupt-feasibility evaluation.
        self.ping = 0
        self.fast_casting = 0
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
        self.hp = data.get("hp", self.hp)
        self.max_hp = data.get("max_hp", self.max_hp)
        self.energy = data.get("energy", self.energy)
        self.max_energy = data.get("max_energy", self.max_energy)
        self.ping = data.get("ping", self.ping)
        self.fast_casting = data.get("fast_casting", self.fast_casting)
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

    def get_buff_data(self, b_id) -> BuffData:
        return self.buffs.get(b_id, None)

    def get_skill_data(self, s_id) -> SkillData:
        return self.skills.get(s_id, None)

