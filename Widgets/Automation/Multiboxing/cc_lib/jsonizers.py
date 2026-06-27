"""JSON (de)serializers for GW blackboard data -- hot-reloadable cc_lib leaf.

Depends only on Py4GWCoreLib / PyEffects; nothing else in cc_lib."""
from typing import Dict, Any

import PyEffects
import Py4GWCoreLib as GW


class Jsonizer:
    @staticmethod
    def effect_from_dict(data: Dict[str, Any]):
        return GW.PyEffects.EffectType(
            skill_id=int(data.get("skill_id", 0)),
            attribute_level=int(data.get("attribute_level", 0)),
            effect_id=int(data.get("effect_id", 0)),
            agent_id=int(data.get("agent_id", 0)),
            duration=float(data.get("duration", 0.0)),
            timestamp=int(data.get("timestamp", 0)),
            time_elapsed=int(data.get("time_elapsed", 0)),
            time_remaining=int(data.get("time_remaining", 0)),
        )

    @staticmethod
    def effect_to_dict(effect) -> Dict[str, Any]:
        return {
            "skill_id": effect.skill_id,
            "attribute_level": effect.attribute_level,
            "effect_id": effect.effect_id,
            "agent_id": effect.agent_id,
            "duration": effect.duration,
            "timestamp": effect.timestamp,
            "time_elapsed": effect.time_elapsed,
            "time_remaining": effect.time_remaining,
        }

    @staticmethod
    def buff_to_dict(buff: PyEffects.BuffType) -> Dict[str, Any]:
        return {
            "skill_id": buff.skill_id,
            "buff_id": buff.buff_id,
            "target": buff.target_agent_id
        }

    @staticmethod
    def get_buffs(id):
        buffs = GW.Effects.GetBuffs(id)
        json_buffs: Dict[str, Any] = dict()
        for b in buffs:
            json_buffs[b.skill_id] = Jsonizer.buff_to_dict(b)
        return json_buffs

    @staticmethod
    def get_effects(id: int) -> Dict[int, Any]:
        effects = GW.Effects.GetEffects(id)
        json_effects: Dict[int, Any] = dict()
        for e in effects:
            json_effects[int(e.skill_id)] = Jsonizer.effect_to_dict(e)
        return json_effects

    @staticmethod
    def skillbarskill_to_dict(skill: GW.PySkillbar.SkillbarSkill) -> Dict[str, Any]:
        return {
            "id": int(skill.id.id),  # SkillID → int
            "adrenaline_a": skill.adrenaline_a,
            "adrenaline_b": skill.adrenaline_b,
            "recharge": skill.recharge,
            "event": skill.event,
            "get_recharge": skill.get_recharge,
        }

    @staticmethod
    def skillbar_skill_from_dict(data: Dict[str, Any]):
        return GW.PySkillbar.SkillbarSkill(
            id=GW.PySkill(data.get("id", 0)),
            adrenaline_a=int(data.get("adrenaline_a", 0)),
            adrenaline_b=int(data.get("adrenaline_b", 0)),
            recharge=int(data.get("recharge", 0)),
            event=int(data.get("event", 0)),
        )

    @staticmethod
    def get_skilldata() -> Dict[int, Any]:
        d: Dict[int, Any] = dict()
        for i in range(1, 9):
            data = Jsonizer.skillbarskill_to_dict(GW.SkillBar.GetSkillData(i))
            d[data["id"]] = data
            d[data["id"]]["slot"] = i
        return d

    @staticmethod
    def player_data() -> Dict[str, Any]:
        d: Dict[str, any] = dict()
        d["id"] = GW.Player.GetAgentID()
        # Resolve our OWN character name here on the client. The host can't resolve a remote
        # client's name from this agent id -- the id only means something in the client's own
        # game instance -- so the name must travel over the wire. GetName() is async and may be
        # "" for the first few ticks until GW fills it in; the host keeps the last good value.
        d["name"] = GW.Player.GetName()
        d["target_id"] = GW.Player.GetTargetID()
        d["hp"] = GW.Agent.GetHealth(d["id"])
        d["max_hp"] = GW.Agent.GetMaxHealth(d["id"])
        d["energy"] = GW.Agent.GetEnergy(d["id"])
        d["max_energy"] = GW.Agent.GetMaxEnergy(d["id"])
        d["skilldata"] = Jsonizer.get_skilldata()
        d["effects"] = Jsonizer.get_effects(d["id"])
        d["buffs"] = Jsonizer.get_buffs(d["id"])
        return d

