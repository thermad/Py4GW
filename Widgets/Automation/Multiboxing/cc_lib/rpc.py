"""RPC registry + host-driven targeting.

Importing this module (re)registers every RPC method via the module-level RPC.clear()
+ register(...) block at the bottom, which is what keeps it hot-reload safe."""
from enum import Enum
from typing import Dict, Callable, Protocol, ParamSpec, TypeVar

import Py4GWCoreLib as GW
from Py4GWCoreLib import Routines

from cc_lib.jsonizers import Jsonizer


P = ParamSpec("P")
R = TypeVar("R")


class RPCMethod(Protocol[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        ...


class RPC:
    """This is a namespace class for all RPC stuff"""
    _methods: Dict[str, Callable[..., object]] = {}

    class CMD(str, Enum):
        MOVE = "move"
        RELATIVE_MOVE = "relmove"
        GETSKILLBAR = "getskillbar"
        GETEFFECTS = "geteffects"
        GET_PLAYER_ID = "getplayerid"
        GET_PLAYER_DATA = "get_player_data"
        USE_SKILL = "use_skill"
        DROP_BOND = "drop_bond"
        CHANGE_TARGET = "change_target"   # host sets a client's target to a concrete agent id it knows
        CAST_TARGETED = "cast_targeted"   # client resolves the target for a Skilltarget spec, then casts

    @classmethod
    def register(
            cls,
            name: CMD,
            method: RPCMethod[P, R],
    ) -> None:
        if name in cls._methods:
            raise ValueError(f"RPC method '{name}' already registered")
        cls._methods[name] = method

    @classmethod
    def clear(cls):
        """Useful for hot-reloading to prevent duplicate registration logic errors."""
        cls._methods.clear()

    @classmethod
    def _call(cls, name: str, args: tuple[object, ...], kwargs: dict[str, object]) -> object:
        try:
            method = cls._methods[name]
        except KeyError:
            raise KeyError(f"RPC method '{name}' not found")
        return method(*args, **kwargs)

    @classmethod
    def call(cls, name: CMD, *args: object, **kwargs: object) -> object:
        try:
            method = cls._methods[name]
        except KeyError:
            raise KeyError(f"RPC method '{name}' not found")
        return method(*args, **kwargs)

    # Static helpers
    @staticmethod
    def relative_move(x, y):
        _x, _y = GW.Player.GetXY()
        GW.Player.Move(_x + x, _y + y)

    # --- host-driven targeting (runs CLIENT-SIDE when the host invokes the RPC) ---
    @staticmethod
    def resolve_combat_target(target_spec: int, skill_id: int = 0) -> int:
        """Resolve the concrete agent id for a HeroAI ``Skilltarget`` spec in THIS client's own
        world -- a trimmed mirror of HeroAI ``CombatClass.GetAppropiateTarget``'s common branches.
        Returns 0 when nothing suitable is in range, which callers treat as 'do not cast'. That 0
        is the whole point: it lets the host drive skills whose precondition lives in the remote
        world it can't see (an interrupt needs an enemy actually casting; a heal needs a hurt ally)
        by delegating the existence/feasibility check to the client that can actually scan for it.

        Broad-but-basic on purpose: the common specs are handled and everything enemy-ish falls back
        to nearest enemy. Extend the dispatch as the generic engine needs finer targeting."""
        try:
            from HeroAI.types import Skilltarget as T
            from HeroAI import targeting as HT
            from Py4GWCoreLib import Range
        except Exception:
            # No HeroAI client-side -> best effort: keep whatever the client already targets.
            return int(GW.Player.GetTargetID() or 0)

        me = int(GW.Player.GetAgentID() or 0)
        spell = Range.Spellcast.value
        spec = int(target_spec)

        if spec == T.Self.value:
            return me
        if spec == T.Ally.value:
            return HT.TargetLowestAlly(filter_skill_id=skill_id)
        if spec == T.OtherAlly.value:
            return HT.TargetLowestAlly(other_ally=True, filter_skill_id=skill_id)
        if spec == T.AllyCaster.value:
            return HT.TargetLowestAllyCaster(filter_skill_id=skill_id) or HT.TargetLowestAlly(filter_skill_id=skill_id)
        if spec == T.AllyMartial.value:
            return HT.TargetLowestAllyMartial(filter_skill_id=skill_id) or HT.TargetLowestAlly(filter_skill_id=skill_id)
        if spec == T.DeadAlly.value:
            return HT.TargetDeadPartyMember(spell)
        if spec == T.ResurrectionAlly.value:
            return Routines.Agents.GetResurrectionTarget(spell, reserve=True, skill_id=skill_id)
        if spec == T.EnemyClustered.value:
            return (HT.TargetClusteredEnemy(spell, skill_id=skill_id, cluster_radius=Range.Earshot.value)
                    or Routines.Agents.GetNearestEnemy(spell))
        if spec in (T.EnemyCasting.value, T.EnemyCastingSpell.value, T.EnemyCastingSpellOrChant.value):
            return HT.GetEnemyCasting(spell)
        if spec == T.EnemyCaster.value:
            return Routines.Agents.GetNearestEnemyCaster(spell) or Routines.Agents.GetNearestEnemy(spell)
        if spec == T.Corpse.value:
            return Routines.Agents.GetNearestCorpse(spell)
        if spec == T.Minion.value:
            return Routines.Agents.GetLowestMinion(spell)
        if spec == T.Spirit.value:
            return Routines.Agents.GetNearestSpirit(spell)
        # Default / all remaining Enemy* specs: the called party target if any, else nearest enemy.
        return int(GW.Party.GetPartyTarget() or 0) or Routines.Agents.GetNearestEnemy(spell)

    @staticmethod
    def cast_targeted(slot: int, skill_id: int, target_spec: int) -> None:
        """CLIENT-SIDE: resolve ``target_spec`` locally, and only if a target exists change to it
        and fire ``slot``. The resolve+change+cast happen atomically in the client's own world, so
        there is no cross-network race between a separate ChangeTarget and the cast, and no reliance
        on the host learning the resolved id (the host ROUTER drops RPC return values by design)."""
        target_id = RPC.resolve_combat_target(int(target_spec), int(skill_id))
        if not target_id:
            return
        GW.Player.ChangeTarget(target_id)
        GW.GLOBAL_CACHE.SkillBar.UseSkill(int(slot), target_id)

    @staticmethod
    def method(name: CMD):
        def decorator(func: RPCMethod[P, R]) -> RPCMethod[P, R]:
            RPC.register(name, func)
            return func

        return decorator


RPC.clear()
RPC.register(RPC.CMD.MOVE, GW.Player.Move)
RPC.register(RPC.CMD.GETSKILLBAR, GW.GLOBAL_CACHE.SkillBar.GetSkillbar)
RPC.register(RPC.CMD.GETEFFECTS, Jsonizer.get_effects)
RPC.register(RPC.CMD.GET_PLAYER_ID, GW.Player.GetAgentID)
RPC.register(RPC.CMD.GET_PLAYER_DATA, Jsonizer.player_data)
RPC.register(RPC.CMD.USE_SKILL, GW.GLOBAL_CACHE.SkillBar.UseSkill)
RPC.register(RPC.CMD.DROP_BOND, GW.GLOBAL_CACHE.Effects.DropBuff)
RPC.register(RPC.CMD.RELATIVE_MOVE, RPC.relative_move)
RPC.register(RPC.CMD.CHANGE_TARGET, GW.Player.ChangeTarget)
RPC.register(RPC.CMD.CAST_TARGETED, RPC.cast_targeted)

