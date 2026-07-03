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
        CAST_AT = "cast_at"               # host-resolved concrete cast: ChangeTarget(agent)+UseSkill(slot)
        INTERACT = "interact"             # interact an agent: host-given id, else client picks an enemy
        RESIGN = "resign"                 # /resign on the client (host "resign all")
        TRAVEL_TO = "travel_to"           # travel to a concrete map/region/district (call to outpost)
        INVITE_PLAYER = "invite_player"   # invite a player by name (the client's "accept" = invite host back)

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
        on the host learning the resolved id (the host ROUTER drops RPC return values by design).

        NOTE: the generic combat engine no longer uses this client-side resolution path -- per the
        CentralCommander 'all decisions host-side' rule the host now resolves the target AND
        evaluates cast conditions itself (cc_lib.combat_conditions) and issues a concrete CAST_AT.
        This is kept for any caller that still wants the client to pick a target for a bare spec."""
        target_id = RPC.resolve_combat_target(int(target_spec), int(skill_id))
        if not target_id:
            return
        GW.Player.ChangeTarget(target_id)
        GW.GLOBAL_CACHE.SkillBar.UseSkill(int(slot), target_id)

    @staticmethod
    def cast_at(slot: int, agent_id: int) -> None:
        """CLIENT-SIDE pure executor: change to the host-resolved ``agent_id`` and fire ``slot`` on
        it -- no resolution, no condition checks (the host already made every decision). agent_id 0
        means self/no-target (UseSkill treats 0 as the caster)."""
        agent_id = int(agent_id or 0)
        if agent_id:
            GW.Player.ChangeTarget(agent_id)
        GW.GLOBAL_CACHE.SkillBar.UseSkill(int(slot), agent_id)

    @staticmethod
    def interact(agent_id: int = 0) -> None:
        """CLIENT-SIDE: interact with an agent. Two modes:
          * host-driven -- the host passes a concrete ``agent_id`` it knows in the shared instance,
            and the client interacts exactly that agent. Reusable for friendly interactions later
            (NPCs, chests, allies, res targets), not just attacking.
          * client-resolved (``agent_id`` == 0) -- the client picks a combat target in its own
            world: the party-called target if it's a live enemy, else the nearest enemy in spell
            range. Mirrors HeroAI CombatClass.ChooseTarget (called target first, then nearest).

        Interacting an enemy starts the client's normal auto-attack loop (it keeps swinging on its
        own from there). No-op when no valid target is found -- the 'is there anything to interact
        with' feasibility check lives here, in the client's own world, exactly like cast_targeted."""
        target_id = int(agent_id or 0)
        if not target_id:
            from Py4GWCoreLib import Range
            spell = Range.Spellcast.value
            target_id = int(GW.Party.GetPartyTarget() or 0)
            if target_id:
                try:
                    valid = GW.Agent.IsAlive(target_id) and GW.Agent.GetAllegiance(target_id)[1] == "Enemy"
                except Exception:
                    valid = False
                if not valid:
                    target_id = 0
            if not target_id:
                target_id = int(Routines.Agents.GetNearestEnemy(spell) or 0)
        if not target_id:
            return
        GW.Player.Interact(target_id, False)

    @staticmethod
    def resign() -> None:
        """CLIENT-SIDE: resign the current explorable (HeroAI sends the chat command twice with a
        short gap; the inbound-RPC drain already runs once per main frame, so a single send here is
        the per-tick equivalent and the host's button can be pressed again if a client misses it)."""
        GW.Player.SendChatCommand("resign")

    @staticmethod
    def travel_to(map_id: int, region: int, district: int, language: int) -> None:
        """CLIENT-SIDE: travel to a concrete map/region/district/language (the host's outpost). No-op
        if the client is already in that exact instance, so a re-issue to a straggler that already
        arrived doesn't pointlessly re-zone it. This is the 'call to outpost' half of party forming;
        the host confirms arrival via the synced map fields before moving on to the invite step."""
        try:
            on_map = (int(GW.Map.GetMapID()) == int(map_id)
                      and int(GW.Map.GetRegion()[0]) == int(region)
                      and int(GW.Map.GetDistrict()) == int(district))
        except Exception:
            on_map = False
        if on_map:
            return
        GW.Map.TravelToRegion(int(map_id), int(region), int(district), int(language))

    @staticmethod
    def invite_player(name: str) -> None:
        """CLIENT-SIDE: invite a player (the host) by character name. Guild Wars merges two players'
        parties when they invite EACH OTHER, so after the host has invited this client, the client
        inviting the host back is what 'accepts' and completes the join."""
        if name:
            GW.Party.Players.InvitePlayer(str(name))

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
RPC.register(RPC.CMD.CAST_AT, RPC.cast_at)
RPC.register(RPC.CMD.INTERACT, RPC.interact)
RPC.register(RPC.CMD.RESIGN, RPC.resign)
RPC.register(RPC.CMD.TRAVEL_TO, RPC.travel_to)
RPC.register(RPC.CMD.INVITE_PLAYER, RPC.invite_player)

