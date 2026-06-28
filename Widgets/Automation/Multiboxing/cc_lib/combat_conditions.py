"""HOST-SIDE combat decisions: target resolution + cast-condition evaluation.

CentralCommander makes ALL decisions on the host -- the clients are pure executors. So this module
runs on the HOST (inside the behavior worker thread, which is map-gated): for a given controlled
client it (1) resolves the concrete cast target by scanning the host's view of the shared instance
around that client's position, and (2) evaluates whether the skill's HeroAI cast conditions are met
against that target, before the host issues a concrete cast order. This is the host-side port of
HeroAI ``CombatClass.AreCastConditionsMet`` (HeroAI/combat.py) + the combo / has-effect / hex-on-
spirit gates from ``IsReadyToCast``, fixing the two combat bugs (dagger chains ignored; energy /
boolean conditions ignored).

Where each input comes from (host vs client):
  * The CLIENT's own self-state -- energy / HP / effects-by-id / current target / its target's
    dagger sequence -- is read from the SYNCED blackboard (``client.game_client``). The dagger
    sequence is the one genuinely client-only input (per-(attacker,target) state the host can't see
    in its own world); it's synced up via the jsonizer. Ping + fast-casting are likewise synced for
    the interrupt-feasibility check.
  * Everything else -- the target's and surrounding agents' alive/condition/hex/enchant/casting
    state, area counts, the client's world position -- the host reads directly via its own GW in the
    shared instance. ``Routines.Checks.Agents.*`` are shared-memory-backed for same-party agents and
    fall back to the agent struct for the target, so the host sees both the client and its target.
    (Assumes the multibox party stays clustered enough for the client's surroundings to be in the
    host's agent array, which host-side targeting already requires.)

REUSES HeroAI's leaf helpers (``GetEnergyValues`` / ``GetEffectAndBuffIds`` / ``IsPartyMember`` /
``IsResurrectablePartyMember`` / ``is_interrupt_feasible``), each of which self-constructs its own
CacheData -- no headless wiring. If HeroAI is unimportable the module self-disables: ``can_cast``
returns True (allow) and ``resolve_target`` falls back to the synced current target.

Fail-open: any unexpected error in ``can_cast`` returns True so a bug here can never brick combat.
"""
from Py4GWCoreLib import GLOBAL_CACHE, Agent, Routines, Range


class _Const:
    """Lazily-resolved HeroAI handles + skill-id constants (name->id is a runtime lookup, resolve
    once). ``ok`` stays False until a successful load; while False the host-side gate is disabled."""
    ok = False
    table = None
    N = None      # SkillNature
    T = None      # Skilltarget
    ST = None     # SkillType
    energy_of = None
    effect_ids_of = None
    is_party_member = None
    is_resurrectable = None
    interrupt_feasible = None
    burning = blind = cracked_armor = dazed = deep_wound = disease = weakness = 0
    ids: dict = {}
    preparations: tuple = ()

    @classmethod
    def load(cls):
        if cls.ok:
            return True
        try:
            from HeroAI.custom_skill import CustomSkillClass
            from HeroAI.types import SkillNature, Skilltarget, SkillType
            from HeroAI.utils import GetEnergyValues, GetEffectAndBuffIds, IsPartyMember
            from HeroAI.targeting import IsResurrectablePartyMember
        except Exception:
            cls.ok = False
            return False
        try:
            from HeroAI.interrupt import is_interrupt_feasible
            cls.interrupt_feasible = is_interrupt_feasible
        except Exception:
            cls.interrupt_feasible = None

        cls.table = CustomSkillClass()
        cls.N, cls.T, cls.ST = SkillNature, Skilltarget, SkillType
        cls.energy_of = GetEnergyValues
        cls.effect_ids_of = GetEffectAndBuffIds
        cls.is_party_member = IsPartyMember
        cls.is_resurrectable = IsResurrectablePartyMember

        gid = GLOBAL_CACHE.Skill.GetID
        cls.burning = gid("Burning"); cls.blind = gid("Blind"); cls.cracked_armor = gid("Cracked_Armor")
        cls.dazed = gid("Dazed"); cls.deep_wound = gid("Deep_Wound"); cls.disease = gid("Disease")
        cls.weakness = gid("Weakness")
        names = [
            "Energy_Drain", "Energy_Tap", "Ether_Feast", "Ether_Lord", "Blood_is_Power",
            "Blood_Ritual", "Essence_Strike", "Glowing_Signet", "Clamor_of_Souls",
            "Waste_Not_Want_Not", "Mend_Body_and_Soul", "Grenths_Balance", "Deaths_Retreat",
            "Plague_Sending", "Plague_Signet", "Plague_Touch", "Golden_Fang_Strike",
            "Golden_Fox_Strike", "Golden_Lotus_Strike", "Golden_Phoenix_Strike",
            "Golden_Skull_Strike", "Brutal_Weapon", "Signet_of_Removal", "Dwaynas_Kiss",
            "Unnatural_Signet", "Toxic_Chill", "Discord", "Empathic_Removal", "Iron_Palm",
            "Melandrus_Resilience", "Necrosis", "Peace_and_Harmony", "Purge_Signet",
            "Resilient_Weapon", "Gaze_from_Beyond", "Spirit_Burn", "Signet_of_Ghostly_Might",
            "Comfort_Animal", "Heal_as_One", "Never_Rampage_Alone", "Whirlwind_Attack",
            "Natures_Blessing", "Relentless_Assault", "Great_Dwarf_Weapon", "Junundu_Wail",
            "Unknown_Junundu_Ability", "Leave_Junundu", "Junundu_Tunnel", "Junundu_Siege",
        ]
        cls.ids = {n: gid(n) for n in names}
        if not cls.ids.get("Junundu_Siege"):
            cls.ids["Junundu_Siege"] = 1441
        cls.preparations = tuple(
            s for s in (
                gid("Apply_Poison"), gid("Barbed_Arrows"), gid("Choking_Gas"),
                gid("Corrupted_Breath"), gid("Disrupting_Accuracy"), gid("Expert_Focus"),
                gid("Glass_Arrows"), gid("Ignite_Arrows"), gid("Kindle_Arrows"),
                gid("Marksmans_Wager"), gid("Melandrus_Arrows"), gid("Rapid_Fire"),
                gid("Read_the_Wind"), gid("Seeking_Arrows"), gid("Trappers_Focus"),
            ) if s
        )
        cls.ok = True
        return True


def _id(name: str) -> int:
    return _Const.ids.get(name, 0)


# --------------------------------------------------------------- per-client accessors
def _me(client) -> int:
    return int(client.game_client.agent_id or 0)


def _pos(agent_id: int):
    try:
        return Agent.GetXY(agent_id)
    except Exception:
        return (0.0, 0.0)


def _self_energy(client) -> float:
    """The client's energy as a 0..1 fraction from the synced blackboard, or -1.0 if not yet synced
    (mirrors HeroAI GetEnergyValues' invalid sentinel)."""
    gc = client.game_client
    return (gc.energy / gc.max_energy) if gc.max_energy else -1.0


def _self_hp(client) -> float:
    gc = client.game_client
    return (gc.hp / gc.max_hp) if gc.max_hp else 1.0


def _energy_fraction(client, agent_id: int) -> float:
    """Energy fraction for any agent: synced blackboard for the client itself, else HeroAI's
    party-aware GetEnergyValues (host-side, shared-memory-backed for party members)."""
    if agent_id == _me(client):
        return _self_energy(client)
    try:
        return _Const.energy_of(agent_id)
    except Exception:
        return -1.0


def _has_effect(client, agent_id: int, skill_id: int, exact_weapon_spell: bool = False) -> bool:
    """Port of CombatClass.HasEffect. For the client itself the active-effect set is the SYNCED
    effects dict (reliable, canonical); for any other agent the host reads it directly
    (GetEffectAndBuffIds: shared memory for party, agent struct for the target)."""
    if agent_id == _me(client):
        active = set(client.game_client.effects.keys())
    else:
        active = set(_Const.effect_ids_of(agent_id) or [])
    desc = _Const.table.get_skill(skill_id)
    shared = desc.Conditions.SharedEffects if desc else []
    result = skill_id in active or any(s in active for s in shared)
    if not result:
        skilltype, _ = GLOBAL_CACHE.Skill.GetType(skill_id)
        if not exact_weapon_spell and skilltype == _Const.ST.WeaponSpell.value:
            result = Routines.Checks.Agents.IsWeaponSpelled(agent_id)
        elif skilltype == _Const.ST.Preparation.value:
            result = any(p in active for p in _Const.preparations)
    return result


# --------------------------------------------------------------- host-side area scans
def _nearest(px, py, ids) -> int:
    best, best_d = 0, None
    for aid in ids:
        ax, ay = _pos(aid)
        d = (ax - px) ** 2 + (ay - py) ** 2
        if best_d is None or d < best_d:
            best, best_d = int(aid), d
    return best


def _enemies(px, py, r) -> list:
    try:
        return list(Routines.Agents.GetFilteredEnemyArray(px, py, r))
    except Exception:
        return []


def _allies(px, py, r, other_ally=False) -> list:
    try:
        return list(Routines.Agents.GetFilteredAllyArray(px, py, r, other_ally=other_ally))
    except Exception:
        return []


def _spirits(px, py, r) -> list:
    try:
        return list(Routines.Agents.GetFilteredSpiritArray(px, py, r))
    except Exception:
        return []


def _minions(px, py, r) -> list:
    try:
        return list(Routines.Agents.GetFilteredMinionArray(px, py, r))
    except Exception:
        return []


def _is_live_enemy(agent_id: int) -> bool:
    if not agent_id:
        return False
    try:
        if not Routines.Checks.Agents.IsAlive(agent_id):
            return False
        _, name = Agent.GetAllegiance(agent_id)
        return name == "Enemy"
    except Exception:
        return False


def _lowest_ally(client, px, py, skill_id, other_ally=False, area=None) -> int:
    """Lowest-HP living ally in range, skipping any that already carry ``skill_id`` (HeroAI's
    filter_skill_id) -- the host-side, position-parameterized analog of TargetLowestAlly."""
    area = area or Range.Spellcast.value
    me = _me(client)
    best, best_hp = 0, 2.0
    for aid in _allies(px, py, area, other_ally=other_ally):
        aid = int(aid)
        if other_ally and aid == me:
            continue
        if not Routines.Checks.Agents.IsAlive(aid):
            continue
        if skill_id and _has_effect(client, aid, skill_id):
            continue
        hp = Routines.Checks.Agents.GetHealth(aid)
        if hp < best_hp:
            best, best_hp = aid, hp
    return best


def _enemy_casting(px, py) -> int:
    for aid in _enemies(px, py, Range.Spellcast.value):
        if Agent.IsCasting(int(aid)):
            return int(aid)
    return 0


def _dead_allies(px, py, r) -> list:
    """Dead PARTY members within ``r`` of the client, nearest first. Position-parameterized analog
    of Routines.Agents.GetDeadAllyArray (which is local-player-relative): the host scans its own
    instance-wide dead-ally array but filters by distance from the CLIENT, so a healer client resses
    a corpse near IT, not near the host."""
    try:
        from Py4GWCoreLib import AgentArray
        arr = AgentArray.GetDeadAllyArray()
        arr = AgentArray.Filter.ByDistance(arr, (px, py), r)
        arr = AgentArray.Filter.ByCondition(arr, _Const.is_party_member)
        arr = AgentArray.Sort.ByDistance(arr, (px, py))
        return [int(a) for a in arr]
    except Exception:
        return []


def _corpse(px, py, r, exploitable=False) -> int:
    """Nearest dead/exploitable corpse (any allegiance) within ``r`` of the client -- for corpse
    exploitation / minion skills. Mirrors Routines.Agents.GetCorpses/GetExploitableCorpses but
    scanned around the client's position."""
    try:
        from Py4GWCoreLib import AgentArray
        arr = AgentArray.GetAgentArray()
        arr = AgentArray.Filter.ByDistance(arr, (px, py), r)
        if exploitable:
            arr = AgentArray.Filter.ByCondition(arr, lambda a: Agent.IsExploitableCorpse(a))
        else:
            arr = AgentArray.Filter.ByCondition(arr, lambda a: Agent.IsDead(a))
        return _nearest(px, py, list(arr))
    except Exception:
        return 0


# --------------------------------------------------------------- host-side targeting
def resolve_target(client, target_spec: int, skill_id: int = 0) -> int:
    """Resolve the concrete cast target for ``skill_id`` in the host's view of the shared instance,
    scanning around the CLIENT's position per HeroAI Skilltarget spec. Returns 0 when nothing
    qualifies. Broad-but-basic (mirrors the spirit of HeroAI GetAppropiateTarget): the common specs
    are handled; everything enemy-ish defaults to the client's current target if it's a live enemy
    (so dagger combos chain on it), else the nearest enemy. Dead-agent specs (DeadAlly /
    ResurrectionAlly / Corpse / ExploitableCorpse) scan the host's instance-wide dead arrays filtered
    by distance from the CLIENT (see _dead_allies / _corpse)."""
    if not _Const.load():
        return int(getattr(client.game_client, "target_id", 0) or 0)
    me = _me(client)
    px, py = _pos(me)
    T = _Const.T
    s = int(target_spec)
    spell = Range.Spellcast.value

    if s == T.Self.value:
        return me
    if s == T.Ally.value:
        return _lowest_ally(client, px, py, skill_id)
    if s == T.OtherAlly.value:
        return _lowest_ally(client, px, py, skill_id, other_ally=True)
    if s in (T.AllyCaster.value, T.AllyMartial.value, T.AllyMartialMelee.value,
             T.AllyMartialRanged.value, T.AllyNonEnchanted.value):
        return _lowest_ally(client, px, py, skill_id)   # profession refinement TODO
    if s in (T.EnemyCasting.value, T.EnemyCastingSpell.value, T.EnemyCastingSpellOrChant.value):
        return _enemy_casting(px, py)
    if s == T.EnemyClustered.value:
        return _nearest(px, py, _enemies(px, py, spell))   # cluster refinement TODO
    if s == T.Spirit.value:
        return _nearest(px, py, _spirits(px, py, spell))
    if s == T.Minion.value:
        return _nearest(px, py, _minions(px, py, spell))
    if s in (T.DeadAlly.value, T.ResurrectionAlly.value):
        dead = _dead_allies(px, py, spell)
        return dead[0] if dead else 0
    if s == T.Corpse.value:
        return _corpse(px, py, spell, exploitable=False)
    if s == T.ExploitableCorpse.value:
        return _corpse(px, py, spell, exploitable=True)

    # Default / all remaining Enemy* specs: prefer the client's current target if it's a live enemy
    # (keeps dagger combos chaining on it), else nearest enemy around the client.
    current = int(getattr(client.game_client, "target_id", 0) or 0)
    if _is_live_enemy(current):
        return current
    return _nearest(px, py, _enemies(px, py, spell))


# --------------------------------------------------------------- gates
def combo_ok(client, skill_id: int, target_id: int) -> bool:
    """Dagger combo-chain gate (HeroAI combat.py:1672). Off-hand needs a landed lead (status 1),
    dual needs a landed off-hand (status 2); lead/non-combo are allowed at status 0 (fresh) or 3
    (chain reset). The dagger status is per (attacker, target) and only synced for the CLIENT's
    CURRENT target, so a combo skill is only evaluable -- and only valid -- against that target."""
    combo_type = GLOBAL_CACHE.Skill.Data.GetCombo(skill_id)
    if not combo_type:
        return True
    gc = client.game_client
    if target_id != int(getattr(gc, "target_id", 0) or 0):
        return False   # no dagger data for a non-current target -> can't chain on it
    dagger_status = int(getattr(gc, "target_dagger_status", 0) or 0)
    if ((combo_type == 1 and dagger_status not in (0, 3)) or
            (combo_type == 2 and dagger_status != 1) or
            (combo_type == 3 and dagger_status != 2)):
        return False
    return True


def _effect_already_present(client, skill_id: int, target_id: int) -> bool:
    desc = _Const.table.get_skill(skill_id)
    if desc is None:
        return False
    conditions = desc.Conditions
    skill_type, _ = GLOBAL_CACHE.Skill.GetType(skill_id)
    exact_ws = (skill_type == _Const.ST.WeaponSpell.value and conditions.AllowOverlapWeaponSpell)
    if (desc.TargetAllegiance != _Const.T.NonWeaponSpelledAlly.value
            and _has_effect(client, target_id, skill_id, exact_weapon_spell=exact_ws)):
        return True
    bip, brit = _id("Blood_is_Power"), _id("Blood_Ritual")
    if skill_id in (bip, brit) and bip and brit:
        if _has_effect(client, target_id, bip) or _has_effect(client, target_id, brit):
            return True
    return False


def _hex_on_spirit_blocked(skill_id: int, target_id: int) -> bool:
    skill_type, _ = GLOBAL_CACHE.Skill.GetType(skill_id)
    if skill_type != _Const.ST.Hex.value:
        return False
    try:
        from Py4GWCoreLib import Allegiance
        allegiance, _name = Agent.GetAllegiance(target_id)
        if Agent.IsSpirit(target_id) or (allegiance == Allegiance.Enemy.value and Agent.IsSpawned(target_id)):
            return True
    except Exception:
        return False
    return False


def _unique_property(client, skill_id: int, desc, target_id: int) -> bool:
    """Faithful port of the ``Conditions.UniqueProperty`` short-circuit block (combat.py:1037),
    re-rooted on the CLIENT: self reads use the client's agent id / synced energy+HP, target reads
    use host GW. An unrecognised UniqueProperty skill returns True (HeroAI's documented default)."""
    C = _Const
    me = _me(client)
    px, py = _pos(me)
    conds = desc.Conditions
    self_energy = _self_energy(client)
    has_valid_energy = self_energy >= 0.0
    self_hp = _self_hp(client)

    def is_(name: str) -> bool:
        return skill_id == _id(name)

    if is_("Energy_Drain") or is_("Energy_Tap") or is_("Ether_Lord"):
        return has_valid_energy and self_energy < conds.LessEnergy
    if is_("Ether_Feast"):
        return self_hp < conds.LessLife
    if is_("Essence_Strike"):
        return has_valid_energy and self_energy < conds.LessEnergy and bool(_spirits(px, py, Range.Spellcast.value))
    if is_("Glowing_Signet"):
        return has_valid_energy and self_energy < conds.LessEnergy and _has_effect(client, target_id, C.burning)
    if is_("Clamor_of_Souls"):
        return has_valid_energy and self_energy < conds.LessEnergy and Agent.IsHoldingItem(me)
    if is_("Waste_Not_Want_Not"):
        return (has_valid_energy and self_energy < conds.LessEnergy
                and not Agent.IsCasting(target_id) and not Routines.Checks.Agents.IsAttacking(target_id))
    if is_("Mend_Body_and_Soul"):
        spirits = bool(_spirits(px, py, Range.Earshot.value))
        return (self_hp < conds.LessLife) or (spirits and Routines.Checks.Agents.IsConditioned(target_id))
    if is_("Grenths_Balance"):
        return self_hp < conds.LessLife and self_hp < Routines.Checks.Agents.GetHealth(target_id)
    if is_("Deaths_Retreat"):
        return self_hp < Routines.Checks.Agents.GetHealth(target_id)
    if is_("Plague_Sending") or is_("Plague_Signet") or is_("Plague_Touch"):
        return Routines.Checks.Agents.IsConditioned(me)
    if (is_("Golden_Fang_Strike") or is_("Golden_Fox_Strike") or is_("Golden_Lotus_Strike")
            or is_("Golden_Phoenix_Strike") or is_("Golden_Skull_Strike")):
        return Routines.Checks.Agents.IsEnchanted(me)
    if is_("Brutal_Weapon"):
        return not Routines.Checks.Agents.IsEnchanted(me)
    if is_("Signet_of_Removal"):
        return (not Routines.Checks.Agents.IsEnchanted(target_id)) and Routines.Checks.Agents.IsConditioned(target_id)
    if is_("Dwaynas_Kiss") or is_("Unnatural_Signet") or is_("Toxic_Chill"):
        return Routines.Checks.Agents.IsHexed(target_id) or Routines.Checks.Agents.IsEnchanted(target_id)
    if is_("Discord"):
        return ((Routines.Checks.Agents.IsHexed(target_id) and Routines.Checks.Agents.IsConditioned(target_id))
                or Routines.Checks.Agents.IsEnchanted(target_id))
    if (is_("Empathic_Removal") or is_("Iron_Palm") or is_("Melandrus_Resilience")
            or is_("Necrosis") or is_("Peace_and_Harmony") or is_("Purge_Signet") or is_("Resilient_Weapon")):
        return Routines.Checks.Agents.IsHexed(target_id) or Routines.Checks.Agents.IsConditioned(target_id)
    if is_("Gaze_from_Beyond") or is_("Spirit_Burn") or is_("Signet_of_Ghostly_Might"):
        return bool(_spirits(px, py, Range.Spellcast.value))
    if is_("Comfort_Animal") or is_("Heal_as_One"):
        try:
            pet_id = int(GLOBAL_CACHE.Party.Pets.GetPetID(me) or 0)
        except Exception:
            pet_id = 0
        if pet_id == 0:
            return False
        return (Routines.Checks.Agents.GetHealth(pet_id) < conds.LessLife) or Routines.Checks.Agents.IsDead(pet_id)
    if is_("Never_Rampage_Alone"):
        try:
            pet_id = int(GLOBAL_CACHE.Party.Pets.GetPetID(me) or 0)
        except Exception:
            pet_id = 0
        return pet_id != 0 and Routines.Checks.Agents.IsAlive(pet_id)
    if is_("Whirlwind_Attack"):
        weapon_type, _ = Agent.GetWeaponType(me)
        return weapon_type not in (1, 6)   # not Bow / Spear
    if is_("Natures_Blessing"):
        return self_hp < conds.LessLife   # NPC-life refinement omitted (rare)
    if is_("Relentless_Assault"):
        return Routines.Checks.Agents.IsHexed(me) or Routines.Checks.Agents.IsConditioned(me)
    if is_("Junundu_Wail"):
        if not _enemies(px, py, Range.Spellcast.value):
            return self_hp < conds.LessLife
        return False
    if is_("Junundu_Tunnel"):
        return not _enemies(px, py, Range.Spellcast.value)
    if is_("Junundu_Siege"):
        near = _enemies(px, py, Range.Nearby.value)
        far = [e for e in _enemies(px, py, Range.Earshot.value) if e not in near]
        return bool(near) and bool(far)
    if is_("Unknown_Junundu_Ability") or is_("Leave_Junundu"):
        return False

    return True


def _interrupt_feasible(client, target_id: int, skill_id: int) -> bool:
    """HeroAI interrupt feasibility using the CLIENT's synced ping + fast-casting level (the host
    can't read those for a remote agent). Falls back to 'target is casting' if the helper is absent."""
    gc = client.game_client
    if _Const.interrupt_feasible is None:
        return Agent.IsCasting(target_id)
    try:
        return bool(_Const.interrupt_feasible(target_agent_id=target_id, our_skill_id=skill_id,
                                               fast_casting_level=int(getattr(gc, "fast_casting", 0) or 0),
                                               ping_ms=int(getattr(gc, "ping", 0) or 0)))
    except Exception:
        return Agent.IsCasting(target_id)


def _conditions_met(client, skill_id: int, target_id: int) -> bool:
    """Host-side port of CombatClass.AreCastConditionsMet (combat.py:1024)."""
    C = _Const
    desc = C.table.get_skill(skill_id)
    if desc is None:
        return True
    conds = desc.Conditions
    me = _me(client)

    if desc.Nature == C.N.Resurrection.value:
        return bool(C.is_resurrectable(target_id) and Routines.Checks.Agents.IsDead(target_id))

    if conds.UniqueProperty:
        return _unique_property(client, skill_id, desc, target_id)

    need = 0
    have = 0

    def req(flag) -> bool:
        nonlocal need
        if flag:
            need += 1
        return bool(flag)

    if req(conds.IsAlive) and Routines.Checks.Agents.IsAlive(target_id):
        have += 1

    # DELIBERATE DEVIATION: HeroAI computes is_conditioned for HasCondition but never adds it to its
    # satisfied-count (combat.py) -- a latent bug that makes any HasCondition skill never fire. We
    # treat HasCondition as the obviously-intended 'target has any condition'.
    if req(conds.HasCondition) and Routines.Checks.Agents.IsConditioned(target_id):
        have += 1
    if req(conds.HasBleeding) and Agent.IsBleeding(target_id):
        have += 1
    if req(conds.HasBlindness) and _has_effect(client, target_id, C.blind):
        have += 1
    if req(conds.HasBurning) and _has_effect(client, target_id, C.burning):
        have += 1
    if req(conds.HasCrackedArmor) and _has_effect(client, target_id, C.cracked_armor):
        have += 1
    if req(conds.HasCrippled) and Agent.IsCrippled(target_id):
        have += 1
    if req(conds.HasDazed) and _has_effect(client, target_id, C.dazed):
        have += 1
    if req(conds.HasDeepWound) and _has_effect(client, target_id, C.deep_wound):
        have += 1
    if req(conds.HasDisease) and _has_effect(client, target_id, C.disease):
        have += 1
    if req(conds.HasPoison) and Agent.IsPoisoned(target_id):
        have += 1
    if req(conds.HasWeakness) and _has_effect(client, target_id, C.weakness):
        have += 1

    if req(conds.HasWeaponSpell):
        if Routines.Checks.Agents.IsWeaponSpelled(target_id):
            if not conds.WeaponSpellList:
                have += 1
            elif any(_has_effect(client, target_id, s, exact_weapon_spell=True) for s in conds.WeaponSpellList):
                have += 1

    if req(conds.HasEnchantment):
        if Routines.Checks.Agents.IsEnchanted(target_id):
            if not conds.EnchantmentList:
                have += 1
            elif any(_has_effect(client, target_id, s) for s in conds.EnchantmentList):
                have += 1

    if req(conds.HasDervishEnchantment):
        for buff in (C.effect_ids_of(target_id) or []):
            stype, _ = GLOBAL_CACHE.Skill.GetType(buff)
            if stype == C.ST.Enchantment.value:
                _, profession = GLOBAL_CACHE.Skill.GetProfession(buff)
                if profession == "Dervish":
                    have += 1
                    break

    if req(conds.HasHex):
        if Routines.Checks.Agents.IsHexed(target_id):
            if not conds.HexList:
                have += 1
            elif any(_has_effect(client, target_id, s) for s in conds.HexList):
                have += 1

    if req(conds.HasChant):
        if C.is_party_member(target_id):
            for buff in (C.effect_ids_of(target_id) or []):
                stype, _ = GLOBAL_CACHE.Skill.GetType(buff)
                if stype == C.ST.Chant.value:
                    if not conds.ChantList:
                        have += 1
                    elif buff in conds.ChantList:
                        have += 1
                        break

    if req(conds.IsCasting):
        if Agent.IsCasting(target_id):
            casting_skill_id = Agent.GetCastingSkillID(target_id)
            if desc.Nature == C.N.Interrupt.value:
                if _interrupt_feasible(client, target_id, skill_id):
                    if not conds.CastingSkillList or casting_skill_id in conds.CastingSkillList:
                        have += 1
            else:
                if GLOBAL_CACHE.Skill.Data.GetActivation(casting_skill_id) >= 0.250:
                    if not conds.CastingSkillList or casting_skill_id in conds.CastingSkillList:
                        have += 1

    if req(conds.IsKnockedDown) and Routines.Checks.Agents.IsKnockedDown(target_id):
        have += 1
    if req(conds.IsMoving) and Agent.IsMoving(target_id):
        have += 1
    if req(conds.IsAttacking) and Routines.Checks.Agents.IsAttacking(target_id):
        have += 1
    if req(conds.IsHoldingItem) and Agent.IsHoldingItem(target_id):
        have += 1

    if req(conds.LessLife > 0) and Routines.Checks.Agents.GetHealth(target_id) < conds.LessLife:
        have += 1
    if req(conds.MoreLife > 0) and Routines.Checks.Agents.GetHealth(target_id) > conds.MoreLife:
        have += 1

    if req(conds.LessEnergy > 0):
        if C.is_party_member(target_id):
            tenergy = _energy_fraction(client, target_id)
            if 0.0 <= tenergy < conds.LessEnergy:
                have += 1
        else:
            have += 1   # non-party doesn't report energy -> treat as satisfied (HeroAI)

    if req(conds.LessSelfEnergyPercentage > 0):
        se = _self_energy(client)
        if 0.0 <= se <= conds.LessSelfEnergyPercentage:
            have += 1

    if req(conds.Overcast > 0):
        try:
            if Agent.GetOvercast(me) < conds.Overcast:
                have += 1
        except Exception:
            have += 1

    # ---- area conditions: each a HARD gate (immediate False when unmet), scanned around the client
    px, py = _pos(me)

    if conds.IsPartyWide:
        need += 1
        if _party_wide_life_met(client, conds, px, py):
            have += 1
        else:
            return False

    if req(conds.RequiresSpiritInEarshot) and _spirits(px, py, Range.Earshot.value):
        have += 1

    if conds.EnemyCount > 0:
        need += 1
        if len(_enemies(px, py, conds.EnemiesInRange)) >= conds.EnemyCount:
            have += 1
        else:
            return False

    if conds.AlliesInRange > 0:
        need += 1
        if len(_allies(px, py, conds.AlliesInRangeArea, other_ally=True)) >= conds.AlliesInRange:
            have += 1
        else:
            return False

    if conds.SpiritsInRange > 0:
        need += 1
        if len(_spirits(px, py, conds.SpiritsInRangeArea)) >= conds.SpiritsInRange:
            have += 1
        else:
            return False

    if conds.MinionsInRange > 0:
        need += 1
        if len(_minions(px, py, conds.MinionsInRangeArea)) >= conds.MinionsInRange:
            have += 1
        else:
            return False

    if conds.CloseToAggro:
        need += 1
        if _enemies(px, py, Range.Earshot.value):
            have += 1
        else:
            return False

    return need == have


def _party_wide_life_met(client, conds, px, py) -> bool:
    try:
        area = Range.SafeCompass.value if conds.PartyWideArea == 0 else conds.PartyWideArea
        members = list(_allies(px, py, area, other_ally=True)) + [_me(client)]
        total, count = 0.0, 0
        for aid in members:
            frac = Routines.Checks.Agents.GetHealth(aid)
            if frac > 0.0:
                total += frac
                count += 1
        return count > 0 and (total / count) < conds.LessLife
    except Exception:
        return False


def can_cast(client, skill_id: int, target_id: int) -> bool:
    """Host-side feasibility gate against an already-resolved target: dagger combo ordering, hex-on-
    spirit guard, don't-re-apply-existing-effect, then the AreCastConditionsMet feature evaluation.
    Fail-open (returns True) when HeroAI is unavailable or on any unexpected error."""
    if not _Const.load():
        return True
    if not target_id:
        return False
    try:
        if not combo_ok(client, skill_id, target_id):
            return False
        if _hex_on_spirit_blocked(skill_id, target_id):
            return False
        if _effect_already_present(client, skill_id, target_id):
            return False
        return _conditions_met(client, skill_id, target_id)
    except Exception:
        return True
