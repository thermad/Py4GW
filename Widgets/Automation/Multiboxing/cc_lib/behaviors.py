"""Behavior layer: utility-AI primitives, the Behavior base + concrete behaviors
(minion printer / permaseed / combat), the generic combat engine, and the behavior
manager. Sits on top of the bt framework; the widget file only touches the concrete
behavior classes and the BehaviorManager."""
import time, uuid
from collections import defaultdict
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Callable, Set, Type, Optional

import Py4GWCoreLib as GW
from Py4GWCoreLib import PyImGui, Routines

from cc_lib.bt import (Status, Node, Task, DoUntil, CastWaitForEffect, CastSkill,
                       MoveTo, Sequence, Parallel, Wait, WaitUntil, TaskManager,
                       State, StateMachine)
from cc_lib.transport import Client
from cc_lib.rpc import RPC
from cc_lib.misc_helpers import Vec2, ThreadManager
from cc_lib.networking import NetworkManager


# ---------------------------------------------------------------- utility engine
def clamp01(x: float) -> float:
    """Clamp to [0, 1] -- the working range for utility scores / response curves."""
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class Tier(int, Enum):
    """Priority class for utility Actions. A running Action is preempted ONLY by an applicable
    Action of a strictly higher tier; within a tier the selector commits to its current pick
    until that pick finishes, so it never thrashes a cast it just started. Scores order
    Actions *within* a tier; the tier is the coarse "interrupt what I'm doing" gate."""
    IDLE = 0          # filler when nothing else wants to act (hold position / auto-attack)
    SUSTAINED = 10    # upkeep / rotation (maintain an enchant, keep a bond up)
    REACTIVE = 20     # respond now (emergency heal, flee, regroup)
    CRITICAL = 30     # overrides everything (scripted / coordinated team play in progress)


class Action:
    """A candidate the UtilitySelector can choose. ``score`` is cheap and pure given ctx;
    ``build`` returns the Node (Task / Sequence / Parallel) that carries it out, so execution
    reuses the existing kernel rather than inventing a second one. Nothing here is per-client:
    a host-coordinated team play is just an Action whose built Node commands several clients
    at once (the permaseed relocation choreography is the model). ``involves`` lets a per-client
    selector know a client is currently owned by a coordinated play and stay quiet meanwhile."""
    name: str = "action"
    tier: "Tier" = Tier.SUSTAINED

    def applicable(self, ctx: "Behavior") -> bool:
        return True

    def score(self, ctx: "Behavior") -> float:
        return 0.0

    def build(self, ctx: "Behavior") -> Node:
        raise NotImplementedError

    def involves(self, client: "Client") -> bool:
        """True while this Action (if running) is commanding ``client`` -- used to suspend
        that client's own selector so it doesn't fight a coordinated play. Default: no client."""
        return False


class ClientAction(Action):
    """An Action bound to a single client -- the common case (self-maintenance, a heal, one
    rotation step). Build a per-client loadout by instantiating these with each client."""
    def __init__(self, client: "Client"):
        self.client = client

    def involves(self, client: "Client") -> bool:
        return client is self.client


class UtilitySelector(Node):
    """Decision node: each tick score the applicable Actions and run the best, then COMMIT to
    its Node until that Node finishes -- re-picking only when the current Action ends or an
    applicable Action of a STRICTLY higher Tier appears (preemption). Commitment is the
    utility-engine analog of the kernel's "one step per tick, no blocking loops" rule: a naive
    argmax-every-tick would thrash mid-cast. This is the Node the refactor memo reserved -- it
    slots in beside State / StateMachine and replaces hand-ordered cascades like _rit_tank_task.

    ``actions_provider`` returns the live candidate list (rebuilt as roles / synergies change).
    ``suspended`` optionally silences the selector entirely (a per-client selector goes quiet
    while a coordinated team play owns its client -- same idea as _rit_tank_task's ``suspended``).
    ``on_pick`` is an optional observer ``(name|None) -> None`` for a UI readout of the choice."""
    def __init__(self, name: str, actions_provider: Callable[[], List[Action]],
                 suspended: Optional[Callable[[], bool]] = None,
                 on_pick: Optional[Callable[[Optional[str]], None]] = None):
        self.name = name
        self.actions_provider = actions_provider
        self.suspended = suspended
        self.on_pick = on_pick
        self.current: Optional[Node] = None
        self.current_action: Optional[Action] = None

    def _set_current(self, action: Optional[Action], node: Optional[Node], ctx: "Behavior") -> None:
        self.current_action = action
        self.current = node
        if node is not None:
            node.enter(ctx)
        if self.on_pick is not None:
            self.on_pick(action.name if action is not None else None)

    def _drop_current(self, ctx: "Behavior") -> None:
        if self.current is None and self.current_action is None:
            return
        if self.current is not None:
            self.current.cancel()
        self._set_current(None, None, ctx)

    def _best(self, ctx: "Behavior"):
        best: Optional[Action] = None
        best_score = 0.0
        for a in self.actions_provider():
            if not a.applicable(ctx):
                continue
            s = a.score(ctx)
            if best is None or (a.tier.value, s) > (best.tier.value, best_score):
                best, best_score = a, s
        return best, best_score

    def tick(self, ctx: "Behavior") -> Status:
        if self.suspended is not None and self.suspended():
            self._drop_current(ctx)
            return Status.RUNNING

        best, best_score = self._best(ctx)

        if self.current is not None:
            # Honor commitment: only a strictly higher tier may interrupt the running Action.
            if best is not None and best.tier.value > self.current_action.tier.value:
                self._drop_current(ctx)
            else:
                if self.current.tick(ctx) != Status.RUNNING:
                    self._drop_current(ctx)  # finished -> re-pick next tick
                return Status.RUNNING

        # Nothing committed: start the winner if it actually wants to act (score > 0).
        if best is not None and best_score > 0.0:
            self._set_current(best, best.build(ctx), ctx)
            if self.current is not None and self.current.tick(ctx) != Status.RUNNING:
                self._drop_current(ctx)
        return Status.RUNNING  # a selector is a perpetual node; it never itself completes

    def cancel(self) -> None:
        if self.current is not None:
            self.current.cancel()
        self.current = None
        self.current_action = None


@dataclass(frozen=True)
class Synergy:
    """A cluster of skills that, present together on a bar, grant a capability. Detection is
    feature-extraction, NOT classification: a bar 'has' every synergy whose ``skills`` are a
    subset of it, so one bar can carry several at once (a support combo AND a tank combo) --
    this is what lets a behavior treat a bar as composable skill combinations rather than one
    role. ``provides`` is a free capability tag; ``optional_skills`` strengthen a synergy when
    present but are not required to detect it (use them in Action scoring, not detection)."""
    name: str
    skills: frozenset
    provides: str = ""
    optional_skills: frozenset = frozenset()


def detect_synergies(bar: Set[int], registry: List[Synergy]) -> List[Synergy]:
    """Every synergy in ``registry`` whose required skills are all on ``bar``. Pure and
    order-stable (registry order), so it stays trivially unit-testable as the library grows."""
    bar = set(bar)
    return [s for s in registry if s.skills <= bar]


def _current_epoch() -> int:
    """Hot-reload generation counter. It lives on the MultiThreading *class*, which is loaded
    once (Py4GWCoreLib is not re-imported when this widget hot-reloads) and so survives a reload
    of this file. The bottom of the module bumps it on every (re)load; a behavior captures it at
    run() start and stops when it changes, so a worker thread orphaned by a hot-reload (the new
    CentralCommander has no handle to it) self-terminates instead of commanding clients forever."""
    return getattr(GW.MultiThreading, "_cc_epoch", 0)


class Behavior:
    """Base for all behavior models. Owns a root decision Node (usually a StateMachine)
    plus a TaskManager. ``run()`` is the worker-thread loop: tick the root, then update
    tasks. Subclasses override ``build_root()`` to define their decision graph. The
    Behavior instance itself is the ``ctx`` handed to every node, so nodes reach the
    blackboard via ``ctx.network_manager`` / ``ctx.roles`` and spawn work via ``ctx.enqueue``."""
    class Role(Enum):
        pass

    @dataclass
    class Signature:
        required_skills: Set[int]

    SIGNATURES: Dict['Behavior.Role', 'Behavior.Signature'] = {}

    def __init__(self, thread_globals: ThreadManager, network_manager: NetworkManager,
                 clients_provider: Optional[Callable[[], Dict[uuid.UUID, 'Client']]] = None):
        self.cache_thread_globals: ThreadManager = thread_globals
        self.network_manager = network_manager
        # A behavior only governs the clients assigned to it. ``clients_provider`` returns the
        # current assigned subset (a live view, so re-assignments take effect without a
        # restart); default None means "all clients" (legacy / single-behavior use).
        self._clients_provider = clients_provider
        self._running = False        # per-instance run flag so behaviors start/stop independently
        self._epoch = _current_epoch()   # captured again at run(); stops the loop after a hot-reload
        self.tasks = TaskManager()
        self.roles: Dict['Behavior.Role', List[Client]] = defaultdict(list)
        self.tick_interval = 0.05
        self.root: Node = self.build_root()

    def build_root(self) -> Node:
        """Override to return the behavior's root decision node."""
        return State()

    def set_clients_provider(self, provider: Callable[[], Dict[uuid.UUID, 'Client']]) -> None:
        self._clients_provider = provider

    def my_clients(self) -> Dict[uuid.UUID, 'Client']:
        """The clients this behavior governs -- its assigned subset, or all of them if no
        provider was set."""
        if self._clients_provider is not None:
            return self._clients_provider()
        return self.network_manager.client_list

    @property
    def is_running(self) -> bool:
        # Stop when this behavior is individually stopped, the whole environment shuts down, OR
        # this widget hot-reloaded out from under the thread (epoch changed -> orphaned worker).
        return (self._running and self.cache_thread_globals.is_threads_running
                and self._epoch == _current_epoch())

    def stop(self) -> None:
        self._running = False

    def enqueue(self, task: Node, owner=None) -> None:
        self.tasks.add(task, owner)
        task.enter(self)  # initialize per-run state (resets a Task, primes a Sequence)

    @staticmethod
    def group_done(tasks: List[Task]) -> bool:
        """True once every task in the group has left RUNNING (succeeded or failed)."""
        return all(t.status != Status.RUNNING for t in tasks)

    # --- blackboard helpers shared by states/tasks ---
    @staticmethod
    def cast(client: 'Client', skill_id: int, target_id: Optional[int] = None) -> bool:
        """Fire one cast of ``skill_id`` from ``client`` if it is off cooldown. Returns
        whether the cast was issued. Slot is resolved from the client's synced skillbar."""
        sd = client.game_client.get_skill_data(int(skill_id))
        if sd is None or sd.slot <= 0 or sd.get_recharge != 0:
            return False
        tgt = client.game_client.agent_id if target_id is None else target_id
        client.transport.send(RPC.CMD.USE_SKILL, sd.slot, tgt)
        return True

    @staticmethod
    def cast_targeted(client: 'Client', skill_id: int, target_spec: int) -> bool:
        """Fire one cast whose concrete target the CLIENT resolves from ``target_spec`` (a HeroAI
        Skilltarget value). Use this for non-self skills: the host can't scan the client's enemies/
        allies, so it ships the targeting *intent* and the client picks the agent in its own world
        (and suppresses the cast if none qualifies). Slot is resolved from the synced skillbar;
        recharge is gated host-side here, target validity client-side in RPC.cast_targeted."""
        sd = client.game_client.get_skill_data(int(skill_id))
        if sd is None or sd.slot <= 0 or sd.get_recharge != 0:
            return False
        client.transport.send(RPC.CMD.CAST_TARGETED, sd.slot, int(skill_id), int(target_spec))
        return True

    @staticmethod
    def effect_remaining(client: 'Client', skill_id: int) -> int:
        eff = client.game_client.effects.get(int(skill_id))
        return eff.time_remaining if eff else 0

    def run(self) -> None:
        ctx = self
        self._running = True
        self._epoch = _current_epoch()   # any later reload bumps the global epoch -> is_running False
        self.root.enter(ctx)
        try:
            while self.is_running:
                self.root.tick(ctx)
                self.tasks.update(ctx)
                time.sleep(self.tick_interval)
        finally:
            self.root.exit(ctx)

    @staticmethod
    def name() -> str:
        return "Behavior"

    def draw(self) -> None:
        pass

    def identify_players(self) -> Dict['Behavior.Role', List[Client]]:
        """Assign each connected client to the first role whose required-skills signature
        it satisfies, based on the skillbar synced into its blackboard."""
        assignments: Dict[Behavior.Role, List[Client]] = defaultdict(list)
        for client_id, client in self.my_clients().items():
            if client.game_client is None:
                continue
            client_skills = set(client.game_client.skills.keys())
            for role in self.Role:
                sig = self.SIGNATURES.get(role)
                if sig and sig.required_skills.issubset(client_skills):
                    assignments[role].append(client)
                    break
        return assignments


class TestBehavior(Behavior):
    """Connectivity smoke test: every few seconds log the roster and broadcast a move."""
    @staticmethod
    def name() -> str:
        return "Test Behavior"

    def build_root(self) -> Node:
        return TestBehavior.StatePing()

    class StatePing(State):
        def __init__(self):
            self._last = 0.0

        def tick(self, ctx: "TestBehavior") -> Status:
            if time.time() - self._last < 3.0:
                return Status.RUNNING
            self._last = time.time()
            # Only the clients assigned to THIS behavior -- not the whole roster. network_manager
            # .broadcast() PUBs to every client regardless of ownership, so two behaviors would
            # both command all clients; sending per-client over each client's own transport keeps
            # each behavior scoped to its own window region.
            clients = ctx.my_clients()
            if clients:
                first = next(iter(clients.values()))
                ua = first.game_client.get_skill_data(268)
                if ua:
                    print(f"[Test] first client UA slot={ua.slot} recharge={ua.get_recharge}. Effects: {first.game_client.effects}")
                target = list(GW.Player.GetXY())
                for client in clients.values():
                    client.transport.send(RPC.CMD.MOVE.value, *target)
            else:
                print("[Test] waiting for a client to connect")
            return Status.RUNNING


class MinionPrinterBehavior(Behavior):
    class Skills(int, Enum):
        weapon_of_quickening = 1268
        seed_of_life = 2105
        heal_area = 280
        kareis_healing_circle = 1119
        blessed_aura = 256
        shielding_hands = 299
        unyielding_aura = 268
        animate_bone_minions = 85
        dark_aura = 116
        agony = 145
        shield_of_absorption = 1399
        balthazars_spirit = 242

    class Role(Enum):
        MONA = auto()
        MONB = auto()
        RITMO = auto()
        UNKNOWN = auto()

    SIGNATURES = {
        Role.RITMO: Behavior.Signature({Skills.weapon_of_quickening.value, Skills.shield_of_absorption.value,
                               Skills.shielding_hands.value, Skills.heal_area.value, Skills.kareis_healing_circle.value,
                               Skills.balthazars_spirit.value}),
        Role.MONA: Behavior.Signature({Skills.unyielding_aura.value, Skills.seed_of_life.value,
                              Skills.blessed_aura.value, Skills.animate_bone_minions.value, Skills.dark_aura.value, Skills.agony.value})
    }

    def __init__(self, thread_globals: ThreadManager, network_manager: NetworkManager,
                 clients_provider: Optional[Callable[[], Dict[uuid.UUID, 'Client']]] = None):
        super().__init__(thread_globals, network_manager, clients_provider)
        self.target_minion_count = 20
        # login_number of the party player the permaseed trio follows (None = fall back to
        # the real party leader). Set from the draw() dropdown, read by StatePermaseed.
        self.follow_login_number: Optional[int] = None

    @staticmethod
    def name() -> str:
        return "Minion Printer"

    def build_root(self) -> Node:
        # HFSM phases: identify roles -> reduce monks to 1 max HP -> print minions -> permaseed.
        return StateMachine(lambda: MinionPrinterBehavior.StateIdentify())

    def draw(self) -> None:
        PyImGui.text("Minion Goal:")
        PyImGui.same_line(0.0, 0.0)
        self.target_minion_count = PyImGui.input_int("#binputminions", self.target_minion_count)

        # Follow-target dropdown: which party player the permaseed trio trails. Selection is
        # stored by login_number (stable across map loads, unlike agent ids) and consumed by
        # StatePermaseed._leader_pos, so it can be changed live without restarting.
        players = GW.Party.GetPlayers() or []
        logins = [p.login_number for p in players]
        names = [GW.Party.Players.GetPlayerNameByLoginNumber(l) or f"Player {l}" for l in logins]
        PyImGui.text("Follow:")
        PyImGui.same_line(0.0, 0.0)
        if not names:
            PyImGui.text("(no party players)")
            return
        if self.follow_login_number in logins:
            idx = logins.index(self.follow_login_number)
        else:
            # Default the selection to the real party leader, else the first player.
            leader_login = GW.Party.Players.GetLoginNumberByAgentID(GW.Party.GetPartyLeaderID())
            idx = logins.index(leader_login) if leader_login in logins else 0
            self.follow_login_number = logins[idx]
        new_idx = PyImGui.combo("#bfollowtarget", idx, names)
        if 0 <= new_idx < len(logins):
            self.follow_login_number = logins[new_idx]

    def _rit_tank_task(self, rit: 'Client', with_seed: bool,
                       monks: Optional[List['Client']] = None,
                       suspended: Optional[Callable[[], bool]] = None) -> Task:
        """Recurring self-maintenance for the Ritualist tank: a hand-ordered priority cascade.
        Pre-seed (``with_seed=False``) it just keeps itself buffed and protected. In the seed
        phase it instead spreads Weapon of Quickening across itself AND both monks so they can
        perma-maintain Seed of Life on it, and it stops maintaining Shielding Hands / Shield of
        Absorption while Seed of Life is active (Seed provides the protection). This is the
        shape a UtilitySelector will later replace -- one scored consideration per branch.

        ``suspended`` is an optional predicate; while it returns True the cascade does nothing.
        The permaseed relocation uses it to halt tank casting during a move -- otherwise the
        rit keeps trying to land Weapon of Quickening on the monks while they're scattered out
        of range, wasting the cast and stalling the move window."""
        S = self.Skills
        woq_targets = [rit] + list(monks or [])

        def cascade():
            if suspended is not None and suspended():
                return
            if self.effect_remaining(rit, S.balthazars_spirit) < 2000:
                self.cast(rit, S.balthazars_spirit); return
            if with_seed:
                # Keep Weapon of Quickening up on the rit AND both monks (fast Seed recharge).
                # Skip a dead monk -- otherwise the rit burns its WoQ cast trying to land it on
                # a corpse (mid-cycle in PermaPrint) instead of the live monk that needs it.
                for tgt in woq_targets:
                    if tgt is not rit and not GW.Agent.IsAlive(tgt.game_client.agent_id):
                        continue
                    if self.effect_remaining(tgt, S.weapon_of_quickening) < 5000:
                        if self.cast(rit, S.weapon_of_quickening, tgt.game_client.agent_id):
                            return
            elif self.effect_remaining(rit, S.weapon_of_quickening) < 4000:
                self.cast(rit, S.weapon_of_quickening); return
            # Maintain hard protection only while NOT covered by Seed of Life.
            seed_up = with_seed and self.effect_remaining(rit, S.seed_of_life) > 0
            if not seed_up and self.effect_remaining(rit, S.shielding_hands) < 3000:
                if self.cast(rit, S.shielding_hands): return
                if self.cast(rit, S.shield_of_absorption): return
            if rit.game_client.energy > 0.9:
                if self.cast(rit, S.heal_area): return
                self.cast(rit, S.kareis_healing_circle)

        return DoUntil(f"rit_tank_{'seed' if with_seed else 'noseed'}",
                       until=lambda: False, action=cascade, interval=0.25)

    @staticmethod
    def _ua_toggle_res(client: "Client", ua_skill: int) -> None:
        """Toggle Unyielding Aura: drop the maintained enchant if present (re-firing the
        resurrect on the next cast), otherwise (re)cast it. UA resurrects its target each
        time it is re-applied. Shared by the first-death and printing phases."""
        buff = client.game_client.buffs.get(int(ua_skill))
        if buff and buff.buff_id:
            client.transport.send(RPC.CMD.DROP_BOND, buff.buff_id)
        else:
            Behavior.cast(client, ua_skill)

    # ---------------------------------------------------------------- states
    class StateIdentify(State):
        """Tag connected clients to roles by skillbar signature; advance once we have the
        2 monks + 1 ritualist the farm needs."""
        def tick(self, ctx: "MinionPrinterBehavior") -> Status:
            ctx.roles = ctx.identify_players()
            return Status.RUNNING

        def transitions(self, ctx: "MinionPrinterBehavior") -> Optional[State]:
            monks = ctx.roles.get(ctx.Role.MONA, [])
            rit = ctx.roles.get(ctx.Role.RITMO, [])
            if len(monks) == 2 and len(rit) == 1:
                print("[MinionPrinter] roles identified -> first death")
                return ctx.StateFirstDeath()
            return None

    class StateFirstDeath(State):
        """Reduce BOTH monks to 1 max HP. Each cycle: Dark-Aura the sacrificer, sacrifice it
        with Agony, and have its partner resurrect it via Unyielding Aura. Roles SWAP every
        cycle so the two monks take deaths in turn rather than one monk dying forever. Loops
        internally (one persistent state) until both monks sit at 1 max HP, then hands off."""
        def enter(self, ctx: "MinionPrinterBehavior") -> None:
            self.invalid = False
            self.done = False
            monks = ctx.roles.get(ctx.Role.MONA, [])
            if len(monks) != 2:
                self.invalid = True
                return
            if all(m.game_client.max_hp == 1 for m in monks):
                self.done = True
                return
            # Start by reducing whichever monk still needs it.
            self.sacker: Client = next((m for m in monks if m.game_client.max_hp != 1), monks[0])
            self.resser: Client = next(m for m in monks if m is not self.sacker)
            self._begin_cycle(ctx)

        def _begin_cycle(self, ctx: "MinionPrinterBehavior") -> None:
            S = ctx.Skills
            self.step = 0
            self.sac: Optional[Task] = None
            self.res: Optional[Task] = None
            self.batch: List[Task] = [CastWaitForEffect(self.sacker, S.dark_aura),
                                      CastWaitForEffect(self.resser, S.unyielding_aura)]
            for t in self.batch:
                ctx.enqueue(t, self)

        def tick(self, ctx: "MinionPrinterBehavior") -> Status:
            if self.invalid or self.done:
                return Status.RUNNING
            S = ctx.Skills
            if self.step == 0 and ctx.group_done(self.batch):
                self.sac = DoUntil(
                    f"sac {self.sacker.game_client.agent_id}",
                    lambda: not GW.Agent.IsAlive(self.sacker.game_client.agent_id),
                    lambda: ctx.cast(self.sacker, S.agony), interval=0.3)
                ctx.enqueue(self.sac, self)
                self.step = 1
            elif self.step == 1 and self.sac.status != Status.RUNNING:
                self.res = DoUntil(
                    f"uares {self.resser.game_client.agent_id}",
                    lambda: GW.Agent.IsAlive(self.sacker.game_client.agent_id),
                    lambda: ctx._ua_toggle_res(self.resser, S.unyielding_aura), interval=0.2)
                ctx.enqueue(self.res, self)
                self.step = 2
            elif self.step == 2 and self.res.status != Status.RUNNING:
                # Cycle complete: done if both reduced, else swap roles and go again.
                monks = ctx.roles.get(ctx.Role.MONA, [])
                if all(m.game_client.max_hp == 1 for m in monks):
                    self.done = True
                else:
                    self.sacker, self.resser = self.resser, self.sacker  # alternate the death
                    # ...but if the new sacker is already reduced, keep reducing the one that isn't.
                    if self.sacker.game_client.max_hp == 1 and self.resser.game_client.max_hp != 1:
                        self.sacker, self.resser = self.resser, self.sacker
                    self._begin_cycle(ctx)
            return Status.RUNNING

        def transitions(self, ctx: "MinionPrinterBehavior") -> Optional[State]:
            if self.invalid:
                return ctx.StateIdentify()
            if self.done:
                print("[MinionPrinter] both monks at 1 max HP -> printing minions")
                return ctx.StatePrintMinions()
            return None

    # NOTE: StatePrintMinions and StatePermaseed below are first-pass ports of the
    # reference choreography (minion_print_loop / permaseed). The structure is sound but
    # the timing/positioning constants want an in-game tuning pass before they're trusted.
    class StatePrintMinions(State):
        """Alternate the two monks: position one outside the rit, Dark-Aura it, sacrifice
        it with Agony, then have the partner hold UA (resurrecting it) while animating
        minions off the fresh corpse. Loops until enough minions have aggroed the rit."""
        def enter(self, ctx: "MinionPrinterBehavior") -> None:
            monks = ctx.roles.get(ctx.Role.MONA, [])
            rit = ctx.roles.get(ctx.Role.RITMO, [])
            self.valid = len(monks) == 2 and len(rit) == 1
            if not self.valid:
                return
            self.rit: Client = rit[0]
            self.sacker: Client = monks[0]
            self.printer: Client = monks[1]
            self.step = 0
            self.batch: List[Task] = []
            self.sac: Optional[Task] = None
            self.printer_res: Optional[Node] = None
            ctx.enqueue(ctx._rit_tank_task(self.rit, with_seed=False), self)
            self._compute_anchor(ctx)
            # Final-cleanup bookkeeping: when the live goal is met at a cycle boundary we set
            # self.finishing, run one extra sac/res on the minion holder (no print), and only
            # then transition. self.finishing latches so a count dip during cleanup is safe.
            self.finishing = False
            self.cleanup_sac: Optional[Task] = None
            self.cleanup_res: Optional[Task] = None
            self.cleanup_done = False

        def _compute_anchor(self, ctx: "MinionPrinterBehavior") -> None:
            rx, ry = GW.Player.GetXY()
            self.rit_pos = Vec2(rx, ry)
            sx, sy = GW.Agent.GetXY(self.sacker.game_client.agent_id)
            d = Vec2(sx - rx, sy - ry)
            self.dir = d.normalized() if d.magnitude() > 1 else Vec2(1.0, 0.0)

        def _print_then_res_seq(self, ctx: "MinionPrinterBehavior") -> Sequence:
            """Build the per-corpse print-then-res choreography as an explicit Sequence:
            ensure UA is up, pause, animate minions off the corpse until that cast fires,
            pause, then toggle UA until the sacker resurrects. Splitting print and res into
            distinct steps fixes the old single-action version, which kept casting animate
            forever and never reached the resurrect. Add Wait/WaitUntil steps here to tune
            spacing or to catch on a condition before advancing."""
            S = ctx.Skills
            printer = self.printer
            sacker_id = self.sacker.game_client.agent_id

            def animate_has_fired() -> bool:
                sd = printer.game_client.get_skill_data(int(S.animate_bone_minions))
                return sd is not None and sd.get_recharge != 0  # on cooldown == it cast

            return Sequence(f"print_then_res {printer.game_client.agent_id}", [
                CastWaitForEffect(printer, S.unyielding_aura),                          # 1. UA up
                Wait(1.5),
                DoUntil("animate", animate_has_fired,
                        lambda: ctx.cast(printer, S.animate_bone_minions), interval=0.3),  # 2. print
                Wait(0.5),
                DoUntil("ua_res", lambda: GW.Agent.IsAlive(sacker_id),
                        lambda: ctx._ua_toggle_res(printer, S.unyielding_aura), interval=0.3),  # 3. res
            ])

        def tick(self, ctx: "MinionPrinterBehavior") -> Status:
            if not self.valid:
                return Status.RUNNING
            S = ctx.Skills
            if self.step == 0:
                mv = MoveTo(self.sacker, self.rit_pos + self.dir * 300, tolerance=120)
                mv2 = MoveTo(self.printer, self.rit_pos + self.dir * 1500, tolerance=120)
                self.batch = [mv, mv2]
                for t in self.batch:
                    ctx.enqueue(t, self)
                self.step = 1
            elif self.step == 1 and ctx.group_done(self.batch):
                self.batch = [CastWaitForEffect(self.sacker, S.dark_aura),
                              CastWaitForEffect(self.printer, S.unyielding_aura)]
                for t in self.batch:
                    ctx.enqueue(t, self)
                self.step = 2
            elif self.step == 2 and ctx.group_done(self.batch):
                self.sac = DoUntil(
                    f"print_sac {self.sacker.game_client.agent_id}",
                    lambda: not GW.Agent.IsAlive(self.sacker.game_client.agent_id),
                    lambda: ctx.cast(self.sacker, S.agony), interval=0.3)
                ctx.enqueue(self.sac, self); self.step = 3
            elif self.step == 3 and self.sac.status != Status.RUNNING:
                self.printer_res = self._print_then_res_seq(ctx)
                ctx.enqueue(self.printer_res, self); self.step = 4
            elif self.step == 4 and self.printer_res.status != Status.RUNNING:
                # Re-read the goal LIVE here (the operator can raise/lower it mid-run); only
                # commit to finishing at this cycle boundary, then latch via self.finishing
                # so a momentary count dip during cleanup can't abort the handoff.
                rx, ry = GW.Player.GetXY()
                target_now = (len(GW.Routines.Agents.GetFilteredEnemyArray(rx, ry, 200))
                              >= ctx.target_minion_count)
                if target_now or self.finishing:
                    self.finishing = True
                    # Target hit. The printer is holding the bone minions it just animated;
                    # in a normal cycle they'd die when it next becomes the sacker, but we're
                    # leaving the loop -- so sac it now (its minions die with it) and revive
                    # it WITHOUT printing, leaving zero allied minions to draw aggro during
                    # the permaseed movement cycle. Dark Aura the holder first.
                    self.batch = [CastWaitForEffect(self.printer, S.dark_aura)]
                    ctx.enqueue(self.batch[0], self)
                    self.step = 5
                else:
                    self.sacker, self.printer = self.printer, self.sacker  # alternate roles
                    self._compute_anchor(ctx)
                    self.step = 0
            elif self.step == 5 and ctx.group_done(self.batch):
                self.cleanup_sac = DoUntil(
                    f"cleanup_sac {self.printer.game_client.agent_id}",
                    lambda: not GW.Agent.IsAlive(self.printer.game_client.agent_id),
                    lambda: ctx.cast(self.printer, S.agony), interval=0.3)
                ctx.enqueue(self.cleanup_sac, self); self.step = 6
            elif self.step == 6 and self.cleanup_sac.status != Status.RUNNING:
                # Plain UA res of the holder by its partner -- no animate, so no new minions.
                self.cleanup_res = DoUntil(
                    f"cleanup_res {self.printer.game_client.agent_id}",
                    lambda: GW.Agent.IsAlive(self.printer.game_client.agent_id),
                    lambda: ctx._ua_toggle_res(self.sacker, S.unyielding_aura), interval=0.3)
                ctx.enqueue(self.cleanup_res, self); self.step = 7
            elif self.step == 7 and self.cleanup_res.status != Status.RUNNING:
                self.cleanup_done = True
            return Status.RUNNING

        def transitions(self, ctx: "MinionPrinterBehavior") -> Optional[State]:
            if not self.valid:
                return ctx.StateIdentify()
            # The goal is consumed live at the step-4 cycle boundary (see tick), so changing
            # it in the UI takes effect on the next cycle. Hand off only once the cleanup
            # sac+res has cleared the last holder's minions -- both monks alive, zero minions.
            if self.cleanup_done:
                print("[MinionPrinter] minion target reached + cleaned up -> permaseed")
                return ctx.StatePermaseed()
            return None

    class StatePermaseed(State):
        """Steady state: keep Seed of Life on the rit, cast by whichever monk currently
        holds Weapon of Quickening (fast recharge), with the rit's own Shielding Hands as
        a fallback. Runs until the behavior is stopped. Ported from permaseed().

        Layered on top is a *very* selective relocation behavior so the farm can creep the
        whole formation toward the party leader without breaking the seed chain. The monks
        must sit within ``MONK_SEED_RANGE`` of the rit to cast Seed (and to receive Weapon
        of Quickening), so any move has to happen in a burst inside a safe window -- one is
        opened only when Seed has just been refreshed AND Weapon of Quickening has >= 8s
        left on everyone. See ``_build_move_seq`` for the choreography."""

        # --- movement tuning constants ---
        MONK_SEED_RANGE = 1190   # monks must be this close to the rit to cast Seed / get WoQ
        MONK_TOL = 50            # tolerance on the seed-range settle moves
        MOVE_OUT_DIST = 1600     # how far monks scatter (away from leader) before the rit steps
        RIT_STEP = 400           # how far the rit creeps toward the leader per relocation
        LEADER_FAR = 2100        # only relocate while the rit is at least this far from the leader
        RESETTLE_RANGE = 200     # minions must re-gather inside this radius before monks return
        WOQ_MIN = 8000           # required Weapon of Quickening remaining (ms) to open a window
        POST_MOVE_COOLDOWN = 5.0 # seconds of plain casting checks required between relocations

        def enter(self, ctx: "MinionPrinterBehavior") -> None:
            monks = ctx.roles.get(ctx.Role.MONA, [])
            rit = ctx.roles.get(ctx.Role.RITMO, [])
            self.valid = len(monks) == 2 and len(rit) == 1
            if not self.valid:
                return
            self.rit: Client = rit[0]
            self.mona: Client = monks[0]
            self.monb: Client = monks[1]
            self.step = 0
            # Relocation state: the in-flight move Sequence (None when settled), the edge
            # detector for "Seed was just refreshed", and the post-move cooldown deadline.
            self.moving: Optional[Sequence] = None
            self._prev_seed = 0
            self.move_cooldown_until = 0.0
            self.batch = [CastWaitForEffect(self.mona, ctx.Skills.blessed_aura),
                          CastWaitForEffect(self.monb, ctx.Skills.blessed_aura)]
            for t in self.batch:
                ctx.enqueue(t, self)
            # Suspend tank casting while a relocation is in flight: the monks are scattered
            # out of range, so Weapon of Quickening on them would just whiff and stall the move.
            ctx.enqueue(ctx._rit_tank_task(self.rit, with_seed=True, monks=[self.mona, self.monb],
                                           suspended=lambda: self.moving is not None), self)

        # ------------------------------------------------------------ positions
        def _pos(self, client: 'Client') -> Vec2:
            return Vec2.from_tuple(GW.Agent.GetXY(client.game_client.agent_id))

        def _leader_pos(self, ctx: "MinionPrinterBehavior") -> Optional[Vec2]:
            # Follow the operator-selected party player (draw() dropdown); if none chosen
            # yet, fall back to the game's real party leader.
            login = ctx.follow_login_number
            leader_id = (GW.Party.Players.GetAgentIDByLoginNumber(login) if login
                         else GW.Party.GetPartyLeaderID())
            if not leader_id:
                return None
            return Vec2.from_tuple(GW.Agent.GetXY(leader_id))

        def _settle_move(self, client: 'Client') -> MoveTo:
            """Pull a monk to exactly ``MONK_SEED_RANGE`` from the rit's current position,
            keeping it along its present offset so it stays on whatever side it is already
            on (the rit hasn't moved yet here). Used for the initial transition settle."""
            rit_pos = self._pos(self.rit)
            off = self._pos(client) - rit_pos
            d = off.normalized() if off.magnitude() > 1 else Vec2(1.0, 0.0)
            return MoveTo(client, rit_pos + d * self.MONK_SEED_RANGE, tolerance=self.MONK_TOL)

        # ------------------------------------------------------------ relocation
        def _move_opportunity(self, ctx: "MinionPrinterBehavior", seed_refreshed: bool) -> bool:
            """A relocation window is open only when: nothing is already moving, the
            post-move cooldown has elapsed, Seed was just refreshed, every WoQ target has
            >= 8s of Weapon of Quickening, and the rit is still far from the party leader."""
            if self.moving is not None or not seed_refreshed:
                return False
            if time.time() < self.move_cooldown_until:
                return False
            woq = ctx.Skills.weapon_of_quickening
            if any(ctx.effect_remaining(c, woq) < self.WOQ_MIN
                   for c in (self.rit, self.mona, self.monb)):
                return False
            leader_pos = self._leader_pos(ctx)
            if leader_pos is None:
                return False
            return (self._pos(self.rit) - leader_pos).magnitude() >= self.LEADER_FAR

        def _build_move_seq(self, ctx: "MinionPrinterBehavior") -> Optional[Sequence]:
            """One relocation burst:
            1. both monks scatter ``MOVE_OUT_DIST`` away from the rit, opposite the leader
               (clear of the minion ball so they don't steal aggro);
            2. the rit creeps ``RIT_STEP`` toward the leader (still requires LEADER_FAR);
            3. the rit holds until the minions re-gather within ``RESETTLE_RANGE``;
            4. the monks return to within ``MONK_SEED_RANGE`` of the rit's new spot,
               back on the safe (away-from-leader) side, ready to seed again."""
            rit_pos = self._pos(self.rit)
            leader_pos = self._leader_pos(ctx)
            if leader_pos is None:
                return None
            away = (rit_pos - leader_pos)
            away = away.normalized() if away.magnitude() > 1 else Vec2(1.0, 0.0)
            rit_target = rit_pos + (-away) * self.RIT_STEP           # toward the leader
            scatter = rit_pos + away * self.MOVE_OUT_DIST            # away from the leader
            settle = rit_target + away * self.MONK_SEED_RANGE        # back within seed range

            def minions_resettled() -> bool:
                rx, ry = GW.Agent.GetXY(self.rit.game_client.agent_id)
                gathered = GW.Routines.Agents.GetFilteredEnemyArray(rx, ry, self.RESETTLE_RANGE)
                return len(gathered) >= ctx.target_minion_count

            def monks_clear() -> bool:
                # The MoveTo above completes off the host's (network-lagged) view of the
                # remote monks, so it can read "arrived" a beat early. Hold the rit until
                # both monks are *observed* genuinely clear of it before it creeps -- this
                # is what stops the rit stepping while the monks are still pulling away.
                rp = self._pos(self.rit)
                return all((self._pos(m) - rp).magnitude() >= self.MOVE_OUT_DIST * 0.75
                           for m in (self.mona, self.monb))

            return Sequence("permaseed_move", [
                Parallel("monks_out", [MoveTo(self.mona, scatter, tolerance=self.MONK_TOL * 2),
                                       MoveTo(self.monb, scatter, tolerance=self.MONK_TOL * 2)]),
                WaitUntil(monks_clear, "monks_clear"),
                # Clean single step: no jitter/nudge so the rit walks straight to the point
                # and stops, keeping the minion ball rather than wandering off it.
                MoveTo(self.rit, rit_target, tolerance=self.MONK_TOL * 2, jitter=False, nudge=False),
                WaitUntil(minions_resettled, "resettle"),
                Parallel("monks_back", [MoveTo(self.mona, settle, tolerance=self.MONK_TOL),
                                        MoveTo(self.monb, settle, tolerance=self.MONK_TOL)]),
            ])

        def tick(self, ctx: "MinionPrinterBehavior") -> Status:
            if not self.valid:
                return Status.RUNNING
            S = ctx.Skills
            if self.step == 0:
                # Blessed Aura on both monks, then pull them to 1200 of the rit (tol 50):
                # the first step of settling into permaseed.
                if ctx.group_done(self.batch):
                    self.batch = [self._settle_move(self.mona), self._settle_move(self.monb)]
                    for t in self.batch:
                        ctx.enqueue(t, self)
                    self.step = 1
                return Status.RUNNING
            if self.step == 1:
                if ctx.group_done(self.batch):
                    self.step = 2
                return Status.RUNNING

            # --- steady state: maintain Seed, and relocate the formation in safe bursts ---
            rit_id = self.rit.game_client.agent_id
            seed_now = ctx.effect_remaining(self.rit, S.seed_of_life)
            seed_refreshed = seed_now > self._prev_seed  # rising edge == just (re)applied
            self._prev_seed = seed_now

            # Only maintain Seed while settled. During a relocation the monks are scattered
            # out of range, and a seed cast at the rit would make GW auto-walk the monk back
            # into casting range -- defeating the scatter and stalling monks_clear so the rit
            # never gets to move. The move window is gated on Seed being fresh, so it's safe
            # to pause upkeep for the brief burst.
            if self.moving is None:
                if seed_now < 1000:
                    if ctx.effect_remaining(self.mona, S.weapon_of_quickening) > 500:
                        ctx.cast(self.mona, S.seed_of_life, rit_id)
                    else:
                        ctx.cast(self.rit, S.shielding_hands, rit_id)
                self.mona, self.monb = self.monb, self.mona  # alternate which monk we lean on

            if self.moving is not None:
                if self.moving.status != Status.RUNNING:        # relocation finished
                    self.moving = None
                    self.move_cooldown_until = time.time() + self.POST_MOVE_COOLDOWN
            elif self._move_opportunity(ctx, seed_refreshed):
                seq = self._build_move_seq(ctx)
                if seq is not None:
                    ctx.enqueue(seq, self)
                    self.moving = seq
            return Status.RUNNING

        def transitions(self, ctx: "MinionPrinterBehavior") -> Optional[State]:
            if not self.valid:
                return ctx.StateIdentify()
            return None  # permaseed loops until the operator stops the behavior


class PermaPrintBehavior(MinionPrinterBehavior):
    """Fused minion-print + permaseed. Instead of printing to a target and THEN switching to
    a seed-maintenance loop, both monks run a single continuous relay cycle that prints AND
    keeps Seed of Life on the rit at the same time. Reuses the role identification, the
    monks-to-1HP first death, the rit tank task, and the follow/minion-count UI from
    MinionPrinterBehavior; only the steady-state phase differs (StatePermaPrint replaces the
    print->permaseed pair).

    Each monk's turn is the same cycle, mirrored and offset by a phase so seed coverage is
    continuous: blessed aura + (rit-supplied) Weapon of Quickening -> Seed on the rit -> UA
    -> animate minions off the partner's corpse -> drop UA to res the partner (which starts
    the partner's cycle) -> sac self with Dark Aura + Signet of Agony (dying to leave a fresh
    corpse for the partner's animate). The two cycles run concurrently, synchronised purely
    through the in-game alive/dead state: a monk can't act while dead, and can't animate
    without the partner's corpse."""

    @staticmethod
    def name() -> str:
        return "Perma Print"

    # Reuse StateIdentify/StateFirstDeath from the parent; only redirect first-death's exit
    # into the fused steady state instead of the separate print phase.
    class StateFirstDeath(MinionPrinterBehavior.StateFirstDeath):
        def transitions(self, ctx: "PermaPrintBehavior") -> Optional[State]:
            if self.invalid:
                return ctx.StateIdentify()
            if self.done:
                print("[PermaPrint] both monks at 1 max HP -> perma print")
                return ctx.StatePermaPrint()
            return None

    class StatePermaPrint(State):
        """Drive the two mirrored monk relay cycles plus the rit tank task. Bootstrap kills
        Monb first so Mona has a corpse to animate from on the very first turn; after that the
        two looping cycles sustain each other (each one's res kicks off the partner, each one's
        sac feeds the partner's animate)."""
        def enter(self, ctx: "PermaPrintBehavior") -> None:
            monks = ctx.roles.get(ctx.Role.MONA, [])
            rit = ctx.roles.get(ctx.Role.RITMO, [])
            self.valid = len(monks) == 2 and len(rit) == 1
            if not self.valid:
                return
            self.rit: Client = rit[0]
            self.mona: Client = monks[0]
            self.monb: Client = monks[1]
            self.step = 0
            S = ctx.Skills
            # Rit: maintain Balthazar's Spirit + Weapon of Quickening on itself and both monks,
            # and protection while Seed is down (with_seed=True is exactly this cascade).
            ctx.enqueue(ctx._rit_tank_task(self.rit, with_seed=True, monks=[self.mona, self.monb]), self)
            # Bootstrap: Monb sacrifices first so the relay has a starting corpse + a dead monk
            # for Mona to res. Without this, both cycles would deadlock waiting on a dead partner.
            self.bootstrap: Node = Sequence("pp_bootstrap", [
                CastWaitForEffect(self.monb, S.dark_aura),
                DoUntil("pp_boot_sac", lambda: not GW.Agent.IsAlive(self.monb.game_client.agent_id),
                        lambda: ctx.cast(self.monb, S.agony), interval=0.3),
            ])
            ctx.enqueue(self.bootstrap, self)

        def _monk_cycle(self, ctx: "PermaPrintBehavior", monk: 'Client', partner: 'Client') -> Sequence:
            """One monk's perpetual relay turn (loops forever). ``monk`` does the work;
            ``partner`` is the other monk whose corpse it animates and whom it resurrects."""
            S = ctx.Skills
            rit_id = self.rit.game_client.agent_id
            monk_id = monk.game_client.agent_id
            partner_id = partner.game_client.agent_id

            def animate_fired() -> bool:
                sd = monk.game_client.get_skill_data(int(S.animate_bone_minions))
                return sd is not None and sd.get_recharge != 0

            def need_minions() -> bool:
                rx, ry = GW.Agent.GetXY(rit_id)
                have = len(GW.Routines.Agents.GetFilteredMinionArray(rx, ry, 1200))
                return have < ctx.target_minion_count

            return Sequence(f"pp_cycle_{monk_id}", [
                # Wait until this monk is alive (i.e. the partner has res'd it).
                WaitUntil(lambda: GW.Agent.IsAlive(monk_id), "wait_self_alive"),
                # Blessed Aura on self + receive Weapon of Quickening (rit-supplied). Order
                # doesn't matter, so run them together and wait for both to land.
                Parallel("pp_buff", [
                    CastWaitForEffect(monk, S.blessed_aura),
                    WaitUntil(lambda: ctx.effect_remaining(monk, S.weapon_of_quickening) > 0, "has_woq"),
                ]),
                # Seed on the rit. CastWaitForEffect watches the *caster's* effects, but Seed
                # lands on the rit, so drive it with a DoUntil that watches the rit instead.
                # Refresh only when it's running low (>1000ms left == already covered by the
                # partner's recent cast), keeping it perma without redundant casts.
                DoUntil("pp_seed", lambda: ctx.effect_remaining(self.rit, S.seed_of_life) > 1000,
                        lambda: ctx.cast(monk, S.seed_of_life, rit_id), interval=0.3),
                CastWaitForEffect(monk, S.unyielding_aura),        # UA up (for the res later)
                # Need the partner's fresh corpse before animating.
                WaitUntil(lambda: not GW.Agent.IsAlive(partner_id), "wait_partner_dead"),
                # Animate minions off the corpse -- but only while below the minion target.
                DoUntil("pp_summon", lambda: (not need_minions()) or animate_fired(),
                        lambda: (ctx.cast(monk, S.animate_bone_minions) if need_minions() else None),
                        interval=0.3),
                # Drop UA to resurrect the partner -> the partner's cycle unblocks and begins.
                DoUntil("pp_res", lambda: GW.Agent.IsAlive(partner_id),
                        lambda: ctx._ua_toggle_res(monk, S.unyielding_aura), interval=0.3),
                # Sac self: Dark Aura then Signet of Agony until dead -> leaves the corpse the
                # partner (now mid-cycle) will animate from.
                CastWaitForEffect(monk, S.dark_aura),
                DoUntil("pp_sac", lambda: not GW.Agent.IsAlive(monk_id),
                        lambda: ctx.cast(monk, S.agony), interval=0.3),
            ], loop=True)

        def tick(self, ctx: "PermaPrintBehavior") -> Status:
            if not self.valid:
                return Status.RUNNING
            if self.step == 0 and self.bootstrap.status != Status.RUNNING:
                # Monb is down; launch both mirrored relay cycles. They self-sustain from here.
                ctx.enqueue(self._monk_cycle(ctx, self.mona, self.monb), self)
                ctx.enqueue(self._monk_cycle(ctx, self.monb, self.mona), self)
                self.step = 1
            return Status.RUNNING

        def transitions(self, ctx: "PermaPrintBehavior") -> Optional[State]:
            if not self.valid:
                return ctx.StateIdentify()
            return None  # runs until the operator stops the behavior


# ------------------------------------------------------- utility combat actions
# Reusable, skill-agnostic Action content for UtilityCombatBehavior. They command clients the
# same way every other behavior does -- host-side via Behavior.cast / the kernel leaves -- so
# they work whether the client is local or remote.
def _cast_once(client: 'Client', skill_id: int, name: str) -> Node:
    """Fire one self-cast and watch it to completion (CastSkill), so the selector commits to the
    cast instead of re-firing it every tick. Recharge is gated inside Behavior.cast."""
    sid = int(skill_id)
    node = CastSkill(client, sid, lambda: Behavior.cast(client, sid))
    node.name = name
    return node


class Idle(ClientAction):
    """IDLE filler so a selector always has *something* to pick (otherwise it would sit with
    no current action and re-score every tick). Does nothing for a beat, then re-evaluates."""
    tier = Tier.IDLE
    name = "idle"

    def score(self, ctx: "Behavior") -> float:
        return 0.02  # a hair above zero so it wins only when nothing real wants to act

    def build(self, ctx: "Behavior") -> Node:
        return Wait(0.25)


# A cast issued over the network takes a beat to show up in the synced skillbar/effects, so
# after firing a cast an Action must go quiet for a moment -- otherwise the selector re-picks it
# every frame (recharge/effect not visible yet) and spams the same skill. This is the utility
# analog of a global cooldown; the printer's hand-rolled cascades space casts the same way.
CAST_SETTLE = 1.0  # seconds an Action stays unapplicable after it fires a cast


class MaintainEnchant(ClientAction):
    """SUSTAINED upkeep of a self-enchant / bond. Scores higher the closer it is to expiring
    and zero while it has comfortable time left, so the selector only spends a cast when the
    buff actually needs a refresh. Not applicable while the skill is recharging or within the
    post-cast settle window (which is what stops the per-frame re-cast spam during sync lag)."""
    tier = Tier.SUSTAINED

    def __init__(self, client: 'Client', skill_id: int, refresh_below: int = 3000,
                 label: str = "enchant", settle: float = CAST_SETTLE):
        super().__init__(client)
        self.skill_id = int(skill_id)
        self.refresh_below = refresh_below
        self.settle = settle
        self._last_fire = 0.0
        self.name = f"maintain {label}"

    def applicable(self, ctx: "Behavior") -> bool:
        if time.time() - self._last_fire < self.settle:
            return False
        sd = self.client.game_client.get_skill_data(self.skill_id)
        return sd is not None and sd.slot > 0 and sd.get_recharge == 0

    def score(self, ctx: "Behavior") -> float:
        rem = Behavior.effect_remaining(self.client, self.skill_id)
        if rem >= self.refresh_below:
            return 0.0
        # 0 just below the window, rising to ~0.9 once the buff is fully down.
        return 0.9 * clamp01((self.refresh_below - rem) / self.refresh_below)

    def build(self, ctx: "Behavior") -> Node:
        self._last_fire = time.time()   # open the settle window so we don't re-fire next frame
        return _cast_once(self.client, self.skill_id, self.name)


class SelfHeal(ClientAction):
    """REACTIVE self-heal: the worse the client's health, the higher the score, so it
    preempts SUSTAINED upkeep when the client is actually hurt. Casts the first off-recharge
    heal in ``skill_ids``. Needs synced max_hp > 0 to know the health fraction."""
    tier = Tier.REACTIVE
    name = "self heal"

    def __init__(self, client: 'Client', skill_ids: List[int], heal_below: float = 0.6,
                 settle: float = CAST_SETTLE):
        super().__init__(client)
        self.skill_ids = [int(s) for s in skill_ids]
        self.heal_below = heal_below
        self.settle = settle
        self._last_fire = 0.0

    def _ready_skill(self) -> Optional[int]:
        for sid in self.skill_ids:
            sd = self.client.game_client.get_skill_data(sid)
            if sd is not None and sd.slot > 0 and sd.get_recharge == 0:
                return sid
        return None

    def _hp_fraction(self) -> Optional[float]:
        gc = self.client.game_client
        if gc.max_hp <= 0:
            return None
        return gc.hp / gc.max_hp

    def applicable(self, ctx: "Behavior") -> bool:
        if time.time() - self._last_fire < self.settle:
            return False
        frac = self._hp_fraction()
        return frac is not None and frac < self.heal_below and self._ready_skill() is not None

    def score(self, ctx: "Behavior") -> float:
        frac = self._hp_fraction()
        if frac is None:
            return 0.0
        return clamp01((self.heal_below - frac) / self.heal_below)  # 0 at the threshold -> 1 near death

    def build(self, ctx: "Behavior") -> Node:
        self._last_fire = time.time()
        sid = self._ready_skill() or self.skill_ids[0]
        return _cast_once(self.client, sid, self.name)


class RegroupAll(Action):
    """A host-coordinated team play: when the controlled clients are spread far from the
    follow target, pull them all in together (one Parallel of MoveTo). Demonstrates the whole
    reason for host-driven control -- one Action commands several clients at once. While it
    runs it ``involves`` every client it moves, so each client's own selector suspends and
    doesn't fight the move; when the Parallel completes the involvement clears and per-client
    upkeep resumes automatically (no manual flag to leak)."""
    tier = Tier.REACTIVE
    name = "regroup on leader"
    FAR = 1500.0          # spread (max client distance from leader) that opens the play
    TOL = 150.0           # how close to the leader each client must get

    def __init__(self):
        self._clients: List['Client'] = []

    @staticmethod
    def _leader_pos(ctx: "Behavior") -> Optional['Vec2']:
        login = getattr(ctx, "follow_login_number", None)
        leader_id = (GW.Party.Players.GetAgentIDByLoginNumber(login) if login
                     else GW.Party.GetPartyLeaderID())
        if not leader_id:
            return None
        return Vec2.from_tuple(GW.Agent.GetXY(leader_id))

    def _max_spread(self, ctx: "Behavior", leader: 'Vec2') -> float:
        far = 0.0
        for c in ctx.my_clients().values():
            p = Vec2.from_tuple(GW.Agent.GetXY(c.game_client.agent_id))
            far = max(far, (p - leader).magnitude())
        return far

    def applicable(self, ctx: "Behavior") -> bool:
        return (getattr(ctx, "regroup_enabled", False)
                and len(ctx.my_clients()) > 0
                and self._leader_pos(ctx) is not None)

    def score(self, ctx: "Behavior") -> float:
        leader = self._leader_pos(ctx)
        if leader is None:
            return 0.0
        far = self._max_spread(ctx, leader)
        if far < self.FAR:
            return 0.0
        return clamp01(far / (self.FAR * 3.0))  # grows with how spread out the group is

    def build(self, ctx: "Behavior") -> Node:
        leader = self._leader_pos(ctx)
        self._clients = list(ctx.my_clients().values())
        moves = [MoveTo(c, leader, tolerance=self.TOL) for c in self._clients]
        return Parallel("regroup", moves)

    def involves(self, client: 'Client') -> bool:
        return client in self._clients


# HeroAI's combat engine is a STRICT hierarchy: PrioritizeSkills() sorts the bar by a fixed
# SkillNature order, then FindCastableSkill() casts the FIRST slot that passes every hard gate
# (AreCastConditionsMet is a pure AND -- feature_count == number_of_features). Ties break by bar
# position; nothing expresses degree. We keep HeroAI's rich per-skill descriptor (Nature /
# TargetAllegiance / Conditions, a pure data table) but REPLACE the order+AND with our tier+score
# selector: Nature -> Tier + a soft base weight (so the old priority survives as a prior), the few
# magnitude Conditions -> 0..1 score modifiers, and the boolean Conditions / target existence stay
# hard gates. Everything generic-engine lives in this region: HeroAISkills (metadata),
# CustomSkillAction (one scored skill), GenericCombatEngine (assembly + combat-state multipliers).
class HeroAISkills:
    """Single holder for HeroAI's descriptor table and the Nature->Tier/score mapping derived from
    it. Replaces the scattered module-level globals so all metadata lookup is in one place. ``load``
    runs once at import; if HeroAI isn't importable, ``available`` stays False and the generic
    engine self-disables (the behavior falls back to Idle/examples only)."""
    available: bool = False
    _table = None                 # CustomSkillClass instance
    Nature = None                 # HeroAI SkillNature enum
    Target = None                 # HeroAI Skilltarget enum
    Type = None                   # HeroAI SkillType enum
    NATURE_TIER: "Dict[int, tuple]" = {}    # nature value -> (Tier, base weight)
    CUSTOM_BONUS: "Dict[int, float]" = {}   # CustomA..N nature value -> additive score bonus

    @classmethod
    def load(cls) -> None:
        try:
            from HeroAI.custom_skill import CustomSkillClass
            from HeroAI.types import SkillNature, Skilltarget, SkillType
        except Exception:
            cls.available = False
            return
        cls._table = CustomSkillClass()       # pure in-memory descriptor table, no game context
        cls.Nature, cls.Target, cls.Type = SkillNature, Skilltarget, SkillType
        N = SkillNature
        # Nature -> (Tier, base weight). The TIER does the "interrupt what I'm doing" gating
        # (a REACTIVE action preempts a running SUSTAINED one); the base weight is only a small
        # within-tier prior reproducing HeroAI's intra-priority order. Keep bases LOW so the
        # situational modifiers (HP/energy deficit, near-expiry) -- the whole point of scoring --
        # have headroom and don't saturate at 1.0, which would flatten degree back into a tie.
        cls.NATURE_TIER = {
            N.Resurrection.value:        (Tier.REACTIVE, 0.50),
            N.Healing.value:             (Tier.REACTIVE, 0.45),
            N.Interrupt.value:           (Tier.REACTIVE, 0.42),
            N.Condi_Cleanse.value:       (Tier.REACTIVE, 0.38),
            N.Hex_Removal.value:         (Tier.REACTIVE, 0.35),
            N.Enchantment_Removal.value: (Tier.REACTIVE, 0.32),
            N.EnergyBuff.value:          (Tier.SUSTAINED, 0.30),
            N.Buff.value:                (Tier.SUSTAINED, 0.26),
            N.SelfTargeted.value:        (Tier.SUSTAINED, 0.24),
            N.Neutral.value:             (Tier.SUSTAINED, 0.20),
            N.OffensiveA.value:          (Tier.SUSTAINED, 0.18),
            N.OffensiveB.value:          (Tier.SUSTAINED, 0.17),
            N.OffensiveC.value:          (Tier.SUSTAINED, 0.16),
            N.Offensive.value:           (Tier.SUSTAINED, 0.15),
        }
        # CustomA..N are HeroAI's manual "do this first" overrides that replace a skill's Nature. We
        # honor them as an additive score bonus (A highest, N lowest) so an operator-flagged skill
        # out-bids generic offense without jumping the reactive-support tier gate.
        cls.CUSTOM_BONUS = {}
        for i, letter in enumerate("ABCDEFGHIJKLMN"):
            member = getattr(N, f"Custom{letter}", None)
            if member is not None:
                cls.CUSTOM_BONUS[member.value] = 0.30 - i * 0.015   # 0.30 down to ~0.11
        cls.available = True

    @classmethod
    def descriptor(cls, skill_id: int):
        """The HeroAI ``CustomSkill`` for ``skill_id``, or None if the table isn't loaded. Unknown
        ids return a benign default descriptor (Nature=Offensive, target=Enemy)."""
        if not cls.available:
            return None
        try:
            return cls._table.get_skill(int(skill_id))
        except Exception:
            return None

    @classmethod
    def tier_and_base(cls, nature: int) -> tuple:
        return cls.NATURE_TIER.get(int(nature), (Tier.SUSTAINED, 0.15))   # default = plain offense

    @classmethod
    def custom_bonus(cls, nature: int) -> float:
        return cls.CUSTOM_BONUS.get(int(nature), 0.0)

    @classmethod
    def is_self_spec(cls, target_spec: int) -> bool:
        """True for targets the host can resolve itself (no client-side scan / no CAST_TARGETED)."""
        return cls.available and int(target_spec) == cls.Target.Self.value

    @classmethod
    def is_self_buff(cls, descriptor) -> bool:
        """A self-targeted enchant/buff to refresh only while it is NOT already up."""
        if not cls.available or descriptor is None:
            return False
        if not cls.is_self_spec(getattr(descriptor, "TargetAllegiance", -1)):
            return False
        nature = int(getattr(descriptor, "Nature", -1))
        N = cls.Nature
        return nature in (N.Buff.value, N.SelfTargeted.value, N.EnergyBuff.value)

    @classmethod
    def is_combat_only(cls, descriptor) -> bool:
        """Mirror of HeroAI ``IsOOCSkill``, inverted: True when this skill should ONLY be used in
        combat. HeroAI lets heals/cleanse/hex-removal/energy/resurrection (and any skill flagged
        ``Conditions.IsOutOfCombat``) fire out of combat; everything else -- offense, most buffs --
        waits for combat. Drives the out-of-combat utility multiplier."""
        if not cls.available or descriptor is None:
            return False
        conds = getattr(descriptor, "Conditions", None)
        if getattr(conds, "IsOutOfCombat", False):
            return False
        nature = int(getattr(descriptor, "Nature", -1))
        N = cls.Nature
        ooc = (N.Healing.value, N.Hex_Removal.value, N.Condi_Cleanse.value,
               N.EnergyBuff.value, N.Resurrection.value)
        return nature not in ooc


HeroAISkills.load()


class CustomSkillAction(ClientAction):
    """The generic default: ONE bar skill scored from its HeroAI descriptor. Replaces HeroAI's
    'first that clears every gate in a fixed order' with 'highest tier+score among the applicable'.

    applicable() -- host-evaluable hard gates only: skill on bar, off recharge, past the settle
    window, and (for a self enchant/buff) not already up. Conditions that live in the remote world
    the host can't see (target hexed, enemy casting, ...) are NOT gated here; for non-self skills the
    client resolver returns 0 when they aren't met, which suppresses the cast at the source.

    score() -- the descriptor's tier base weight + magnitude modifiers (self-HP deficit for heals,
    self-energy deficit for energy skills, near-expiry for self enchants) + any CustomA..N bonus,
    then context multipliers (currently: drop combat-only skills toward zero out of combat).

    build() -- self specs cast directly (host-known target); everything else ships the targeting
    intent via cast_targeted and lets the client pick the concrete agent in its own world."""
    def __init__(self, client: 'Client', skill_id: int, descriptor,
                 settle: float = CAST_SETTLE):
        super().__init__(client)
        self.skill_id = int(skill_id)
        self.desc = descriptor
        self.settle = settle
        self._last_fire = 0.0
        nature = int(getattr(descriptor, "Nature", -1))
        self.tier, self._base = HeroAISkills.tier_and_base(nature)
        self._custom_bonus = HeroAISkills.custom_bonus(nature)
        self._self_spec = HeroAISkills.is_self_spec(getattr(descriptor, "TargetAllegiance", -1))
        self._target_spec = int(getattr(descriptor, "TargetAllegiance",
                                        HeroAISkills.Target.Enemy.value if HeroAISkills.available else 0))
        self._self_buff = HeroAISkills.is_self_buff(descriptor)
        self._combat_only = HeroAISkills.is_combat_only(descriptor)
        self.name = f"skill {self.skill_id}"

    def applicable(self, ctx: "Behavior") -> bool:
        if time.time() - self._last_fire < self.settle:
            return False
        sd = self.client.game_client.get_skill_data(self.skill_id)
        if sd is None or sd.slot <= 0 or sd.get_recharge != 0:
            return False
        # Don't re-apply a self enchant/buff that is still up (the host knows its own effects).
        if self._self_buff and Behavior.effect_remaining(self.client, self.skill_id) > 0:
            return False
        return True

    def score(self, ctx: "Behavior") -> float:
        s = self._base + self._custom_bonus
        gc = self.client.game_client
        conditions = getattr(self.desc, "Conditions", None)

        # --- magnitude modifiers: turn HeroAI's threshold features into degree ---
        # Self-heal style: the worse our HP, the stronger the pull (only meaningful self-side).
        less_life = float(getattr(conditions, "LessLife", 0) or 0)
        if less_life > 0 and gc.max_hp > 0 and self._self_spec:
            frac = gc.hp / gc.max_hp
            if frac >= less_life:
                return 0.0          # comfortable -> don't spend the cast
            s += 0.6 * clamp01((less_life - frac) / less_life)

        # Energy skills: stronger the lower our energy is relative to the skill's threshold.
        less_energy = float(getattr(conditions, "LessEnergy", 0) or 0)
        if less_energy > 0 and gc.max_energy > 0:
            efrac = gc.energy / gc.max_energy
            s += 0.3 * clamp01((less_energy - efrac) / less_energy)

        # Self enchant upkeep: rise as it nears expiry (mirrors MaintainEnchant's curve).
        if self._self_buff:
            rem = Behavior.effect_remaining(self.client, self.skill_id)
            s += 0.3 * clamp01((4000 - rem) / 4000)

        return clamp01(s) * self._context_multiplier(ctx)

    def _context_multiplier(self, ctx: "Behavior") -> float:
        """Situational utility multipliers (extend here). Currently: a combat-only skill is scaled
        down hard when the client isn't in combat, so the engine doesn't burn offense/buffs while
        idle. The behavior provides combat state; absent that, assume in-combat (multiplier 1)."""
        mult = 1.0
        if self._combat_only:
            in_combat = getattr(ctx, "client_in_combat", None)
            if in_combat is not None and not in_combat(self.client):
                mult *= GenericCombatEngine.OUT_OF_COMBAT_FACTOR
        return mult

    def build(self, ctx: "Behavior") -> Node:
        self._last_fire = time.time()       # settle window: a backstop while recharge re-syncs
        if self._self_spec:
            fire = (lambda: Behavior.cast(self.client, self.skill_id))
        else:
            fire = (lambda: Behavior.cast_targeted(self.client, self.skill_id, self._target_spec))
        node = CastSkill(self.client, self.skill_id, fire)
        node.name = self.name
        return node


class GenericCombatEngine:
    """Assembles the generic default loadout: one CustomSkillAction per bar skill that HeroAI has a
    descriptor for and no detected synergy claims (composite-build skills stay owned by their synergy
    module). Stateless -- all knobs are class constants -- so it reads as one unit and the behavior
    just calls ``actions_for``."""
    OUT_OF_COMBAT_FACTOR = 0.0    # multiplier on a combat-only skill's utility while out of combat

    @staticmethod
    def claimed_skills(client: 'Client', synergies: List[Synergy]) -> Set[int]:
        """Skills owned by a detected synergy (required + optional) -- excluded from the default."""
        bar = set(client.game_client.skills.keys())
        claimed: Set[int] = set()
        for syn in detect_synergies(bar, synergies):
            claimed |= set(syn.skills) | set(syn.optional_skills)
        return claimed

    @classmethod
    def actions_for(cls, client: 'Client', synergies: List[Synergy]) -> List[Action]:
        if not HeroAISkills.available:
            return []
        claimed = cls.claimed_skills(client, synergies)
        acts: List[Action] = []
        for skill_id in client.game_client.skills.keys():
            if skill_id in claimed:
                continue
            desc = HeroAISkills.descriptor(skill_id)
            if desc is None:
                continue
            acts.append(CustomSkillAction(client, skill_id, desc))
        return acts


class UtilityCombatBehavior(Behavior):
    """Host-driven utility combat. Every controlled client runs its OWN UtilitySelector, built
    from generic actions plus the action modules of whatever synergies its skillbar carries --
    so a bar that holds several skill combinations contributes all of their actions to one
    selector, which arbitrates them by tier + score (no per-combo special-casing). A separate
    behavior-level team selector runs coordinated plays that command several clients at once
    (RegroupAll), which is the host-coordination this design favors over client-local AI.
    Everything runs host-side over each client's transport, exactly like the permaseed behaviors.

    The example synergies / actions below are a starting point meant to be replaced with the
    real synergy library; the engine itself (UtilitySelector, Synergy, detect_synergies) is the
    durable part."""
    class Skills(int, Enum):
        balthazars_spirit = 242
        weapon_of_quickening = 1268
        shielding_hands = 299
        shield_of_absorption = 1399
        heal_area = 280
        kareis_healing_circle = 1119

    # A bar that carries all three of these is a self-protecting tank; detected independently of
    # any other combos on the bar.
    SYNERGIES: List[Synergy] = [
        Synergy("rit_tank",
                frozenset({Skills.weapon_of_quickening.value,
                           Skills.shielding_hands.value,
                           Skills.shield_of_absorption.value}),
                provides="tank"),
    ]

    def __init__(self, thread_globals: ThreadManager, network_manager: NetworkManager,
                 clients_provider: Optional[Callable[[], Dict[uuid.UUID, 'Client']]] = None):
        super().__init__(thread_globals, network_manager, clients_provider)
        self.follow_login_number: Optional[int] = None
        # The "leader" anchors team-wide combat state: a client counts as in combat if enemies are
        # near IT or near the selected leader (so a back-line caster with no enemy in earshot still
        # fights once the leader engages). Defaults to the party leader until the operator picks one.
        self.leader_login_number: Optional[int] = None
        self._combat_cache: Dict['Client', tuple] = {}   # client -> (timestamp, in_combat); short TTL
        # Start INERT: with no real synergy library authored yet, the behavior should command
        # nothing until the operator opts in. The example actions below (balth/heal upkeep, the
        # rit_tank synergy, regroup) are demo content to validate the engine, not a default loadout
        # -- so they ship OFF and are toggled on in draw().
        self.examples_enabled = False
        # The HeroAI-descriptor generic engine IS the default action set for unclaimed skills, so it
        # ships ON. It self-disables if HeroAI's descriptor table failed to import. Conservative by
        # construction: settle-gated, self-buffs skip while up, combat-only skills idle out of combat,
        # and the client suppresses any cast whose target doesn't resolve.
        self.generic_enabled = HeroAISkills.available
        self.regroup_enabled = True
        self.team_actions: List[Action] = [RegroupAll()]
        self.team_selector: Optional[UtilitySelector] = None
        self.selectors: Dict['Client', UtilitySelector] = {}
        self.client_action_label: Dict['Client', Optional[str]] = {}

    @staticmethod
    def name() -> str:
        return "Utility Combat"

    def build_root(self) -> Node:
        return StateMachine(lambda: UtilityCombatBehavior.StateCombat())

    # ---- loadout assembly: bar -> actions (generic + per-synergy modules) ----
    def _generic_actions(self, client: 'Client') -> List[Action]:
        S = self.Skills
        bar = set(client.game_client.skills.keys())
        acts: List[Action] = [Idle(client)]
        if S.balthazars_spirit.value in bar:
            acts.append(MaintainEnchant(client, S.balthazars_spirit.value, 2000, "balthazar's spirit"))
        heals = [s.value for s in (S.heal_area, S.kareis_healing_circle) if s.value in bar]
        if heals:
            acts.append(SelfHeal(client, heals))
        return acts

    def _synergy_actions(self, client: 'Client') -> List[Action]:
        S = self.Skills
        modules: Dict[str, Callable[['Client'], List[Action]]] = {
            "rit_tank": lambda c: [
                MaintainEnchant(c, S.weapon_of_quickening.value, 4000, "weapon of quickening"),
                MaintainEnchant(c, S.shielding_hands.value, 3000, "shielding hands"),
                MaintainEnchant(c, S.shield_of_absorption.value, 3000, "shield of absorption"),
            ],
        }
        bar = set(client.game_client.skills.keys())
        acts: List[Action] = []
        for syn in detect_synergies(bar, self.SYNERGIES):
            acts += modules.get(syn.name, lambda c: [])(client)
        return acts

    def _client_actions(self, client: 'Client') -> List[Action]:
        acts: List[Action] = [Idle(client)]
        if self.generic_enabled:                               # the default engine for unclaimed skills
            acts += GenericCombatEngine.actions_for(client, self.SYNERGIES)
        if self.examples_enabled:                              # demo content, separate opt-in
            acts += self._generic_actions(client) + self._synergy_actions(client)
        return acts

    # ---- combat state (host-side; commander shares the clients' map instance) ----
    def _enemies_near(self, agent_id: int) -> bool:
        """True if any enemy is within Earshot of ``agent_id``. Scanned host-side: agent ids are
        consistent within the shared server instance, so the host sees the same world the clients
        do (same pattern RegroupAll uses for GW.Agent.GetXY on remote clients)."""
        if not agent_id:
            return False
        try:
            x, y = GW.Agent.GetXY(agent_id)
        except Exception:
            return False
        if not x and not y:
            return False
        enemies = Routines.Agents.GetFilteredEnemyArray(x, y, int(GW.Range.Earshot.value))
        return len(enemies) > 0

    def client_in_combat(self, client: 'Client') -> bool:
        """Combat state for one client, cached with a short TTL (every action queries it each tick).
        In combat if enemies are near the client OR near the selected leader (team-wide anchor)."""
        now = time.time()
        cached = self._combat_cache.get(client)
        if cached is not None and now - cached[0] < 0.25:
            return cached[1]
        in_combat = self._enemies_near(client.game_client.agent_id)
        if not in_combat and self.leader_login_number:
            leader_id = GW.Party.Players.GetAgentIDByLoginNumber(self.leader_login_number)
            in_combat = self._enemies_near(leader_id)
        self._combat_cache[client] = (now, in_combat)
        return in_combat

    def _coordinating(self, client: 'Client') -> bool:
        """True while the team selector is running a coordinated play that owns this client."""
        a = self.team_selector.current_action if self.team_selector else None
        return a is not None and a.involves(client)

    @staticmethod
    def _login_dropdown(label: str, imgui_id: str, current: Optional[int]) -> Optional[int]:
        """A 'pick a party player' dropdown keyed by login_number (survives map loads). Defaults to
        the party leader when ``current`` isn't a present player. Returns the chosen login_number."""
        players = GW.Party.GetPlayers() or []
        logins = [p.login_number for p in players]
        names = [GW.Party.Players.GetPlayerNameByLoginNumber(l) or f"Player {l}" for l in logins]
        PyImGui.text(label)
        PyImGui.same_line(0.0, 0.0)
        if not names:
            PyImGui.text("(no party players)")
            return current
        if current in logins:
            idx = logins.index(current)
        else:
            leader_login = GW.Party.Players.GetLoginNumberByAgentID(GW.Party.GetPartyLeaderID())
            idx = logins.index(leader_login) if leader_login in logins else 0
            current = logins[idx]
        new_idx = PyImGui.combo(imgui_id, idx, names)
        if 0 <= new_idx < len(logins):
            current = logins[new_idx]
        return current

    def draw(self) -> None:
        if HeroAISkills.available:
            self.generic_enabled = PyImGui.checkbox("Generic engine (unclaimed skills)", self.generic_enabled)
        else:
            PyImGui.text("Generic engine: HeroAI metadata unavailable")
        self.examples_enabled = PyImGui.checkbox("Enable example actions", self.examples_enabled)
        if self.examples_enabled:
            self.regroup_enabled = PyImGui.checkbox("Regroup on follow target", self.regroup_enabled)

        # Leader: anchors team-wide combat state (enemies near the leader put the whole team in
        # combat). Follow: which player RegroupAll pulls toward.
        self.leader_login_number = self._login_dropdown("Leader:", "#ucleader", self.leader_login_number)
        self.follow_login_number = self._login_dropdown("Follow:", "#ucfollow", self.follow_login_number)

        # Live readout of what each client's selector is currently doing -- the fastest way to
        # eyeball-validate the engine in-game.
        coord = self.team_selector.current_action.name if (self.team_selector and
                                                           self.team_selector.current_action) else None
        PyImGui.text(f"Team: {coord or 'idle'}")
        for client in list(self.selectors.keys()):
            label = self.client_action_label.get(client) or "-"
            PyImGui.text(f"  {client.game_client.name or 'client'}: {label}")

    class StateCombat(State):
        """Build a per-client UtilitySelector from each client's loadout and tick them all
        every frame, plus the behavior's team selector. Selectors are rebuilt whenever the
        loadout signature changes (a client joins/leaves, its skillbar syncs, or the synergies
        on its bar change), so live re-assignment and late skillbar arrival are handled."""
        def enter(self, ctx: "UtilityCombatBehavior") -> None:
            # Team plays are example content too -> empty until opted in (read live, no rebuild).
            ctx.team_selector = UtilitySelector(
                "team", lambda: ctx.team_actions if ctx.examples_enabled else [])
            ctx.selectors = {}
            ctx.client_action_label = {}
            self._sig: Optional[dict] = None
            self._rebuild(ctx)

        def _signature(self, ctx: "UtilityCombatBehavior") -> dict:
            sig = {"__examples__": ctx.examples_enabled,    # toggling rebuilds per-client loadouts
                   "__generic__": ctx.generic_enabled}
            for cid, c in ctx.my_clients().items():
                if c.game_client is None:
                    sig[cid] = (False, ())
                    continue
                bar = set(c.game_client.skills.keys())
                syns = tuple(s.name for s in detect_synergies(bar, ctx.SYNERGIES))
                sig[cid] = (bool(bar), syns)
            return sig

        def _rebuild(self, ctx: "UtilityCombatBehavior") -> None:
            ctx.selectors = {}
            for cid, client in ctx.my_clients().items():
                if client.game_client is None:
                    continue
                actions = ctx._client_actions(client)
                # Bind client into each closure so every selector targets its own client.
                on_pick = (lambda c: (lambda nm: ctx.client_action_label.__setitem__(c, nm)))(client)
                ctx.selectors[client] = UtilitySelector(
                    f"sel_{cid}",
                    (lambda a=actions: a),
                    suspended=(lambda c=client: ctx._coordinating(c)),
                    on_pick=on_pick)
            self._sig = self._signature(ctx)

        def tick(self, ctx: "UtilityCombatBehavior") -> Status:
            if self._signature(ctx) != self._sig:
                self._rebuild(ctx)
            if ctx.team_selector is not None:
                ctx.team_selector.tick(ctx)
            for sel in list(ctx.selectors.values()):
                sel.tick(ctx)
            return Status.RUNNING




class BehaviorSlot:
    """One behavior (running or stopped) plus the set of clients assigned to it. A slot owns
    its own worker thread name and behavior instance so slots start/stop independently."""
    _counter = 0

    def __init__(self, behavior_cls: Type['Behavior']):
        BehaviorSlot._counter += 1
        self.id = BehaviorSlot._counter
        self.behavior_cls = behavior_cls
        self.assigned: Set[uuid.UUID] = set()
        self.instance: Optional['Behavior'] = None
        self.thread_name = f"BEHAVIOR_{self.id}"

    @property
    def running(self) -> bool:
        return self.instance is not None and self.instance.is_running


class BehaviorManager:
    """Owns all behavior slots and the client->behavior assignment. Invariant: a client is
    ruled by AT MOST ONE slot (assigning moves it off any other), so behaviors never fight
    over the same client. Clients not in any slot are 'unassigned' (the pool)."""
    def __init__(self, network: 'NetworkManager', thread_manager: ThreadManager):
        self.network = network
        self.thread_manager = thread_manager
        self.slots: List[BehaviorSlot] = []

    # --- slots ---
    def add_slot(self, behavior_cls: Type['Behavior']) -> BehaviorSlot:
        slot = BehaviorSlot(behavior_cls)
        self.slots.append(slot)
        return slot

    def remove_slot(self, slot: BehaviorSlot) -> None:
        self.stop(slot)                      # its clients simply fall back to the pool
        if slot in self.slots:
            self.slots.remove(slot)

    # --- assignment (mutually exclusive) ---
    def assign(self, client_id: uuid.UUID, slot: BehaviorSlot) -> None:
        for s in self.slots:                 # guarantee a single owner
            s.assigned.discard(client_id)
        slot.assigned.add(client_id)

    def unassign(self, client_id: uuid.UUID) -> None:
        for s in self.slots:
            s.assigned.discard(client_id)

    def unassigned_ids(self) -> List[uuid.UUID]:
        owned: Set[uuid.UUID] = set()
        for s in self.slots:
            owned |= s.assigned
        # Snapshot the keys with list() FIRST (a single atomic C-level copy, GIL-safe) before
        # the Python-level filter below. Iterating client_list.keys() directly in a comprehension
        # runs Python bytecode per element, so if the network thread inserts/removes a client
        # mid-iteration it raises "dictionary changed size during iteration" -- which used to
        # blow up the UI draw mid-frame and unbalance ImGui's begin/end stack (the flicker).
        keys = list(self.network.client_list.keys())
        return [cid for cid in keys if cid not in owned]

    # --- lifecycle ---
    def start(self, slot: BehaviorSlot) -> None:
        if slot.running:
            return
        # The provider is a LIVE view of the slot's assignment, so re-assigning clients while
        # the behavior runs is reflected without a restart (next identify_players tick).
        b = slot.behavior_cls(
            self.thread_manager, self.network,
            clients_provider=lambda s=slot: {cid: self.network.client_list[cid]
                                             for cid in s.assigned
                                             if cid in self.network.client_list})
        slot.instance = b
        self.thread_manager.thread_manager.add_thread(slot.thread_name, b.run)

    def stop(self, slot: BehaviorSlot) -> None:
        if slot.instance is not None:
            slot.instance.stop()             # break its run loop
            self.thread_manager.thread_manager.stop_thread(slot.thread_name)
            slot.instance = None

    def stop_all(self) -> None:
        for s in self.slots:
            self.stop(s)


BEHAVIOR_MAP = {
            0: TestBehavior,
            1: MinionPrinterBehavior,
            2: PermaPrintBehavior,
            3: UtilityCombatBehavior
        }