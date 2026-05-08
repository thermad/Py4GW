# region Imports
# from __future__ import annotations
import os
import sys
import random
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict

import Py4GW  # type: ignore
import PyEffects
from Py4GWCoreLib import IniHandler
from Py4GWCoreLib import PyImGui
from Py4GWCoreLib import Routines
from Py4GWCoreLib import Timer
import Py4GWCoreLib as GW
import time
import socket
from typing import List, Dict, Protocol, ParamSpec, TypeVar, Any, Callable, Set, Literal, Type
from dataclasses import dataclass, field
import json
from enum import Enum, auto
import math
import uuid


lib_path = sys.prefix + "\\Py4GWCoreLib\\ExternalLibs"
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
import Py4GWCoreLib.ExternalLibs.zmq as zmq
# endregion


# region Jsonizers
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
        d["target_id"] = GW.Player.GetTargetID()
        d["hp"] = GW.Agent.GetHealth(d["id"])
        d["max_hp"] = GW.Agent.GetMaxHealth(d["id"])
        d["energy"] = GW.Agent.GetEnergy(d["id"])
        d["max_energy"] = GW.Agent.GetMaxEnergy(d["id"])
        d["skilldata"] = Jsonizer.get_skilldata()
        d["effects"] = Jsonizer.get_effects(d["id"])
        d["buffs"] = Jsonizer.get_buffs(d["id"])
        return d


# endregion


# region RPC setup
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
# endregion


# region MiscHelperClasses
class MultithreadBoosterCache:
    def __init__(self):
        self.timer = time.time()
        self.fps_poll = time.time()
        self.load_delay = time.time()
        self.fps = 60


class Vec2:
    __slots__ = ("x", "y")

    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    @classmethod
    def from_tuple(cls, values):
        if len(values) != 2:
            raise ValueError("Tuple must have exactly 2 elements")
        return cls(values[0], values[1])

    def __repr__(self):
        return f"Vec2({self.x}, {self.y})"

    # --- Basic arithmetic ---

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __neg__(self):
        return Vec2(-self.x, -self.y)

    def __mul__(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        if scalar == 0:
            raise ZeroDivisionError("Division by zero")
        inv = 1.0 / scalar
        return Vec2(self.x * inv, self.y * inv)

    # --- Comparisons ---

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    # --- Vector math ---

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def cross(self, other):
        """2D cross product (scalar result)"""
        return self.x * other.y - self.y * other.x

    def magnitude(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def magnitude_squared(self):
        """Avoids sqrt — useful for comparisons"""
        return self.x * self.x + self.y * self.y

    def normalized(self):
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize zero vector")
        inv = 1.0 / mag
        return Vec2(self.x * inv, self.y * inv)

    def distance_to(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def distance_squared_to(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy

    # --- Geometry helpers ---

    def angle(self):
        """Angle from x-axis in radians"""
        return math.atan2(self.y, self.x)

    def angle_to(self, other):
        """Signed angle to another vector"""
        return math.atan2(self.cross(other), self.dot(other))

    def rotated(self, radians):
        cos_r = math.cos(radians)
        sin_r = math.sin(radians)
        return Vec2(
            self.x * cos_r - self.y * sin_r,
            self.x * sin_r + self.y * cos_r
        )

    def perpendicular(self):
        """90° counterclockwise"""
        return Vec2(-self.y, self.x)

    # --- Utility ---

    def copy(self):
        return Vec2(self.x, self.y)

    def as_tuple(self):
        return (self.x, self.y)

    def as_list(self):
        return [self.x, self.y]


class ThreadManager:
    def __init__(self):
        self.thread_manager = GW.MultiThreading(2.0, log_actions=True)
        self.is_threads_running = False

    def start_thread(self, name, func):
        self.is_threads_running = True
        # Add sequential threads
        self.thread_manager.add_thread(name, func)
        # Watchdog thread is necessary to async close other running threads
        self.thread_manager.start_watchdog(name)

    def stop_sequential_environment(self):
        self.thread_manager.stop_all_threads()
        self.is_threads_running = False

# endregion


# region blackboard
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
        self.target_id = 0
        self.hp = 0
        self.max_hp = 0
        self.energy = 0
        self.max_energy = 0
        # Use dictionaries for these as they are dynamic sets of effects/skills
        self.skills: Dict[int, SkillData] = {}
        self.effects: Dict[int, EffectData] = {}
        self.buffs: Dict[int, BuffData] = {}

    def update_from_dict(self, data: Dict[str, Any]):
        """converts dictionary to object attributes"""
        self.agent_id = data.get("id", self.agent_id)
        self.target_id = data.get("target_id", self.target_id)
        self.hp = data.get("hp", self.hp)
        self.max_hp = data.get("max_hp", self.max_hp)
        self.energy = data.get("energy", self.energy)
        self.max_energy = data.get("max_energy", self.max_energy)
        # print(f"""Skill update data {data.get("skilldata", {}).items()}""")
        # Update Skills
        for s_id, s_val in data.get("skilldata", {}).items():
            self.skills[int(s_id)] = SkillData(**s_val)
        # Update Effects
        for e_id, e_val in data.get("effects", {}).items():
            self.effects[int(e_id)] = EffectData(**e_val)
        # Update Buffs
        for b_id, b_val in data.get("buffs", {}).items():
            self.buffs[int(b_id)] = BuffData(**b_val)

    def get_buff_data(self, b_id) -> BuffData:
        return self.buffs.get(b_id, None)

    def get_skill_data(self, s_id) -> SkillData:
        return self.skills.get(s_id, None)
# endregion


# region client transport
class Client:
    class Transport(ABC):
        @abstractmethod
        def send(self, method :RPC.CMD, *args: object, **kwargs: object):
            ...

    def __init__(self, game: GameClient, transport: Transport):
        self.game_client = game
        self.transport = transport


class LocalTransport(Client.Transport):
    def send(self, method: RPC.CMD, *args: object, **kwargs: object) -> None:
        RPC.call(method, args, kwargs)
# endregion


# region networking
class ZMQTransport(Client.Transport):
    def __init__(self, router, client_id: uuid):
        self.client_id = client_id
        self.router = router

    def send(self, method: RPC.CMD, *args: object, **kwargs: object) -> None:
        self.router.send_multipart([
            self.client_id.bytes,
            b"",
            json.dumps({
                "method": method,
                "args": args,
                "kwargs": kwargs
            }).encode()
        ])


class NetworkManager:
    def __init__(self, thread_man: ThreadManager, local_game_client: GameClient):
        self.thread_manager = thread_man
        self.client_list: Dict[uuid.UUID, Client] = {}
        client_id = uuid.uuid4()
        c = Client(local_game_client, LocalTransport())
        self.client_list[client_id] = c #Adding the local client first helps

        # Host Sockets
        self.router = None  # RPC (Bidirectional)
        self.pub = None     # Broadcast (Host -> All)
        self.pull = None    # State Updates (All -> Host)

        self.context = None
        self.poller = None

    def setup_zmq_host(self, port):
        self.context = zmq.Context()

        # ROUTER: Handles RPC requests from clients and sends direct replies
        self.router = self.context.socket(zmq.ROUTER)
        self.router.bind(f"tcp://*:{port}")

        # PUB: Broadcasts data to all connected clients
        self.pub = self.context.socket(zmq.PUB)
        self.pub.bind(f"tcp://*:{port + 1}")

        # PULL: Collects state updates from all clients (Async/Non-blocking)
        self.pull = self.context.socket(zmq.PULL)
        self.pull.bind(f"tcp://*:{port + 2}")

        self.poller = zmq.Poller()
        self.poller.register(self.router, zmq.POLLIN)
        self.poller.register(self.pull, zmq.POLLIN)

    def broadcast(self, method: str, args: list = None, kwargs: dict = None):
        """
        Sends a message to ALL connected clients via the PUB socket.
        Best for: 'Move to X', 'Attack Target Y', 'Global Sync'.
        """
        if not self.pub:
            print("Error: PUB socket not initialized.")
            return

        payload = {
            "method": method,
            "args": args or [],
            "kwargs": kwargs or {}
        }
        # PUB sockets send as a single frame unless multipart is specifically needed
        self.pub.send_json(payload)

    def maintain_host(self, port, terminal_function):
        self.setup_zmq_host(port)
        try:
            while self.thread_manager.is_threads_running:
                events = dict(self.poller.poll(timeout=10))

                # 1. Handle RPC Requests (ROUTER)
                if self.router in events:
                    # ROUTER format: [identity, empty, payload]
                    identity, _, message = self.router.recv_multipart()
                    data = json.loads(message)
                    # Convert the ZMQ identity bytes to a UUID object
                    client_id = uuid.UUID(bytes=identity)

                    if data.get("method") == "handshake":
                        # Create the identity response
                        reply_data = {
                            "status": "registered"
                        }
                        reply = json.dumps(reply_data).encode()
                        print(f"Registering client {client_id}")
                        # Send back to the specific identity
                        self.router.send_multipart([identity, b"", reply])

                        # Initialize client in your tracking lists
                        if client_id not in self.client_list:
                            c = Client(GameClient(), ZMQTransport(self.router, client_id))
                            self.client_list[client_id] = c
                    # result = registry.call(
                    #     data["method"],
                    #     data.get("args", []),
                    #     data.get("kwargs", {})
                    # )
                    # reply = json.dumps({
                    #     "method": data["method"],
                    #     "returned": result
                    # }).encode()
                    # self.router.send_multipart([identity, b"", reply])

                # 2. Handle State Updates (PULL)
                if self.pull in events:
                    # PULL format: [payload]
                    msg = self.pull.recv_json()
                    cid_str = msg.get("client_id")
                    if cid_str:
                        client_id = uuid.UUID(cid_str)
                        if client_id not in self.client_list:
                            print("Push notification from unknown client")
                        else:
                            self.client_list[client_id].game_client.update_from_dict(msg)

                # 3. Optional: Broadcast global state to all clients via PUB
                # self.pub.send_json({"type": "sync", "data": global_state})

        finally:
            self._cleanup_host()
            if terminal_function: terminal_function()

    def maintain_client(self, ip, port, terminal_function):
        context = zmq.Context()
        dealer = None
        sub = None
        push = None
        try:
            client_id = uuid.uuid4()
            # DEALER: RPC communication with Host ROUTER
            dealer = context.socket(zmq.DEALER)
            dealer.setsockopt(zmq.IDENTITY, client_id.bytes)
            dealer.connect(f"tcp://{ip}:{port}")
            #add a handshake
            # Send a request to the host to ask "Who am I?"
            print("Beginning handshake")
            handshake_msg = json.dumps({"method": "handshake"}).encode()
            dealer.send_multipart([b"", handshake_msg])
            _, response_raw = dealer.recv_multipart()
            response = json.loads(response_raw)
            if response.get("status"):
                print(f"Connected, server okay'd {client_id}")
            else:
                print("Server didn't okay, aborting")
                return

            # SUB: Listen for broadcasts from Host PUB
            sub = context.socket(zmq.SUB)
            sub.connect(f"tcp://{ip}:{port + 1}")
            sub.setsockopt_string(zmq.SUBSCRIBE, "")

            # PUSH: Send state updates to Host PULL
            push = context.socket(zmq.PUSH)
            push.connect(f"tcp://{ip}:{port + 2}")

            poller = zmq.Poller()
            poller.register(dealer, zmq.POLLIN)
            poller.register(sub, zmq.POLLIN)

            while self.thread_manager.is_threads_running:
                # A. Send local state update to host
                # A. Send local state update to host
                state = Jsonizer.player_data()
                state["client_id"] = str(client_id)
                push.send_json(state)

                # B. Check for incoming messages
                events = dict(poller.poll(timeout=10))

                if dealer in events:
                    # Handle RPC response from Host
                    msg_full = dealer.recv_multipart()
                    msg: Dict = json.loads(msg_full[-1].decode('utf-8'))

                    # Use registry to handle the returned data (e.g., updating local state)
                    method = msg.get("method", None)
                    args = msg.get("args", [])
                    kwargs = msg.get("kwargs", {})
                    if method:
                        ret = RPC.call(method, *args, **kwargs)
                        j: Dict = {"method": method, "returned": ret}
                        dealer.send_multipart([b"", json.dumps(j).encode()])

                if sub in events:
                    broadcast_data = sub.recv_json()
                    # Assume the broadcast message looks like: {"method": "move_to", "args": [100, 200]}
                    RPC.call(
                        broadcast_data.get("method"),
                        *broadcast_data.get("args", []),
                        **broadcast_data.get("kwargs", {})
                    )

                time.sleep(0.05)
        finally:
            if dealer: dealer.close()
            if sub: sub.close()
            if push: push.close()
            context.term()
            if terminal_function: terminal_function()

    def _cleanup_host(self):
        if self.router: self.router.close()
        if self.pub: self.pub.close()
        if self.pull: self.pull.close()
        if self.context: self.context.term()
        self.client_list.clear()
# endregion


# region behavior logic
class ScheduledTask(ABC):
    """Abstract base class for a task that runs on a timer."""
    def __init__(self, interval: float):
        self.interval = interval
        self.last_run = 0.0
        self.terminate = False

    def is_ready(self) -> bool:
        now = time.time()
        if now - self.last_run >= self.interval:
            return True
        return False

    def tick(self, context):
        """Handles the timing and execution logic."""
        if self.is_ready():
            self.last_run = time.time()
            return self.execute(context)
        return None

    @abstractmethod
    def execute(self, context) -> any:
        """The actual logic to perform. 'context' is usually the Behavior instance."""
        pass


class TaskScheduler:
    """Container to manage and execute multiple ScheduledTasks."""
    def __init__(self):
        self.tasks: list[ScheduledTask] = []

    def add_task(self, task: ScheduledTask):
        self.tasks.append(task)

    def run_pending(self, context):
        """Call this every frame/tick in the main loop."""
        for task in self.tasks:
            task.tick(context)
        self.tasks = [t for t in self.tasks if not t.terminate]


class State:
    def enter(self, bot): pass
    def execute(self, bot): pass
    def exit(self, bot): pass


class Behavior:
    def __init__(self, thread_globals: ThreadManager, network_manager: NetworkManager):
        self.cache_thread_globals: ThreadManager = thread_globals
        self.network_manager = network_manager
        self.current_state: State = State()
        self.scheduler: TaskScheduler = TaskScheduler()

    def run(self):
        self.current_state.execute(self)
        self.scheduler.run_pending(self)

    @staticmethod
    def name() -> str:
        ...

    def draw(self):
        pass

    def wait_for_thread(self, thread_name):
        while self.cache_thread_globals.is_threads_running and self.cache_thread_globals.thread_manager.threads.get(
                thread_name, 0) != 0:
            time.sleep(0.01)

    class CastWaitForEffect(ScheduledTask):
        def __init__(self, interval: float, client: Client, skill_id: int):
            super().__init__(interval)
            self.client = client
            self.skill_id = skill_id

        def execute(self, context) -> any:
            if self.client.game_client.effects.get(self.skill_id, False):
                self.terminate = True
                return
            self.client.transport.send(RPC.CMD.USE_SKILL, self.skill_id, self.client.game_client.agent_id)

    class WaitForBool(ScheduledTask):
        def __init__(self, interval: float, check_func):
            super().__init__(interval)
            self.check_func = check_func

        def execute(self, context) -> any:
            if self.check_func():
                self.terminate = True
                return

    class DoUntil(ScheduledTask):
        def __init__(self, interval: float, check_func, do_function):
            super().__init__(interval)
            self.check_func = check_func
            self.do_function = do_function

        def execute(self, context) -> any:
            self.do_function()
            if self.check_func():
                self.terminate = True
                return


class TestBehavior(Behavior):
    def run(self):
        print("=== TestBehavior started ===")
        while self.cache_thread_globals.is_threads_running:
            if len(self.network_manager.client_list) > 0:
                print(f"Test Behavior client_list: {self.network_manager.client_list}")
                s: SkillData = next(iter(self.network_manager.client_list.values())).game_client.get_skill_data(268)
                if s:
                    print(f"First client UA: {s.recharge}, {s.get_recharge}, {s.slot}, {s.id}")
                self.network_manager.broadcast(RPC.CMD.MOVE.value, list(GW.Player.GetXY()))
            else:
                print("Waiting for a client to connect")
            time.sleep(3)


    @staticmethod
    def name() -> str:
        return "Test Behavior"


class MinionPrinterBehavior(Behavior):
    class StateInitializing(State):
        def execute(self, bot: 'MinionPrinterBehavior'):
            bot.roles = bot.identify_players()
            c: Client
            result = {
                role.name: [c.game_client.agent_id for c in clients]
                for role, clients in bot.roles.items()
            }
            if len(result[bot.Role.MONA]) == 2 and len(result[bot.Role.RITMO]) == 1:
                print(f"Identified: {result}. Moving to next state.")
                bot.current_state = bot.StateFirstDeath()
            time.sleep(1)

    class StateFirstDeath(State):
        def execute(self, bot: 'MinionPrinterBehavior'):
            monks: List[Client] = bot.roles[bot.Role.MONA]
            ritmo: List[Client] = bot.roles[bot.Role.RITMO]
            da_wait = Behavior.CastWaitForEffect(0.5, monks[0], bot.Skills.dark_aura)
            ua_wait = Behavior.CastWaitForEffect(0.5, monks[1], bot.Skills.unyielding_aura)
            bot.scheduler.add_task(da_wait)
            bot.scheduler.add_task(ua_wait)
            while bot.cache_thread_globals.is_threads_running and (not da_wait.terminate or not ua_wait.terminate):
                bot.scheduler.run_pending(bot)

            def cast_agony():
                monks[0].transport.send(RPC.CMD.USE_SKILL, bot.Skills.agony)

            def wait_dead() -> bool:
                return monks[0].game_client.hp == 0

            cast_agony_until_dead = Behavior.DoUntil(2.0, wait_dead, cast_agony)
            bot.scheduler.add_task(cast_agony_until_dead)
            while bot.cache_thread_globals.is_threads_running and not cast_agony_until_dead.terminate:
                bot.scheduler.run_pending(bot)



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

    @dataclass
    class Signature:
        required_skills: Set[int]

    SIGNATURES = {
        Role.RITMO: Signature({Skills.weapon_of_quickening.value, Skills.shield_of_absorption.value,
                               Skills.shielding_hands.value, Skills.heal_area.value, Skills.kareis_healing_circle.value,
                               Skills.balthazars_spirit.value}),
        Role.MONA: Signature({Skills.unyielding_aura.value, Skills.seed_of_life.value,
                              Skills.blessed_aura.value, Skills.animate_bone_minions.value, Skills.dark_aura.value, Skills.agony.value})
    }

    def __init__(self, thread_globals: ThreadManager, network_manager: NetworkManager):
        super().__init__(thread_globals, network_manager)
        self.target_minion_count = 20
        self.roles: Dict[MinionPrinterBehavior.Role, List[Client]] = defaultdict()

    @staticmethod
    def name() -> str:
        return "Minion Printer"

    def state_rectify(self) -> State:
        monks = self.roles[self.Role.MONA]
        rit = self.roles[self.Role.RITMO]
        if len(monks) != 2 or len(rit) != 1:
            return self.StateInitializing()
        if monks[0].game_client.max_hp > 1 or monks[1].game_client.max_hp > 1:
            pass

    def draw(self):
        PyImGui.text("Minion Goal:")
        PyImGui.same_line(0.0, 0.0)
        self.target_minion_count = PyImGui.input_int("#binputminions", self.target_minion_count)

    def identify_players(self) -> Dict[Role, list[Client]]:
        assignments: Dict[MinionPrinterBehavior.Role, List[Client]] = defaultdict()
        claimed_clients = set()  # Store the object IDs or client IDs

        # Order matters here! Put the most specific/unique roles first.
        role_priority = [self.Role.RITMO, self.Role.MONA, self.Role.MONB]

        for client_id, client in self.network_manager.client_list.items():
            if client.game_client is None:
                continue
            for role in role_priority:
                sig = self.SIGNATURES[role]
                client_skills = set(client.game_client.skills.keys())
                if sig.required_skills.issubset(client_skills):
                    assignments[role].append(client)
                    break

        return assignments

    def run(self):
        self.current_state = self.StateInitializing()
        while self.cache_thread_globals.is_threads_running:
            super().run()




# endregion


# region UI
class UICache:
    def __init__(self):
        self.WIDTH = 300
        self.ip_entry = [127, 0, 0, 1]
        self.port = 54321
        self.is_host = False
        self.is_connected = False
        self.ui_timer = 0
        self.confirm = False
        self.header_child_height = 60
        self.behavior_select_index = 0
        self.network_manager: NetworkManager = None
        self.BEHAVIOR_THREAD_NAME = "BEHAVIOR_THREAD"
        self.NETWORK_THREAD_NAME = "NETWORK_THREAD"
        self.behavior_map = {
            0: TestBehavior,
            1: MinionPrinterBehavior
        }


# region Logic
class CentralCommanderLogic:
    def __init__(self, ui_cache: UICache, thread_manager: ThreadManager, network: NetworkManager):
        self.cache = ui_cache
        self.cache.network_manager = network
        self.thread_manager = thread_manager
        self.active_behavior: Behavior = None

    def scrub_ip_octet(self, index, value):
        try:
            # Only allow integers; scrub everything else to 0
            self.cache.ip_entry[index] = int(value)
        except (ValueError, TypeError):
            self.cache.ip_entry[index] = 0

    def scrub_port(self, value):
        try:
            self.cache.port = int(value)
        except (ValueError, TypeError):
            self.cache.port = 0

    def handle_connect(self):
        # Perform final validation here before attempting network connection
        print(f"Connecting to {self.get_full_ip()}...")
        self.cache.is_connected = True

        def client_connection_thread_func():
            self.cache.network_manager.maintain_client(self.get_full_ip(), self.cache.port, None)
        self.thread_manager.start_thread(self.cache.NETWORK_THREAD_NAME, client_connection_thread_func)

    def handle_host(self):
        self.cache.is_host = True

        def host_thread_func():
            self.cache.network_manager.maintain_host(self.cache.port, None)
        self.thread_manager.start_thread(self.cache.NETWORK_THREAD_NAME, host_thread_func)

    def handle_disconnect(self):
        self.cache.is_host = False
        self.cache.is_connected = False
        self.cache.confirm = False

    def get_full_ip(self):
        return ".".join(map(str, self.cache.ip_entry))

    def start_behavior(self):
        """Handles the logic of initializing and starting a behavior thread."""
        behavior_class = self.cache.behavior_map.get(self.cache.behavior_select_index)
        if not behavior_class:
            return None

        # Instantiate and configure
        b = behavior_class(self.thread_manager, self.cache.network_manager)
        # b.client_list = client_list

        if b.run:
            # The Logic class tells the Thread Manager to start the loop
            self.thread_manager.thread_manager.add_thread(self.cache.BEHAVIOR_THREAD_NAME, b.run)
            self.active_behavior = b

    def stop_behavior(self):
        """Logic for cleaning up the environment."""
        self.thread_manager.thread_manager.stop_thread(self.cache.BEHAVIOR_THREAD_NAME)
        self.active_behavior = None
# endregion


class CentralCommanderUI:
    def __init__(self, ui_cache: UICache, logic: CentralCommanderLogic):
        self.cache = ui_cache
        self.logic = logic

    def get_ip(self):
        return f"{self.cache.ip_entry[0]}." \
               f"{self.cache.ip_entry[1]}." \
               f"{self.cache.ip_entry[2]}." \
               f"{self.cache.ip_entry[3]}"

    def draw_main_window(self):
        PyImGui.dummy(self.cache.WIDTH, 0)
        PyImGui.set_cursor_pos_y(PyImGui.get_cursor_pos_y() - 10)

        if not (self.cache.is_host or self.cache.is_connected):
            self._draw_connection_inputs()
        else:
            self._draw_active_state_controls()

    def _draw_connection_inputs(self):
        PyImGui.text("IP:")
        PyImGui.same_line(0.0, -1.0)

        # Draw IP Octets
        for i in range(4):
            PyImGui.push_item_width(30)
            # 1. UI READ: Get value from cache
            val = str(self.cache.ip_entry[i])
            # 2. UI INTERACTION: PyImGui returns new value if changed
            new_val = PyImGui.input_text(f"##ip{i}", val)
            # 3. LOGIC TRIGGER: Tell logic to scrub and save it
            if new_val != val:
                self.logic.scrub_ip_octet(i, new_val)

            PyImGui.pop_item_width()

            # Layout spacers
            PyImGui.same_line(0.0, 0.0)
            PyImGui.text("." if i < 3 else ":")
            PyImGui.same_line(0.0, 0.0)

        # Draw Port
        PyImGui.push_item_width(60)
        port_val = str(self.cache.port)
        new_port = PyImGui.input_text("##port", port_val)
        if new_port != port_val:
            self.logic.scrub_port(new_port)
        PyImGui.pop_item_width()

        # Action Buttons
        PyImGui.same_line(0.0, 10.0)
        if PyImGui.button("Connect"):
            self.logic.handle_connect()

        PyImGui.same_line(0.0, 10.0)
        if PyImGui.button("Host"):
            self.logic.handle_host()

    def _draw_active_state_controls(self):
        btn_lbl = "Stop Hosting" if self.cache.is_host else "Disconnect"

        if not self.cache.confirm:
            if PyImGui.button(btn_lbl, self.cache.WIDTH):
                self.cache.confirm = True
        else:
            # The UI simply triggers the logic function
            if PyImGui.button("Confirm?", self.cache.WIDTH / 2):
                self.logic.handle_disconnect()

            PyImGui.same_line(0.0, 0.0)
            if PyImGui.button("No!", self.cache.WIDTH / 2):
                self.cache.confirm = False
        if self.cache.is_host:
            self.draw_host_window()
        else:
            self.draw_connected_window()

    def draw_host_window(self):
        """The Host-specific UI section."""
        if PyImGui.begin_child("Commander Host", (self.cache.WIDTH, 600), True):
            # Display status (Read-only)
            PyImGui.text(f"Timer: {self.cache.ui_timer}")
            PyImGui.separator()

            # Behavior Selection
            behavior_names = [cls.name() for cls in self.cache.behavior_map.values()]
            self.cache.behavior_select_index = PyImGui.combo(
                "Behavior", self.cache.behavior_select_index, behavior_names
            )

            # If there is an active behavior, let the behavior draw its own specific settings
            if self.logic.active_behavior:
                PyImGui.text("Active Settings:")
                self.logic.active_behavior.draw()
                PyImGui.separator()

            # Action Buttons
            if self.logic.active_behavior is None:
                if PyImGui.button("Start Behavior", self.cache.WIDTH):
                    # Logic Trigger: Request to start a behavior
                    self.logic.start_behavior()
            else:
                if PyImGui.button("Stop Behavior", self.cache.WIDTH):
                    # Logic Trigger: Request to stop threads
                    self.logic.stop_behavior()
        for c in self.cache.network_manager.client_list.keys():
            PyImGui.text(f"Client: {c}")

        PyImGui.end_child()

    def draw_connected_window(self):
        if PyImGui.begin_child("Connected_child", (self.cache.WIDTH, 200), True):
            PyImGui.text("Connecting...")
        PyImGui.end_child()
# endregion


class CentralCommander:
    def __init__(self):
        self.cache_multithread_booster = MultithreadBoosterCache()
        self.thread_manager = ThreadManager()
        self.local_game_client: GameClient = GameClient()
        self.network_manager = NetworkManager(self.thread_manager, self.local_game_client)
        self.cache_ui = UICache()
        self.logic = CentralCommanderLogic(self.cache_ui, self.thread_manager, self.network_manager)
        self.ui_main = CentralCommanderUI(self.cache_ui, self.logic)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.thread_manager.thread_manager.stop_all_threads()
        print("Central Command Exit called")

    def draw_in_window(self):
        self.ui_main.draw_main_window()

    def update(self):
        self.update_thread_manager()
        self.local_game_client.update_from_dict(Jsonizer.player_data())

    def update_thread_manager(self):
        if self.thread_manager.is_threads_running:
            self.thread_manager.thread_manager.update_all_keepalives()
            # stop all threads when not trying to connect or host, basically go dormant
            if not (self.cache_ui.is_host or self.cache_ui.is_connected):
                self.thread_manager.is_threads_running = False
                self.logic.stop_behavior()
                self.thread_manager.stop_sequential_environment()

    def main_thread_performance_delay(self):
        if not GW.Routines.Checks.Map.MapValid():
            self.cache_multithread_booster.load_delay = time.time()
            return
        current = time.time()
        if current - self.cache_multithread_booster.load_delay < 5:  # since this widget exists only to increase single thread performance of other scripts the delay shouldn't be an issue
            return
        if current - self.cache_multithread_booster.fps_poll > 1:
            self.cache_multithread_booster.fps = GW.UIManager.GetFPSLimit()
            self.cache_multithread_booster.fps_poll = current
        if self.cache_multithread_booster.fps != 0:  # 0 is uncapped fps
            delta = current - self.cache_multithread_booster.timer
            if delta < 1 / self.cache_multithread_booster.fps:
                time.sleep((1 / self.cache_multithread_booster.fps - delta))
                self.cache_multithread_booster.timer = time.time()


# region Widget Code
"""
CHECKLIST:
 - Make a copy of this file and name your module, follow the file paths listed for configs and modify these functions.
 - Widgets//widget_manager//default_settings.py - need to add some basic configs
 - Py4GW.ini - need to add some basic configs
"""

script_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_directory, os.pardir))

first_run = True

BASE_DIR = os.path.join(project_root, "Widgets/Config")
INI_WIDGET_WINDOW_PATH = os.path.join(BASE_DIR, "central_commander.ini")
os.makedirs(BASE_DIR, exist_ok=True)

# ——— Window Persistence Setup ———
ini_window = IniHandler(INI_WIDGET_WINDOW_PATH)
save_window_timer = Timer()
save_window_timer.Start()

# String consts
MODULE_NAME = "Central Commander ZMQ"  # Change this Module name
COLLAPSED = "collapsed"
X_POS = "x"
Y_POS = "y"

# load last‐saved window state (fallback to 100,100 / un-collapsed)
window_x = ini_window.read_int(MODULE_NAME, X_POS, 100)
window_y = ini_window.read_int(MODULE_NAME, Y_POS, 100)
window_collapsed = ini_window.read_bool(MODULE_NAME, COLLAPSED, False)
central_commander = CentralCommander()


def draw_widget():
    global window_x, window_y, window_collapsed, first_run, central_commander

    if first_run:
        PyImGui.set_next_window_pos(window_x, window_y)
        PyImGui.set_next_window_collapsed(window_collapsed, 0)
        first_run = False

    is_window_opened = PyImGui.begin(MODULE_NAME, PyImGui.WindowFlags.AlwaysAutoResize)
    new_collapsed = PyImGui.is_window_collapsed()
    end_pos = PyImGui.get_window_pos()

    if is_window_opened:
        central_commander.draw_in_window()

    PyImGui.end()
    if save_window_timer.HasElapsed(1000):
        # Position changed?
        if (end_pos[0], end_pos[1]) != (window_x, window_y):
            window_x, window_y = int(end_pos[0]), int(end_pos[1])
            ini_window.write_key(MODULE_NAME, X_POS, str(window_x))
            ini_window.write_key(MODULE_NAME, Y_POS, str(window_y))
        # Collapsed state changed?
        if new_collapsed != window_collapsed:
            window_collapsed = new_collapsed
            ini_window.write_key(MODULE_NAME, COLLAPSED, str(window_collapsed))
        save_window_timer.Reset()


def configure():
    pass


def main():
    global central_commander
    try:
        if not Routines.Checks.Map.MapValid():
            return

        if Routines.Checks.Map.IsMapReady() and Routines.Checks.Party.IsPartyLoaded():
            draw_widget()
            central_commander.update()

    except ImportError as e:
        Py4GW.Console.Log(MODULE_NAME, f"ImportError encountered: {str(e)}", Py4GW.Console.MessageType.Error)
        Py4GW.Console.Log(MODULE_NAME, f"Stack trace: {traceback.format_exc()}", Py4GW.Console.MessageType.Error)
    except ValueError as e:
        Py4GW.Console.Log(MODULE_NAME, f"ValueError encountered: {str(e)}", Py4GW.Console.MessageType.Error)
        Py4GW.Console.Log(MODULE_NAME, f"Stack trace: {traceback.format_exc()}", Py4GW.Console.MessageType.Error)
    except TypeError as e:
        Py4GW.Console.Log(MODULE_NAME, f"TypeError encountered: {str(e)}", Py4GW.Console.MessageType.Error)
        Py4GW.Console.Log(MODULE_NAME, f"Stack trace: {traceback.format_exc()}", Py4GW.Console.MessageType.Error)
    except Exception as e:
        # Catch-all for any other unexpected exceptions
        Py4GW.Console.Log(MODULE_NAME, f"Unexpected error encountered: {str(e)}", Py4GW.Console.MessageType.Error)
        Py4GW.Console.Log(MODULE_NAME, f"Stack trace: {traceback.format_exc()}", Py4GW.Console.MessageType.Error)
    finally:
        pass


if __name__ == "__main__":
    main()
# endregion
