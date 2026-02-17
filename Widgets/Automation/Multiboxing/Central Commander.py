# region Imports
import os
import random
import traceback
from collections import deque

import Py4GW  # type: ignore
import PyEffects
from Py4GWCoreLib import IniHandler
from Py4GWCoreLib import PyImGui
from Py4GWCoreLib import Routines
from Py4GWCoreLib import Timer
import Py4GWCoreLib as GW
import time
import socket
from collections.abc import Mapping
from typing import List, Dict, Protocol, ParamSpec, TypeVar, Any, Callable
import threading
import json, struct
from enum import Enum
import math
# endregion


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


class RPCRegistry:
    def __init__(self) -> None:
        self._methods: Dict[str, Callable[..., object]] = {}

    def register(
            self,
            name: str,
            method: RPCMethod[P, R],
    ) -> None:
        if name in self._methods:
            raise ValueError(f"RPC method '{name}' already registered")

        self._methods[name] = method

    def call(
            self,
            name: str,
            args: tuple[object, ...],
            kwargs: dict[str, object],
    ) -> object:
        try:
            method = self._methods[name]
        except KeyError:
            raise KeyError(f"RPC method '{name}' not found")
        return method(*args, **kwargs)


def RelativeMove(x, y):
    _x, _y = GW.Player.GetXY()
    GW.Player.Move(_x + x, _y + y)


class RPC(Enum):
    MOVE = "move"
    RELATIVE_MOVE = "relmove"
    GETSKILLBAR = "getskillbar"
    GETEFFECTS = "geteffects"
    GET_PLAYER_ID = "getplayerid"
    GET_PLAYER_DATA = "get_player_data"
    USE_SKILL = "user_skill"
    DROP_BOND = "drop_bond"


registry = RPCRegistry()
registry.register(RPC.MOVE.value, GW.Player.Move)
registry.register(RPC.GETSKILLBAR.value, GW.GLOBAL_CACHE.SkillBar.GetSkillbar)
registry.register(RPC.GETEFFECTS.value, Jsonizer.get_effects)
registry.register(RPC.GET_PLAYER_ID.value, GW.Player.GetAgentID)
registry.register(RPC.GET_PLAYER_DATA.value, Jsonizer.player_data)
registry.register(RPC.USE_SKILL.value, GW.GLOBAL_CACHE.SkillBar.UseSkill)
registry.register(RPC.DROP_BOND.value, GW.GLOBAL_CACHE.Effects.DropBuff)
registry.register(RPC.RELATIVE_MOVE.value, RelativeMove)


def rpc_method(name: str):
    def decorator(func: RPCMethod[P, R]) -> RPCMethod[P, R]:
        registry.register(name, func)
        return func

    return decorator


# endregion


class TCPUtils:
    @staticmethod
    def json_fallback(obj: Any) -> str:
        return str(obj)

    @staticmethod
    def send_message(sock, message: dict):
        payload = json.dumps(message, default=TCPUtils.json_fallback)
        data = payload.encode("utf-8")
        length = struct.pack("!I", len(data))
        sock.sendall(length + data)

    @staticmethod
    def recv_exact(sock, n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Connection closed")
            buf += chunk
        return buf

    @staticmethod
    def recv_message(sock) -> dict:
        length_bytes = TCPUtils.recv_exact(sock, 4)
        length = struct.unpack("!I", length_bytes)[0]
        data = TCPUtils.recv_exact(sock, length)
        return json.loads(data.decode("utf-8"))


class MultithreadBoosterCache:
    def __init__(self):
        self.timer = time.time()
        self.fps_poll = time.time()
        self.load_delay = time.time()
        self.fps = 60


class CacheThreadGlobals:
    def __init__(self):
        self.thread_manager = GW.MultiThreading(2.0, log_actions=True)
        self.is_threads_running = False


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


class Client:
    def __init__(self, socket_: socket, cache_thread_globals: CacheThreadGlobals, PACKET_SIZE):
        self.socket: socket = socket_
        self.cache_thread_globals = cache_thread_globals
        self.PACKET_SIZE = PACKET_SIZE
        self.command_queue: deque[Dict] = deque()
        self.return_queue: deque = deque()
        self.containing_list: List[Client] = None

    def send_command(self, rpc: RPC, args: List):
        self.command_queue.append({"method": rpc.value, "args": args})

    def manage_client_connection(self, sock: socket):
        print("Managing a client")
        try:
            with sock:
                TCPUtils.send_message(sock, {"method": RPC.GET_PLAYER_DATA.value, "args": []})
                while self.cache_thread_globals.is_threads_running:
                    data = TCPUtils.recv_message(sock)  # sock.recv(self.PACKET_SIZE)
                    if not data:
                        break
                    self.return_queue.append(data)
                    while len(self.command_queue) < 1:
                        time.sleep(0.1)
                    get_data = deque()
                    other_calls = deque()
                    for x in self.command_queue:
                        if isinstance(x, Mapping):
                            (get_data if x.get("method", 0) == RPC.GET_PLAYER_DATA.value else other_calls).append(x)
                    if len(other_calls) > 0:
                        TCPUtils.send_message(sock, other_calls.popleft())
                    elif len(get_data) > 0:
                        TCPUtils.send_message(sock, get_data.popleft())
                    self.command_queue = other_calls

        finally:
            print("Client disconnected")
            self.containing_list.remove(self)


class Behavior:
    def __init__(self, thread_globals: CacheThreadGlobals):
        self.cache_thread_globals: CacheThreadGlobals = thread_globals
        self.client_list: List[Client] = list()
        self.client_data: Dict[Client, Dict] = dict()

    def run(self):
        ...

    def draw(self):
        pass

    def start_up(self):
        self.cache_thread_globals.thread_manager.add_thread("UPDATE_CLIENT_DATA", self.update_client_info)
        for c in self.client_list:
            c.return_queue.clear()

    def get_effect(self, client: Client, skill_id):
        return self.client_data.get(client, dict()).get("effects", dict()).get(str(skill_id), dict()).get("time_remaining", 0)

    def get_recharge(self, client: Client, skill_id):
        return self.client_data.get(client, dict()).get("skilldata", dict()).get(str(skill_id), dict()).get("get_recharge", float('inf'))

    def get_id(self, client: Client):
        return self.client_data.get(client, dict()).get("id", False)

    def get_skill_slot_by_id(self, client: Client, skill_id):
        data = self.client_data.get(client, dict()).get("skilldata", dict())
        return data.get(str(skill_id), dict()).get("slot", False)

    def use_skill_by_id(self, client: Client, skill_id, target=0):
        slot = self.get_skill_slot_by_id(client, skill_id)
        print(f"using skill by id. ID: {skill_id}, slot {slot}")
        client.send_command(RPC.USE_SKILL, [slot, target])

    def update_client_info(self):
        while self.cache_thread_globals.is_threads_running:
            time.sleep(0.1)
            for c in self.client_list:
                if len(c.command_queue) < 1:
                    c.command_queue.append({"method": RPC.GET_PLAYER_DATA.value, "args": []})
            for c in self.client_list:
                if len(c.return_queue) > 0:
                    get_data = deque()
                    other_returns = deque()
                    for x in c.return_queue:
                        if isinstance(x, Mapping):
                            (get_data if x.get("method", 0) == RPC.GET_PLAYER_DATA.value else other_returns).append(x)
                    data = get_data.pop() #pop like a stack and get the most recent data
                    #TODO use other_returns to interact with non-data fetch returns
                    if isinstance(data, Mapping):
                        if data.get("method", 0) == RPC.GET_PLAYER_DATA.value:
                            # self.client_data[c].clear()
                            if self.client_data.get(c, 0) == 0:
                                self.client_data[c] = data["returned"]
                            else:
                                self.client_data[c].clear()
                                self.client_data[c].update(data["returned"])

    def cast_skill_wait_for_effect(self, client: Client, skill_id, my_name):
        try:
            skill_slot = self.get_skill_slot_by_id(client, skill_id)
            if not skill_slot: return
            client.send_command(RPC.USE_SKILL, [skill_slot, 0])
            # print(f"mon thread data: {data}")

            if self.get_id(client) == 0:
                return
            while (self.cache_thread_globals.is_threads_running and
                   self.get_effect(client, skill_id) == 0):
                if self.get_recharge(client, skill_id) == 0 and not GW.Agent.IsAttacking(self.get_id(client)) and not GW.Agent.IsCasting(self.get_id(client)):
                    client.send_command(RPC.USE_SKILL, [skill_slot, self.get_id(client)])
                time.sleep(0.1)
        finally:
            print(f"popping thread {my_name}")
            self.cache_thread_globals.thread_manager.threads.pop(my_name)

    def wait_for_thread(self, thread_name):
        while self.cache_thread_globals.is_threads_running and self.cache_thread_globals.thread_manager.threads.get(
                thread_name, 0) != 0:
            time.sleep(0.01)

    def move_to_wait(self, client: Client, pos: Vec2, my_name, tolerance=100):
        try:
            if not client:
                return
            p: Vec2 = Vec2.from_tuple(GW.Agent.GetXY(self.get_id(client)))
            counter = 0
            while self.cache_thread_globals.is_threads_running and (p - pos).magnitude() > tolerance:
                if counter % 15 == 0:
                    client.send_command(RPC.RELATIVE_MOVE, [20, 20])
                counter += 1
                x, y = pos.as_tuple()
                client.send_command(RPC.MOVE, [x + random.randrange(-20, 20), y + random.randrange(-20, 20)])
                time.sleep(0.1)
                p = Vec2.from_tuple(GW.Agent.GetXY(self.get_id(client)))
        finally:
            print(f"Popped movement thread {my_name}.")
            self.cache_thread_globals.thread_manager.threads.pop(my_name)


class TestBehavior(Behavior):
    def run(self):
        self.start_up()
        time.sleep(0.5)
        while self.cache_thread_globals.is_threads_running:
            mona: Client = next((x for x in self.client_list), None)
            if not mona:
                print("No mona")
                continue
            rit_x, rit_y = GW.Player.GetXY()
            mona_x, mona_y = GW.Agent.GetXY(self.get_id(mona))
            vec_from_rit: Vec2 = Vec2(mona_x - rit_x, mona_y - rit_y)
            vec_from_rit = vec_from_rit.normalized()
            mona_thread = "mona_thread"
            self.cache_thread_globals.thread_manager.add_thread(mona_thread, self.move_to_wait, mona, (
                        Vec2.from_tuple(GW.Player.GetXY()) + (vec_from_rit * 300)), mona_thread)
            self.wait_for_thread(mona_thread)
            self.cache_thread_globals.thread_manager.add_thread(mona_thread, self.move_to_wait, mona, (
                    Vec2.from_tuple(GW.Player.GetXY()) + (vec_from_rit * 600)), mona_thread)
            self.wait_for_thread(mona_thread)
            time.sleep(3)


class BehaviorPermaseedPrinter(Behavior):
    def __init__(self, thread_globals: CacheThreadGlobals):
        super().__init__(thread_globals)
        self.target_minion_count = 20
        self.state = 0

    def run_(self):
        weapon_of_quickening = 1268
        kareis_healing_circle = 1119
        heal_area = 280
        shielding_hands = 299
        shield_of_absorption = 1399
        fomf = 791
        ee = 2420
        balth_spirit = 242

        ua = 268
        seed = 2105
        blessed_aura = 256
        life_bond = 241
        animate_minions = 85
        dark_aura = 116
        oop = 134

        ritmo_busy = GW.Timer()
        ritmo_t = 0
        mona_busy = GW.Timer()
        mona_t = 0
        monb_busy = GW.Timer()
        monb_t = 0
        mona = 0
        monb = 0
        c: Client
        if ritmo_busy.HasElapsed(ritmo_t):
            ritmo_busy.Reset()
            ritmo_busy.Stop()
        if mona_busy.HasElapsed(mona_t):
            mona_busy.Reset()
            mona_busy.Stop()
        if monb_busy.HasElapsed(monb_t):
            monb_busy.Reset()
            monb_busy.Stop()
        #ritmo logic
        if not ritmo_busy.IsRunning():
            # Maintain balth spirit on Ritmo
            if GW.Effects.GetEffectTimeRemaining(GW.Player.GetAgentID(), balth_spirit) == 0:
                GW.SkillBar.UseSkill(GW.SkillBar.GetSlotBySkillID(balth_spirit), GW.Player.GetAgentID())
                ritmo_t = 3000
                ritmo_busy.Start()
            elif (GW.SkillBar.GetSkillData(GW.SkillBar.GetSlotBySkillID(weapon_of_quickening)).get_recharge == 0 and
                  GW.Effects.GetEffectTimeRemaining(GW.Player.GetAgentID(), weapon_of_quickening) < 3000):
                GW.SkillBar.UseSkill(GW.SkillBar.GetSlotBySkillID(weapon_of_quickening), GW.Player.GetAgentID())
                ritmo_t = 3000
                ritmo_busy.Start()
            elif GW.Agent.GetEnergy(GW.Player.GetAgentID()) > 0.9 and GW.SkillBar.GetSkillData(GW.SkillBar.GetSlotBySkillID(heal_area)).get_recharge == 0:
                GW.SkillBar.UseSkill(GW.SkillBar.GetSlotBySkillID(heal_area), GW.Player.GetAgentID())
                ritmo_t = 2000
                ritmo_busy.Start()
            elif (len(self.client_list) >= 1 and GW.Agent.GetEnergy(GW.Player.GetAgentID()) > 0.5 and
                  GW.SkillBar.GetSkillData(GW.SkillBar.GetSlotBySkillID(weapon_of_quickening)).get_recharge == 0 and
                  self.client_data.get(self.client_list[0], dict()).get("effects", dict()).get(str(weapon_of_quickening), dict()).get("time_remaining", 0) < 3000):
                GW.SkillBar.UseSkill(GW.SkillBar.GetSlotBySkillID(weapon_of_quickening), self.client_data.get(self.client_list[0], dict()).get("id", 0))
                ritmo_t = 3000
                ritmo_busy.Start()
            elif (len(self.client_list) >= 2 and GW.Agent.GetEnergy(GW.Player.GetAgentID()) > 0.5 and
                  GW.SkillBar.GetSkillData(GW.SkillBar.GetSlotBySkillID(weapon_of_quickening)).get_recharge == 0 and
                  self.client_data.get(self.client_list[1], dict()).get("effects", dict()).get(str(weapon_of_quickening), dict()).get("time_remaining", 0) < 3000):
                GW.SkillBar.UseSkill(GW.SkillBar.GetSlotBySkillID(weapon_of_quickening), self.client_data.get(self.client_list[1], dict()).get("id", 0))
                ritmo_t = 3000
                ritmo_busy.Start()
        else:
            pass # print(f"Ritmo busy {ritmo_busy.GetElapsedTime()}")
        #mona logic
        for c in self.client_list:
            pass
            # maintain dark aura on both
            # state track who to kill
            #   sac A
            #   b make minion and res A
            #   b use blessed aura and then seed ritmo
            #   switch a and b

    def draw(self):
        PyImGui.text("Minion Goal:")
        PyImGui.same_line(0.0, 0.0)
        self.target_minion_count = PyImGui.input_int("#binputminions", self.target_minion_count)

    def first_death_sequence(self, dark_aura, ua, agony) -> bool:
        print("Entering first death sequence")
        mona: Client = next((x for x in self.client_list if (self.client_data.get(x, dict()).get("max_hp", 100) != 1)), None)
        if mona is None: return len(self.client_list) > 1 #only returns true when at least 2 clients are connected and both have 1 max hp
        monb: Client = next((x for x in self.client_list if x is not mona), None)
        if monb is None: return False

        mona_thread = "mona_casting_thread"
        monb_thread = "monb_casting_thread"
        self.cache_thread_globals.thread_manager.add_thread(mona_thread, self.cast_skill_wait_for_effect, mona, dark_aura, mona_thread)
        self.cache_thread_globals.thread_manager.add_thread(monb_thread, self.cast_skill_wait_for_effect, monb, ua, monb_thread)
        self.wait_for_thread(mona_thread)
        self.wait_for_thread(monb_thread)
        time.sleep(0.75)
        self.cache_thread_globals.thread_manager.add_thread(mona_thread, self.sac_wait, mona, agony,
                                                            mona_thread)
        self.wait_for_thread(mona_thread)
        time.sleep(0.1)
        self.ua_res(monb, self.get_id(mona), ua)
        return False

    def ua_res(self, client: Client, res_target_id, ua):
        while self.cache_thread_globals.is_threads_running and not GW.Agent.IsAlive(res_target_id):
            buff_id = self.client_data.get(client, dict()).get("buffs", dict()).get(str(ua), dict()).get("buff_id", 0)
            if buff_id != 0:
                client.send_command(RPC.DROP_BOND, [buff_id])
            else:
                self.use_skill_by_id(client, ua, 0)
            time.sleep(0.2)

    def use_ua_minion_res_drop(self, client: Client, res_target_id, ua, minion):
        while self.cache_thread_globals.is_threads_running:
            buff_id = self.client_data.get(client, dict()).get("buffs", dict()).get(str(ua), dict()).get("buff_id", 0)
            if buff_id == 0:
                self.use_skill_by_id(client, ua, 0)
            else:
                break
            time.sleep(0.2)
        while self.cache_thread_globals.is_threads_running and GW.Agent.GetCastingSkillID(self.get_id(client)) != minion:
            self.use_skill_by_id(client, minion, 0)
            time.sleep(0.5)
        print("casting minion skill detected")
        while self.cache_thread_globals.is_threads_running and self.client_data.get(client, dict()).get("skilldata", dict()).get(str(minion), dict()).get("get_recharge", 0) == 0:
            time.sleep(0.1)
        print("cooldown on skill detected, resing")
        self.ua_res(client, res_target_id, ua)

    def sac_wait(self, client: Client, sac_skill, my_name):
        try:
            if not client:
                return
            counter = 1
            while self.cache_thread_globals.is_threads_running and GW.Agent.IsAlive(self.get_id(client)) and self.client_data[client].get("hp", 1) > 0:
                self.use_skill_by_id(client, sac_skill)
                time.sleep(0.3)
                if counter % 15 == 0:
                    client.send_command(RPC.RELATIVE_MOVE, [20, 20])
        finally:
            print(f"Popped sacc thread {my_name}")
            self.cache_thread_globals.thread_manager.threads.pop(my_name)
    
    def tank_minions_without_seed(self, woq, soa, sh, ha, khc, balth_spirit):
        try:
            while self.cache_thread_globals.is_threads_running:
                time.sleep(0.05)
                id = GW.Player.GetAgentID()
                if GW.GLOBAL_CACHE.Effects.GetEffectTimeRemaining(id, balth_spirit) < 2000:
                    GW.GLOBAL_CACHE.SkillBar.UseSkill(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(balth_spirit), id)
                    time.sleep(2.75)
                    continue
                if GW.GLOBAL_CACHE.Effects.GetEffectTimeRemaining(id, woq) < 4000:
                    GW.GLOBAL_CACHE.SkillBar.UseSkill(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(woq), id)
                    time.sleep(2.75)
                    continue
                if GW.GLOBAL_CACHE.Effects.GetEffectTimeRemaining(id, sh) < 3000:
                    if GW.GLOBAL_CACHE.SkillBar.GetSkillData(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(sh)).get_recharge == 0:
                        GW.GLOBAL_CACHE.SkillBar.UseSkill(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(sh), id)
                        time.sleep(1)
                        continue
                    elif GW.GLOBAL_CACHE.SkillBar.GetSkillData(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(soa)).get_recharge == 0:
                        GW.GLOBAL_CACHE.SkillBar.UseSkill(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(soa), id)
                        time.sleep(1.75)
                        continue
                if GW.Agent.GetEnergy(id) > 0.9:
                    if GW.GLOBAL_CACHE.SkillBar.GetSkillData(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(ha)).get_recharge == 0:
                        GW.GLOBAL_CACHE.SkillBar.UseSkill(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(ha), id)
                        time.sleep(1.75)
                        continue
                    elif GW.GLOBAL_CACHE.SkillBar.GetSkillData(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(khc)).get_recharge == 0:
                        GW.GLOBAL_CACHE.SkillBar.UseSkill(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(khc), id)
                        time.sleep(1.75)
                        continue
        finally:
            pass

    def minion_print_loop(self, dark_aura, agony, balth_spirit, minion, ua, weapon_of_quickening, soa, sh, ha, khc):
        mona: Client = next((x for x in self.client_list if (self.client_data.get(x, dict()).get("max_hp", 100) == 1)), None)
        monb: Client = next((x for x in self.client_list if x is not mona and (self.client_data.get(x, dict()).get("max_hp", 100) == 1)), None)
        if not (mona and monb):
            print("mona and monb not found")
            return False
        mona_thread = "mona_casting_thread"
        monb_thread = "monb_casting_thread"
        rit_thread = "rit_thread"
        rit_x, rit_y = GW.Player.GetXY()
        if len(GW.Routines.Agents.GetFilteredEnemyArray(rit_x, rit_y, 200)) >= self.target_minion_count:
            return
        mona_x, mona_y = GW.Agent.GetXY(self.get_id(mona))
        vec_from_rit: Vec2 = Vec2(mona_x - rit_x, mona_y - rit_y)
        vec_from_rit = vec_from_rit.normalized()
        #First time setup, sac close to rit to get minions aggroed on rit.
        self.cache_thread_globals.thread_manager.add_thread(rit_thread, self.tank_minions_without_seed, weapon_of_quickening, soa, sh, ha, khc, balth_spirit)
        self.cache_thread_globals.thread_manager.add_thread(mona_thread, self.move_to_wait, mona, (Vec2.from_tuple(GW.Player.GetXY()) + (vec_from_rit * 300)), mona_thread)
        self.cache_thread_globals.thread_manager.add_thread(monb_thread, self.move_to_wait, monb, (Vec2.from_tuple(GW.Player.GetXY()) + (vec_from_rit * 300)), monb_thread)
        self.wait_for_thread(monb_thread)
        self.wait_for_thread(mona_thread)
        self.cache_thread_globals.thread_manager.add_thread(monb_thread, self.cast_skill_wait_for_effect, monb, dark_aura, monb_thread)
        self.cache_thread_globals.thread_manager.add_thread(mona_thread, self.cast_skill_wait_for_effect, mona, ua, mona_thread)
        self.wait_for_thread(monb_thread)
        self.wait_for_thread(mona_thread)
        self.cache_thread_globals.thread_manager.add_thread(monb_thread, self.sac_wait, monb, agony,
                                                            monb_thread)
        self.wait_for_thread(mona)
        self.use_ua_minion_res_drop(mona, self.get_id(monb), ua, minion)
        print("Entering main printing loop")
        while self.cache_thread_globals.is_threads_running and len(GW.Routines.Agents.GetFilteredEnemyArray(rit_x, rit_y, 200)) < self.target_minion_count:
            monb.send_command(RPC.MOVE, (Vec2.from_tuple(GW.Player.GetXY()) + vec_from_rit * 1500).as_list())
            # self.cache_thread_globals.thread_manager.add_thread(
            #     mona_thread, self.cast_skill_wait_for_effect, mona, dark_aura, mona_thread)
            self.cache_thread_globals.thread_manager.add_thread(monb_thread, self.move_to_wait, monb, (
                        Vec2.from_tuple(GW.Player.GetXY()) + (vec_from_rit * 1500)), monb_thread)
            # self.wait_for_thread(mona_thread)
            self.wait_for_thread(monb_thread)
            self.cache_thread_globals.thread_manager.add_thread(mona_thread, self.sac_wait, mona, agony,
                                                                mona_thread)
            self.wait_for_thread(mona_thread)
            self.use_ua_minion_res_drop(monb, self.get_id(mona), ua, minion)
            self.cache_thread_globals.thread_manager.add_thread(monb_thread, self.move_to_wait, monb, (
                    Vec2.from_tuple(GW.Player.GetXY()) + (vec_from_rit * 600)), monb_thread)
            self.wait_for_thread(monb_thread)
            mona, monb = monb, mona
            time.sleep(0.2)
        self.cache_thread_globals.thread_manager.add_thread(mona_thread, self.sac_wait, mona, agony,
                                                            mona_thread)
        self.wait_for_thread(mona_thread)
        self.ua_res(monb, self.get_id(mona), ua)
        time.sleep(0.3)
        self.cache_thread_globals.thread_manager.add_thread(monb_thread, self.move_to_wait, monb, (
                Vec2.from_tuple(GW.Player.GetXY()) + (vec_from_rit * 1000)), monb_thread)
        self.cache_thread_globals.thread_manager.add_thread(mona_thread, self.move_to_wait, mona, (
                Vec2.from_tuple(GW.Player.GetXY()) + (vec_from_rit * 1000)), mona_thread)
        self.wait_for_thread(mona_thread)
        self.wait_for_thread(monb_thread)
        self.cache_thread_globals.thread_manager.stop_thread(rit_thread)

    def rit_tank_with_seed(self, woq, ha, khc, sh, mona: Client, monb: Client):
        try:
            last_heal = time.time()
            last_mona_woq = 0
            last_monb_woq = 0
            step = 1
            while self.cache_thread_globals.is_threads_running:
                time.sleep(0.05)
                id = GW.Player.GetAgentID()
                mona_woq = self.get_effect(mona, woq)
                monb_woq = self.get_effect(monb, woq)
                if GW.Agent.GetEnergy(id) > 0.9:
                    if GW.GLOBAL_CACHE.SkillBar.GetSkillData(
                            GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(ha)).get_recharge == 0:
                        GW.GLOBAL_CACHE.SkillBar.UseSkill(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(ha), id)
                    elif GW.GLOBAL_CACHE.SkillBar.GetSkillData(
                            GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(khc)).get_recharge == 0:
                        GW.GLOBAL_CACHE.SkillBar.UseSkill(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(khc), id)
                    time.sleep(2)
                    if GW.GLOBAL_CACHE.SkillBar.GetSkillData(
                            GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(woq)).get_recharge != 0:
                        continue
                    match step:
                        case 1:
                            GW.GLOBAL_CACHE.SkillBar.UseSkill(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(woq), id)
                            time.sleep(1.75)
                        case 3:
                            GW.GLOBAL_CACHE.SkillBar.UseSkill(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(woq),
                                                              self.get_id(monb))
                            time.sleep(1.75)
                        case 5:
                            GW.GLOBAL_CACHE.SkillBar.UseSkill(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(woq),
                                                              self.get_id(mona))
                            time.sleep(1.75)
                    step = (step + 1) % 6
        finally:
            pass

    def rit_tank_with_seed_(self, woq, ha, khc, sh, mona: Client, monb: Client):
        try:
            last_heal = time.time()
            last_mona_woq = 0
            last_monb_woq = 0
            while self.cache_thread_globals.is_threads_running:
                time.sleep(0.05)
                id = GW.Player.GetAgentID()
                mona_woq = self.get_effect(mona, woq)
                monb_woq = self.get_effect(monb, woq)
                if GW.Agent.GetEnergy(id) > 0.9 or time.time() - last_heal > 7:
                    if GW.GLOBAL_CACHE.SkillBar.GetSkillData(
                            GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(ha)).get_recharge == 0:
                        GW.GLOBAL_CACHE.SkillBar.UseSkill(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(ha), id)
                        time.sleep(1.75)
                        last_heal = time.time()
                        continue
                    elif GW.GLOBAL_CACHE.SkillBar.GetSkillData(
                            GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(khc)).get_recharge == 0:
                        GW.GLOBAL_CACHE.SkillBar.UseSkill(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(khc), id)
                        time.sleep(1.75)
                        last_heal = time.time()
                        continue
                if mona_woq < 8000:
                    GW.GLOBAL_CACHE.SkillBar.UseSkill(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(woq), self.get_id(mona))
                    time.sleep(2.75)
                    continue
                if monb_woq < 8000:
                    GW.GLOBAL_CACHE.SkillBar.UseSkill(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(woq), self.get_id(monb))
                    time.sleep(2.75)
                    continue
                if mona_woq == last_mona_woq:
                    print(f"woq on mona {self.get_effect(mona, woq)} full data: {self.client_data[mona]}")
                else:
                    last_mona_woq = mona_woq
                    last_monb_woq = monb_woq
                if GW.GLOBAL_CACHE.Effects.GetEffectTimeRemaining(id, woq) < 5000:
                    GW.GLOBAL_CACHE.SkillBar.UseSkill(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(woq), id)
                    time.sleep(2.75)
                    continue
        finally:
            pass

    def permaseed(self, weapon_of_quickening, seed, heal_area, khc, blessed_aura, sh):
        mona: Client = next((x for x in self.client_list if (self.client_data.get(x, dict()).get("max_hp", 100) == 1)),
                            None)
        monb: Client = next((x for x in self.client_list if
                             x is not mona and (self.client_data.get(x, dict()).get("max_hp", 100) == 1)), None)
        if not (mona and monb):
            print("mona and monb not found")
            return False
        mona_thread = "mona_casting_thread"
        monb_thread = "monb_casting_thread"
        rit_thread = "rit_thread"
        self.cache_thread_globals.thread_manager.add_thread(rit_thread, self.rit_tank_with_seed, weapon_of_quickening, heal_area, khc, sh, mona, monb)
        self.cache_thread_globals.thread_manager.add_thread(monb_thread, self.cast_skill_wait_for_effect, monb,
                                                            blessed_aura, monb_thread)
        self.cache_thread_globals.thread_manager.add_thread(mona_thread, self.cast_skill_wait_for_effect, mona, blessed_aura,
                                                            mona_thread)
        self.wait_for_thread(monb_thread)
        self.wait_for_thread(mona_thread)
        while self.cache_thread_globals.is_threads_running:
            mona_quickening = self.get_effect(mona, weapon_of_quickening)
            if GW.GLOBAL_CACHE.Effects.GetEffectTimeRemaining(GW.Player.GetAgentID(), seed) < 1000:
                if mona_quickening > 500:
                    self.use_skill_by_id(mona, seed, GW.Player.GetAgentID())
                    # mona.send_command(RPC.USE_SKILL, [seed, GW.Player.GetAgentID()])
                else:
                    GW.GLOBAL_CACHE.SkillBar.UseSkill(GW.GLOBAL_CACHE.SkillBar.GetSlotBySkillID(sh), GW.Player.GetAgentID())
            time.sleep(0.25)
            mona, monb = monb, mona

    def run(self):
        weapon_of_quickening = 1268
        kareis_healing_circle = 1119
        heal_area = 280
        shielding_hands = 299
        shield_of_absorption = 1399
        fomf = 791
        ee = 2420
        balth_spirit = 242

        ua = 268
        seed = 2105
        blessed_aura = 256
        life_bond = 241
        animate_minions = 85
        dark_aura = 116
        agony = 145
        self.start_up()
        time.sleep(0.5) #give connections time to clear initial interactions and get the data stream going
        client: Client
        all_clients_1_max_hp: List = [False]
        mona: Client
        monb: Client
        #slow kill until 1 max hp
        while self.cache_thread_globals.is_threads_running and not self.first_death_sequence(dark_aura, ua, agony):
            time.sleep(0.1)
            print("First Death sequence")
        #self.state = 1
        #print minions
        time.sleep(0.5)
        print("Entering minon print")
        self.minion_print_loop(dark_aura, agony, balth_spirit, animate_minions, ua, weapon_of_quickening, shield_of_absorption, shielding_hands, heal_area, kareis_healing_circle)
        #self.state = 2
        print("Finished printing minions")
        self.permaseed(weapon_of_quickening, seed, heal_area, kareis_healing_circle, blessed_aura, shielding_hands)



class CentralCommander:
    def __init__(self):
        self.cache_multithread_booster = MultithreadBoosterCache()
        self.cache_ui = UICache()
        self.cache_thread_globals = CacheThreadGlobals()
        self.is_host_thread_running = False
        self.is_client_thread_running = False
        self.client_list: List[Client] = list()
        self.MAIN_THREAD_NAME = "MaintainConnections"
        self.BEHAVIOR_THREAD_NAME = "Behavior_thread"
        self.PACKET_SIZE = 4
        self.socket: socket.socket = None
        self.behavior: Behavior = None

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cache_thread_globals.thread_manager.stop_all_threads()
        print("Central Command Exit called")

    def draw_in_window(self):
        PyImGui.dummy(self.cache_ui.WIDTH, 0)
        PyImGui.set_cursor_pos_y(PyImGui.get_cursor_pos_y() - 10)
        if not (self.cache_ui.is_host or self.cache_ui.is_connected):
            PyImGui.text("IP:")
            PyImGui.same_line(0.0, -1.0)
            for i in range(0, 3):
                PyImGui.push_item_width(30)
                self.cache_ui.ip_entry[i] = PyImGui.input_text(f"##ip{i}", f"{self.cache_ui.ip_entry[i]}")
                try:
                    self.cache_ui.ip_entry[i] = int(self.cache_ui.ip_entry[i])
                except:
                    self.cache_ui.ip_entry[i] = 0
                PyImGui.pop_item_width()
                # PyImGui.set_cursor_pos(pos[0] + i * 40 + 30, pos[1])
                PyImGui.same_line(0.0, 0.0)
                PyImGui.text(".")
                PyImGui.same_line(0.0, 0.0)
            PyImGui.push_item_width(30)
            self.cache_ui.ip_entry[3] = PyImGui.input_text(f"##ip{3}", f"{self.cache_ui.ip_entry[3]}")
            try:
                self.cache_ui.ip_entry[3] = int(self.cache_ui.ip_entry[3])
            except:
                self.cache_ui.ip_entry[3] = 0
            PyImGui.pop_item_width()
            PyImGui.same_line(0.0, 0.0)
            PyImGui.text(":")
            PyImGui.same_line(0.0, 0.0)
            PyImGui.push_item_width(60)
            self.cache_ui.port = PyImGui.input_text(f"##port", f"{self.cache_ui.port}")
            try:
                self.cache_ui.port = int(self.cache_ui.port)
            except:
                self.cache_ui.port = 0
            PyImGui.pop_item_width()
            if PyImGui.button("Connect"):
                self.cache_ui.is_connected = True
            PyImGui.same_line(0.0, 10.0)
            if PyImGui.button("Host"):
                self.cache_ui.is_host = True
        else:
            btn_lbl = "huh?"
            if self.cache_ui.is_host:
                btn_lbl = "Stop Hosting"
            elif self.cache_ui.is_connected:
                btn_lbl = "Disconnect"
            if not self.cache_ui.confirm and PyImGui.button(btn_lbl, self.cache_ui.WIDTH):
                self.cache_ui.confirm = True
            if self.cache_ui.confirm:
                if PyImGui.button("Confirm?", self.cache_ui.WIDTH / 2):
                    if self.cache_ui.is_host:
                        self.cache_ui.is_host = False
                    elif self.cache_ui.is_connected:
                        self.cache_ui.is_connected = False
                    self.cache_ui.confirm = False
                PyImGui.same_line(0.0, 0.0)
                if PyImGui.button("No!", self.cache_ui.WIDTH / 2):
                    self.cache_ui.confirm = False
            if self.cache_ui.is_host:
                self.draw_host_window()
            elif self.cache_ui.is_connected:
                self.draw_connected_window()

    def delay(self):
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

    def run_socket_accept(self, socket: socket.socket, receiver):
        try:
            receiver[0], receiver[1] = socket.accept()
            # socket.accept blocks the normal method of terminating threads with the watchdog
            # It is put in a normal thread, and the host thread that is watched by the watchdog closes it correctly in the finally block
        except OSError:
            pass
        finally:
            print("Exited inner thread")

    def maintain_host(self):
        s = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            host = ''  # Listen on all local interfaces
            s.bind((host, self.cache_ui.port))
            s.listen(1)
            thread_receiver = [0, 0]
            t = threading.Thread(target=self.run_socket_accept, args=(s, thread_receiver))
            t.daemon = True
            while self.cache_thread_globals.is_threads_running:
                if not t.is_alive():
                    client = thread_receiver[0]
                    address = thread_receiver[1]
                    c = Client(client, self.cache_thread_globals, self.PACKET_SIZE)
                    self.client_list.append(c)
                    c.containing_list = self.client_list
                    self.cache_thread_globals.thread_manager.add_thread(f"client_connect_thread_{address}",
                                                                        execute_fn=c.manage_client_connection,
                                                                        sock=client)
                    self.cache_thread_globals.thread_manager.start_watchdog(f"client_connect_thread_{address}")
                    t = threading.Thread(target=self.run_socket_accept, args=(s, thread_receiver))
                    t.start()
            #s.shutdown(socket.SHUT_RDWR)
            s.close()
        finally:
            print("Host finally clause called")
            if s is not None:
                if isinstance(s, socket.socket):
                    print("Closing socket")
                    # s.shutdown(socket.)
                    #s.shutdown(socket.SHUT_RDWR)
                    s.close()
                    self.client_list.clear()
            self.cache_ui.is_host = False
            self.cache_thread_globals.is_threads_running = False
            self.is_host_thread_running = False

    def maintain_client(self):
        try:
            with socket.create_connection((f"{self.cache_ui.ip_entry[0]}.{self.cache_ui.ip_entry[1]}"
                                           f".{self.cache_ui.ip_entry[2]}.{self.cache_ui.ip_entry[3]}",
                                           self.cache_ui.port)) as tcp_socket:
                while self.cache_thread_globals.is_threads_running and tcp_socket:
                    try:
                        data = TCPUtils.recv_message(tcp_socket)  # tcp_socket.recv(self.PACKET_SIZE)
                        # print(f"Data was {data}")
                        method = data["method"]
                        data = {"returned": registry.call(data["method"], data["args"],
                                                          data["kwargs"] if data.keys().__contains__("kwargs") else {}),
                                "method": method}
                        # print(f"Data return {data}")
                        if data:  # echo data back to server
                            TCPUtils.send_message(tcp_socket, data)

                    finally:
                        pass
        finally:
            print("maintain client finally clause was called")
            self.cache_ui.is_connected = False
            self.cache_thread_globals.is_threads_running = False
            self.is_client_thread_running = False

    def start_host_environment(self):
        self.cache_thread_globals.is_threads_running = True
        self.cache_thread_globals.thread_manager.stop_all_threads()
        # Add sequential threads
        self.cache_thread_globals.thread_manager.add_thread(self.MAIN_THREAD_NAME, self.maintain_host)
        # Watchdog thread is necessary to async close other running threads
        self.cache_thread_globals.thread_manager.start_watchdog(self.MAIN_THREAD_NAME)

    def start_client_environment(self):
        self.cache_thread_globals.is_threads_running = True
        self.cache_thread_globals.thread_manager.stop_all_threads()
        # Add sequential threads
        self.cache_thread_globals.thread_manager.add_thread(self.MAIN_THREAD_NAME, self.maintain_client)
        # Watchdog thread is necessary to async close other running threads
        self.cache_thread_globals.thread_manager.start_watchdog(self.MAIN_THREAD_NAME)

    def stop_sequential_environment(self):
        self.cache_thread_globals.thread_manager.stop_all_threads()
        self.cache_thread_globals.thread_manager.stop_thread(self.MAIN_THREAD_NAME)
        self.cache_thread_globals.is_threads_running = False
        self.behavior = None

    def update(self):
        if self.cache_thread_globals.is_threads_running:
            self.cache_thread_globals.thread_manager.update_all_keepalives()
            if not (self.cache_ui.is_host or self.cache_ui.is_connected):
                self.cache_thread_globals.is_threads_running = False
                self.is_host_thread_running = False
                self.is_client_thread_running = False
                self.stop_sequential_environment()
        if self.cache_ui.is_host and not self.is_host_thread_running:
            self.is_host_thread_running = True
            self.start_host_environment()
            self.cache_thread_globals.is_threads_running = True
        if self.cache_ui.is_connected and not self.is_client_thread_running:
            self.cache_thread_globals.is_threads_running = True
            self.is_client_thread_running = True
            self.start_client_environment()

    def draw_host_window(self):
        if PyImGui.begin_child("Commander Host", (self.cache_ui.WIDTH, 600), True):
            PyImGui.text(f"Hey {self.cache_thread_globals.is_threads_running}")
            PyImGui.text(f"Timer {self.cache_ui.ui_timer}")
            self.cache_ui.behavior_select_index = PyImGui.combo("Behavior", self.cache_ui.behavior_select_index,
                                                                ["Test", "Permaseed Printer"])
            if self.behavior:
                self.behavior.draw()
            if PyImGui.button("Start Behavior"):
                b: Behavior = None
                match self.cache_ui.behavior_select_index:
                    case 0:
                        b = TestBehavior(self.cache_thread_globals)
                        b.client_list = self.client_list
                    case 1:
                        b = BehaviorPermaseedPrinter(self.cache_thread_globals)
                        b.client_list = self.client_list
                if b and b.run:
                    self.cache_thread_globals.thread_manager.add_thread(self.BEHAVIOR_THREAD_NAME, b.run)
                    self.behavior = b
            if PyImGui.button("Stop Threads"):
                self.stop_sequential_environment()
        PyImGui.end_child()

    def draw_connected_window(self):
        if PyImGui.begin_child("Connected_child", (self.cache_ui.WIDTH, 200), True):
            PyImGui.text("Connecting...")
        PyImGui.end_child()


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
MODULE_NAME = "Central Commander"  # Change this Module name
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
