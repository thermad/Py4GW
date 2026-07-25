# region Imports
import os, sys, time, traceback

import Py4GW  # type: ignore
from Py4GWCoreLib import IniHandler, PyImGui, Routines, Timer
import Py4GWCoreLib as GW


# reloading imports so the file can be split up
def get_widget_lib_dir(widget_file: str, subfolder: str = "lib") -> str:
    """Resolve the import folder relative to the widget file itself."""
    widget_dir = os.path.dirname(os.path.abspath(widget_file))
    return os.path.join(widget_dir, subfolder)


def ensure_on_path(lib_dir: str):
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)


def purge_by_path(root_dir: str):
    """Drop every module whose __file__ lives under root_dir from sys.modules."""
    root_dir = os.path.abspath(root_dir)
    to_remove = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and os.path.abspath(f).startswith(root_dir + os.sep):
            to_remove.append(name)
    for name in to_remove:
        del sys.modules[name]


_WIDGET_DIR = os.path.dirname(os.path.abspath(__file__))   # client codebase root
_LIB_DIR = os.path.join(_WIDGET_DIR, "cc_lib")

# `import cc_lib.X` resolves cc_lib as a package, so the *parent* of cc_lib (this widget dir) must be
# on sys.path. Py4GW does not guarantee this, so add it explicitly.
ensure_on_path(_WIDGET_DIR)
purge_by_path(_LIB_DIR)          # drop every already-imported cc_lib.* module (client slice)

# --- CLIENT slice cc_lib modules. This is the THIN client: it observes the game, ships snapshots up
# to the standalone server, and executes the concrete orders the server sends back. The decision
# engine (behaviors / combat / world-merge) lives ONLY in the server codebase now -- none of it is
# imported here. Party-wide commands are just REQUESTS pushed up to the server (see update()). ---
from cc_lib.misc_helpers import (MultithreadBoosterCache, ThreadManager, map_ready,
                                 take_pending_map_traps)
from cc_lib.blackboard import GameClient
from cc_lib.networking import NetworkManager
from cc_lib import encode as cc_encode
from cc_lib import wire_delta as cc_wire
from cc_lib import cast_observer as cc_cast_observer

WORLD_PUSH_INTERVAL = 0.1   # seconds between world-snapshot ships to the server
# Flip on to log per-encode cost (ms) + agent count to the console. Default OFF -- pure diagnostics
# for the frame-draw optimization; leave off in normal play.
PROFILE_ENCODE = False
# endregion


# region UI
class UICache:
    def __init__(self):
        self.WIDTH = 400
        self.ip_entry = [127, 0, 0, 1]
        self.port = 54321
        self.is_connected = False
        self.confirm = False
        self.network_manager: NetworkManager = None
        self.NETWORK_THREAD_NAME = "NETWORK_THREAD"
        # A one-shot control request to piggyback on the next upstream PUSH (see update()). Set by a
        # party-command button; cleared once attached to an outbound snapshot.
        self.pending_request = None
        self.last_request_note = ""       # transient "sent X" label for operator feedback


# region Logic
class CentralCommanderLogic:
    def __init__(self, ui_cache: UICache, thread_manager: ThreadManager, network: NetworkManager):
        self.cache = ui_cache
        self.cache.network_manager = network
        self.thread_manager = thread_manager

    def scrub_ip_octet(self, index, value):
        try:
            self.cache.ip_entry[index] = int(value)
        except (ValueError, TypeError):
            self.cache.ip_entry[index] = 0

    def scrub_port(self, value):
        try:
            self.cache.port = int(value)
        except (ValueError, TypeError):
            self.cache.port = 0

    def handle_connect(self):
        print(f"Connecting to {self.get_full_ip()}...")
        self.cache.is_connected = True

        def client_connection_thread_func():
            self.cache.network_manager.maintain_client(self.get_full_ip(), self.cache.port, None)
        self.thread_manager.start_thread(self.cache.NETWORK_THREAD_NAME, client_connection_thread_func)

    def handle_disconnect(self):
        self.cache.is_connected = False
        self.cache.confirm = False

    def request(self, cmd: str, **extra) -> None:
        """Queue a control request for the server (form_party / resign_all / interact_all / ...). It
        rides the next player_data PUSH as ``cc_request``; the server dispatches it (server.py)."""
        self.cache.pending_request = {"cmd": cmd, **extra}
        self.cache.last_request_note = f"sent: {cmd}"

    def get_full_ip(self):
        return ".".join(map(str, self.cache.ip_entry))
# endregion


class CentralCommanderUI:
    def __init__(self, ui_cache: UICache, logic: CentralCommanderLogic):
        self.cache = ui_cache
        self.logic = logic

    def draw_main_window(self):
        PyImGui.dummy(self.cache.WIDTH, 0)
        PyImGui.set_cursor_pos_y(PyImGui.get_cursor_pos_y() - 10)
        if not self.cache.is_connected:
            self._draw_connection_inputs()
        else:
            self._draw_connected_controls()

    def _draw_connection_inputs(self):
        PyImGui.text("IP:")
        PyImGui.same_line(0.0, -1.0)
        for i in range(4):
            PyImGui.push_item_width(30)
            val = str(self.cache.ip_entry[i])
            new_val = PyImGui.input_text(f"##ip{i}", val)
            if new_val != val:
                self.logic.scrub_ip_octet(i, new_val)
            PyImGui.pop_item_width()
            PyImGui.same_line(0.0, 0.0)
            PyImGui.text("." if i < 3 else ":")
            PyImGui.same_line(0.0, 0.0)

        PyImGui.push_item_width(60)
        port_val = str(self.cache.port)
        new_port = PyImGui.input_text("##port", port_val)
        if new_port != port_val:
            self.logic.scrub_port(new_port)
        PyImGui.pop_item_width()

        PyImGui.same_line(0.0, 10.0)
        if PyImGui.button("Connect"):
            self.logic.handle_connect()

    def _draw_connected_controls(self):
        if not self.cache.confirm:
            if PyImGui.button("Disconnect", self.cache.WIDTH):
                self.cache.confirm = True
        else:
            if PyImGui.button("Confirm?", self.cache.WIDTH / 2):
                self.logic.handle_disconnect()
            PyImGui.same_line(0.0, 0.0)
            if PyImGui.button("No!", self.cache.WIDTH / 2):
                self.cache.confirm = False

        PyImGui.separator()
        PyImGui.text("Connected. Party commands (run on the server):")
        # Each button just asks the SERVER to perform the action -- the server orchestrates every
        # client (host makes all decisions). This account is the natural rally point for a party.
        if PyImGui.button("Call to Outpost + Join"):
            self.logic.request("form_party")
        PyImGui.same_line(0.0, 6.0)
        if PyImGui.button("Resign All"):
            self.logic.request("resign_all")
        PyImGui.same_line(0.0, 6.0)
        if PyImGui.button("Interact All"):
            try:
                target = int(GW.Player.GetTargetID() or 0)
            except Exception:
                target = 0
            self.logic.request("interact_all", target=target)
        if self.cache.last_request_note:
            PyImGui.text_disabled(self.cache.last_request_note)
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
        # World-feed pacing: rebuilding + shipping the observed-world snapshot is expensive, so a
        # connected client does it at ~10 Hz, not every frame.
        self._last_world_push = 0.0
        # Last encoded wire Snapshot (cc_world), re-sent each frame between rate-gated encodes.
        self._last_cc_world = None

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.thread_manager.thread_manager.stop_all_threads()
        print("Central Command Exit called")

    def draw_in_window(self):
        self.ui_main.draw_main_window()

    def update(self):
        # Runs only when the map is fully live (main() gate). Does the GW-touching work the network
        # thread must NOT do: encode the observed-world Snapshot (GW reads) for the network loop to
        # send, and execute any RPCs the network loop queued (their GW calls run here, main thread).
        if self.cache_ui.is_connected:
            # Fold combat events into per-agent cast state (source of remaining cast time, which GW
            # has no live getter for). Cheap; must run each frame so casts aren't missed.
            cc_cast_observer.pump()
            now = time.time()
            if now - self._last_world_push >= WORLD_PUSH_INTERVAL:
                self._last_world_push = now
                try:
                    self_id = int(GW.Player.GetAgentID() or 0)
                    if self_id:
                        # encode_snapshot gathers the observed set from the LIVING-type shared-memory
                        # buckets itself (struct-direct); no full GetAgentArray scan needed here.
                        if PROFILE_ENCODE:
                            _t0 = time.perf_counter()
                            _snap = cc_encode.encode_snapshot(self_id)
                            self._last_cc_world = cc_wire.to_wire(_snap)
                            _dt = (time.perf_counter() - _t0) * 1000.0
                            print(f"[cc] encode+wire {_dt:.2f}ms agents={len(_snap.world)}")
                        else:
                            self._last_cc_world = cc_wire.to_wire(cc_encode.encode_snapshot(self_id))
                except Exception as e:
                    print(f"world feed encode failed: {e}")
            # The upstream payload is now ONLY the wire Snapshot (cc_world) + a one-shot control
            # request. The player_data/Jsonizer path is gone -- the server derives everything (self
            # state, map, party) from the decoded Snapshot. cc_world is re-sent each frame (refreshed
            # by the rate-gated encode above) so the server's per-client sync stays warm.
            outbound = {}
            if self._last_cc_world is not None:
                outbound["cc_world"] = self._last_cc_world
            # Attach a queued control request (form party / resign all / ...) for the server, once.
            if self.cache_ui.pending_request is not None:
                outbound["cc_request"] = self.cache_ui.pending_request
                print(f"[client] attaching cc_request to push: {self.cache_ui.pending_request}")
                self.cache_ui.pending_request = None
            self.network_manager.pump_outbound_state(outbound)
            # One-shot: a host DUMP_MAP_TRAPS RPC (drained the previous frame) bundled the instance's
            # pathing trapezoids. Ship the (large) bundle via the dedicated one-shot queue -- NOT the
            # per-frame outbound_state, which the next frame overwrites before the ~60ms network loop
            # can send it (that race left the server re-requesting forever). The queue guarantees one
            # delivery per bundle.
            traps = take_pending_map_traps()
            if traps is not None:
                self.network_manager.queue_oneshot({"cc_map_traps": traps})
                print(f"[client] queued cc_map_traps for push: map={traps.get('map_id')} "
                      f"quads={len(traps.get('quads', []))}")
        self.network_manager.drain_inbound_rpcs()

    def update_thread_manager(self):
        if self.thread_manager.is_threads_running:
            self.thread_manager.thread_manager.update_all_keepalives()
            # Go dormant (stop network threads) when not connected.
            if not self.cache_ui.is_connected:
                self.thread_manager.is_threads_running = False
                self.thread_manager.stop_sequential_environment()

    def main_thread_performance_delay(self):
        if not GW.Routines.Checks.Map.MapValid():
            self.cache_multithread_booster.load_delay = time.time()
            return
        current = time.time()
        if current - self.cache_multithread_booster.load_delay < 5:
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
MODULE_NAME = "Central Commander"
COLLAPSED = "collapsed"
X_POS = "x"
Y_POS = "y"

# load last‐saved window state (fallback to 100,100 / un-collapsed)
window_x = ini_window.read_int(MODULE_NAME, X_POS, 100)
window_y = ini_window.read_int(MODULE_NAME, Y_POS, 100)
window_collapsed = ini_window.read_bool(MODULE_NAME, COLLAPSED, False)

# Bump the hot-reload epoch BEFORE building the new CentralCommander. The client network loop from a
# previous load captures the old epoch and exits once it sees it change, so a reload can't leave an
# orphaned network thread. (Threads started before this guard existed won't self-stop -- restart the
# GW client once to clear those.)
try:
    GW.MultiThreading._cc_epoch = getattr(GW.MultiThreading, "_cc_epoch", 0) + 1
except Exception:
    pass
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

    # begin() MUST be paired with end() EVERY frame, even if the body throws, or ImGui's window stack
    # unbalances and the next frame renders corrupted (flicker). try/finally makes it unconditional.
    try:
        if is_window_opened:
            central_commander.draw_in_window()
    finally:
        PyImGui.end()

    if save_window_timer.HasElapsed(1000):
        if (end_pos[0], end_pos[1]) != (window_x, window_y):
            window_x, window_y = int(end_pos[0]), int(end_pos[1])
            ini_window.write_key(MODULE_NAME, X_POS, str(window_x))
            ini_window.write_key(MODULE_NAME, Y_POS, str(window_y))
        if new_collapsed != window_collapsed:
            window_collapsed = new_collapsed
            ini_window.write_key(MODULE_NAME, COLLAPSED, str(window_collapsed))
        save_window_timer.Reset()


def configure():
    pass


def main():
    global central_commander
    try:
        # Publish the "is this instance live" flag for background threads, computed HERE on the main
        # thread where the GW frame cache is valid (map_ready() uses frame-cached checks that return
        # stale values off-thread). The client network loop reads thread_manager.map_live.
        central_commander.thread_manager.publish_map_live(map_ready())

        # Pump watchdog keepalives EVERY frame, before the map gate, or the network thread's keepalive
        # expires (2s) in the post-zone gap and the watchdog reaps it, dropping the connection.
        central_commander.update_thread_manager()

        # Draw the control window whenever map DATA is loaded (NOT gated on a healthy party) so a
        # briefly-unhealthy party can't make the whole window vanish permanently.
        if Routines.Checks.Map.IsMapReady():
            draw_widget()

        # GW-data refresh (player_data reads the local agent) keeps the full, strict gate.
        if map_ready():
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
        Py4GW.Console.Log(MODULE_NAME, f"Unexpected error encountered: {str(e)}", Py4GW.Console.MessageType.Error)
        Py4GW.Console.Log(MODULE_NAME, f"Stack trace: {traceback.format_exc()}", Py4GW.Console.MessageType.Error)
    finally:
        pass


if __name__ == "__main__":
    main()
# endregion
