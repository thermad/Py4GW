# region Imports
import os, sys, time, traceback, uuid

import Py4GW  # type: ignore
from Py4GWCoreLib import IniHandler, PyImGui, Routines, Timer
import Py4GWCoreLib as GW
from typing import Dict, Optional


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


_WIDGET_DIR = os.path.dirname(os.path.abspath(__file__))   # project root
_LIB_DIR = os.path.join(_WIDGET_DIR, "cc_lib")

# `import cc_lib.X` resolves cc_lib as a package, so the *parent* of cc_lib
# (the widget dir) must be on sys.path -- not cc_lib itself. Py4GW does not
# guarantee this, so add it explicitly.
ensure_on_path(_WIDGET_DIR)

purge_by_path(_LIB_DIR)          # drop every already-imported cc_lib.* module

# --- split-out cc_lib modules (purged & re-imported above every hot-reload). bt/rpc load
# transitively via behaviors/transport, so only the names used here are imported directly. ---
from cc_lib.jsonizers import Jsonizer
from cc_lib.misc_helpers import MultithreadBoosterCache, ThreadManager, map_ready
from cc_lib.blackboard import GameClient
from cc_lib.rpc import RPC
from cc_lib.transport import Client
from cc_lib.networking import NetworkManager
from cc_lib.behaviors import BehaviorSlot, BehaviorManager, BEHAVIOR_MAP
# endregion


# region UI
class UICache:
    def __init__(self):
        self.WIDTH = 400
        self.ip_entry = [127, 0, 0, 1]
        self.port = 54321
        self.is_host = False
        self.is_connected = False
        self.ui_timer = 0
        self.confirm = False
        self.header_child_height = 60
        self.add_behavior_index = 0          # combo selection for the "Add Behavior" control
        self.network_manager: NetworkManager = None
        self.NETWORK_THREAD_NAME = "NETWORK_THREAD"
        self.behavior_map = BEHAVIOR_MAP
        # --- virtual drag-and-drop state (native ImGui DnD isn't exposed by this binding) ---
        # While a row is dragged we stash the client id here; each drop target captures its
        # screen rect this frame; on release we hit-test the mouse against those rects.
        self.drag_client_id: Optional[uuid.UUID] = None
        self.drop_rects: Dict[object, tuple] = {}


# region Logic
class PartyFormationFlow:
    """Sequenced 'call everyone to my outpost and form a party' flow, ticked once per frame on the
    host. A single button press can't do this in one shot because travel takes time and must be
    confirmed before inviting -- so this runs as a small state machine:

        travel  -> send each off-map client to the host's instance (re-issued to stragglers)
        confirm -> wait until every called client's SYNCED map matches the host's (it arrived)
        invite  -> the host invites each present client by name
        accept  -> after a short gap (so the order holds), each client invites the host back
                   (Guild Wars merges two parties when they invite each other) -> done

    Reads the host's map + name on the main thread; sends orders over each client's transport
    (remote = ZMQ, same path the behaviors already use)."""
    TRAVEL_TIMEOUT = 45.0     # stop waiting for stragglers after this long, invite whoever made it
    REISSUE_TRAVEL = 4.0      # re-send travel to clients not yet arrived, this often
    ACCEPT_GAP = 0.75         # pause between 'host invited' and 'clients accept' so ordering holds

    def __init__(self, network: NetworkManager):
        self.network = network
        self.phase = "idle"            # idle | travel | invite | done
        self.status = ""
        self._t0 = 0.0
        self._last_travel = 0.0
        self._invited_at = 0.0
        self.host_map = None           # (map_id, region, district, language)
        self.host_name = ""

    @property
    def active(self) -> bool:
        return self.phase not in ("idle", "done")

    def start(self) -> None:
        try:
            self.host_map = (int(GW.Map.GetMapID()), int(GW.Map.GetRegion()[0]),
                             int(GW.Map.GetDistrict()), int(GW.Map.GetLanguage()[0]))
            self.host_name = GW.Player.GetName() or ""
        except Exception:
            self.phase = "done"
            self.status = "Failed to read host map/name"
            return
        self._t0 = time.time()
        self._last_travel = 0.0
        self._invited_at = 0.0
        self.phase = "travel"
        self.status = "Calling clients to outpost..."

    def cancel(self) -> None:
        self.phase = "idle"
        self.status = ""

    def _remote_clients(self):
        return [(cid, c) for cid, c in list(self.network.client_list.items())
                if cid != self.network.local_client_id]

    def _on_host_map(self, client) -> bool:
        gc = client.game_client
        if gc is None or self.host_map is None:
            return False
        return (gc.map_id, gc.map_region, gc.map_district, gc.map_language) == self.host_map

    def tick(self) -> None:
        if not self.active:
            return
        now = time.time()
        if self.phase == "travel":
            remotes = self._remote_clients()
            if not remotes:
                self.phase = "done"
                self.status = "No clients connected"
                return
            need = [c for _cid, c in remotes if not self._on_host_map(c)]
            if not need:
                self.phase = "invite"
                self._invited_at = 0.0
                self.status = "All clients arrived; inviting..."
                return
            if now - self._last_travel >= self.REISSUE_TRAVEL:
                self._last_travel = now
                mid, reg, dist, lang = self.host_map
                for c in need:
                    c.transport.send(RPC.CMD.TRAVEL_TO, mid, reg, dist, lang)
            self.status = f"Waiting for {len(need)} client(s) to arrive..."
            if now - self._t0 >= self.TRAVEL_TIMEOUT:
                self.phase = "invite"
                self._invited_at = 0.0
                self.status = "Travel timed out; inviting those present..."
        elif self.phase == "invite":
            present = [c for _cid, c in self._remote_clients() if self._on_host_map(c)]
            if not present:
                self.phase = "done"
                self.status = "No clients on the outpost to invite"
                return
            if self._invited_at == 0.0:
                # Host invites each present client first...
                for c in present:
                    name = c.game_client.name if c.game_client else ""
                    if name:
                        try:
                            GW.Party.Players.InvitePlayer(name)
                        except Exception:
                            pass
                self._invited_at = now
                self.status = "Host invited; waiting for clients to accept..."
            elif now - self._invited_at >= self.ACCEPT_GAP:
                # ...then each client accepts by inviting the host back (mutual invite = join).
                for c in present:
                    if self.host_name:
                        c.transport.send(RPC.CMD.INVITE_PLAYER, self.host_name)
                self.phase = "done"
                self.status = "Party formation complete"


class CentralCommanderLogic:
    def __init__(self, ui_cache: UICache, thread_manager: ThreadManager, network: NetworkManager):
        self.cache = ui_cache
        self.cache.network_manager = network
        self.thread_manager = thread_manager
        self.behaviors = BehaviorManager(network, thread_manager)
        # Multi-frame 'call everyone to outpost + form party' flow (ticked from update() while hosting).
        self.party_flow = PartyFormationFlow(network)

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
        self.behaviors.stop_all()
        self.cache.is_host = False
        self.cache.is_connected = False
        self.cache.confirm = False

    def get_full_ip(self):
        return ".".join(map(str, self.cache.ip_entry))
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

    # ----------------------------------------------------------- host UI helpers
    def _client_name(self, cid: uuid.UUID, client: 'Client') -> str:
        """The character name as reported by the client over the wire (see player_data). We do
        NOT call GW.Agent.GetNameByID here: a remote client's agent id is only meaningful in its
        OWN game instance, so resolving it against the host's world always returns "" -- which
        meant this label was re-querying an async API every single frame and never settling,
        flickering the whole row. The blackboard name is a plain string, so it renders stably."""
        name = client.game_client.name if client.game_client else ""
        return name if name else f"Pending name... ({str(cid)[:8]})"

    def _capture_rect(self, key: object) -> None:
        """Record the current child window's screen rect as a drop target for this frame."""
        px, py = PyImGui.get_window_pos()
        sw, sh = PyImGui.get_window_size()
        self.cache.drop_rects[key] = (px, py, sw, sh)

    def _draw_client_row(self, cid: uuid.UUID, client: 'Client') -> None:
        """A client row that can be picked up as a virtual-drag payload. The drag begins once
        the mouse moves while the row is held (is_item_active), and is resolved on release."""
        name = self._client_name(cid, client)
        label = f"{name} : {client.game_client.agent_id} : {cid.__str__()[:4]}"
        # Highlight the row while it is the active drag payload.
        PyImGui.selectable(label, self.cache.drag_client_id == cid,
                           PyImGui.SelectableFlags.NoFlag, (0, 0))
        # Begin a virtual drag once the row is held down AND the mouse moves. is_item_active()
        # refers to the selectable just drawn and stays true while the button is held even after
        # the cursor leaves the row -- exactly the lifetime a drag needs. (The selectable's click
        # RETURN value is the wrong signal: it only fires on a clean click-release, which never
        # coincides with is_mouse_dragging, so drags never started.)
        if (self.cache.drag_client_id is None
                and PyImGui.is_item_active() and PyImGui.is_mouse_dragging(0, -1)):
            self.cache.drag_client_id = cid

    def _draw_behavior_slot(self, slot: BehaviorSlot) -> None:
        mgr = self.logic.behaviors
        # Make sure the instance exists so its settings render even before Start (and persist across
        # Stop). The instance the UI mutates is the same one the worker thread reads.
        mgr.ensure_instance(slot)
        remove = False
        visible = PyImGui.begin_child(f"slot_{slot.id}", (self.cache.WIDTH - 24, 170), True)
        try:
            if visible:
                self._capture_rect(slot.id)   # this whole window is a drop target
                if PyImGui.button(f"X##rm{slot.id}"):
                    remove = True
                PyImGui.same_line(0.0, 1.0)
                PyImGui.text(slot.behavior_cls.name() + ("  [running]" if slot.running else ""))
                if slot.running:
                    if PyImGui.button(f"Stop##{slot.id}"):
                        mgr.stop(slot)
                elif PyImGui.button(f"Start##{slot.id}"):
                    mgr.start(slot)
                PyImGui.same_line(0.0, 6.0)
                if PyImGui.button((f"Dock##pop{slot.id}") if slot.popped_out else (f"Pop out##pop{slot.id}")):
                    slot.popped_out = not slot.popped_out
                # The behavior's own settings render inline UNLESS popped out to a floating window
                # (drawn separately in _draw_popped_out_windows, at top level outside this child).
                if slot.instance is not None and not slot.popped_out:
                    slot.instance.draw()  # the behavior's own settings (minion goal, follow, ...)
                elif slot.popped_out:
                    PyImGui.text_disabled("(settings in pop-out window)")
                PyImGui.separator()
                PyImGui.text("Clients (drag here):")
                for cid in list(slot.assigned):
                    client = self.cache.network_manager.client_list.get(cid)
                    if client is not None:
                        self._draw_client_row(cid, client)
        finally:
            PyImGui.end_child()    # must close even if a behavior's draw() raised, or the stack unbalances
        if remove:
            mgr.remove_slot(slot)

    def _draw_popped_out_windows(self) -> None:
        """Render each popped-out behavior's settings in its own top-level floating window. Called
        OUTSIDE the host child (ImGui windows must be top-level, not nested in a begin_child). A
        'Dock back in' button (or toggling the slot's Pop out button) returns the settings inline."""
        mgr = self.logic.behaviors
        for slot in list(self.logic.behaviors.slots):
            if not slot.popped_out or slot.instance is None:
                continue
            title = f"{slot.behavior_cls.name()}{'  [running]' if slot.running else ''}##popwin{slot.id}"
            opened = PyImGui.begin(title, PyImGui.WindowFlags.AlwaysAutoResize)
            try:
                if opened:
                    # Same Start/Stop control as the inline slot -- the pop-out is otherwise a
                    # dead end (you'd have to dock back in just to start/stop the behavior).
                    if slot.running:
                        if PyImGui.button(f"Stop##pop{slot.id}"):
                            mgr.stop(slot)
                    elif PyImGui.button(f"Start##pop{slot.id}"):
                        mgr.start(slot)
                    PyImGui.same_line(0.0, 6.0)
                    if PyImGui.button(f"Dock back in##dock{slot.id}"):
                        slot.popped_out = False
                    else:
                        slot.instance.draw()
            finally:
                PyImGui.end()

    def _resolve_drag(self) -> None:
        """End-of-frame: show the drag ghost, and on mouse release drop the client into
        whichever target window the cursor is over -- defaulting to the unassigned pool when
        the release lands on no target (or the dragged client/target vanished)."""
        cid = self.cache.drag_client_id
        if cid is None:
            return
        mgr = self.logic.behaviors
        client = self.cache.network_manager.client_list.get(cid)
        if client is not None:
            PyImGui.set_tooltip(self._client_name(cid, client))
        if PyImGui.is_mouse_released(0):
            io = PyImGui.get_io()
            target = None
            for key, (px, py, sw, sh) in self.cache.drop_rects.items():
                if px <= io.mouse_pos_x <= px + sw and py <= io.mouse_pos_y <= py + sh:
                    target = key
                    break
            if isinstance(target, int):  # a slot id
                slot = next((s for s in mgr.slots if s.id == target), None)
                if slot is not None:
                    mgr.assign(cid, slot)
                else:
                    mgr.unassign(cid)
            else:                        # "pool" or no target -> unassigned pool
                mgr.unassign(cid)
            self.cache.drag_client_id = None
        elif not PyImGui.is_mouse_down(0):
            self.cache.drag_client_id = None   # button already up but we missed the release

    # ----------------------------------------------------------- party-wide commands
    def _all_clients(self):
        """Every connected client (including the host's own local client)."""
        return list(self.cache.network_manager.client_list.items())

    def _draw_party_controls(self):
        """Host-only buttons that command the whole roster at once (HeroAI-style team commands):
        call everyone to the host's outpost and form a party, resign all, interact-with-target all.
        All GW reads here run on the main thread (draw is gated on IsMapReady) and the orders go out
        over each client's transport (the local client executes in-process, remotes over ZMQ)."""
        flow = self.logic.party_flow

        # Sequenced flow: travel -> confirm arrival -> host invites -> clients accept (see
        # PartyFormationFlow). The button kicks it off; it then advances each frame in update().
        if not flow.active:
            if PyImGui.button("Call to Outpost + Join"):
                flow.start()
        else:
            if PyImGui.button("Cancel Call"):
                flow.cancel()

        PyImGui.same_line(0.0, 6.0)
        if PyImGui.button("Resign All"):
            for _cid, client in self._all_clients():
                client.transport.send(RPC.CMD.RESIGN)

        PyImGui.same_line(0.0, 6.0)
        if PyImGui.button("Interact All"):
            # Drive every client to interact with the host's current target (a shared-instance agent
            # id). 0 = no target -> each client picks the nearest enemy in its own world.
            try:
                target_id = int(GW.Player.GetTargetID() or 0)
            except Exception:
                target_id = 0
            for _cid, client in self._all_clients():
                client.transport.send(RPC.CMD.INTERACT, target_id)

        if flow.status:
            PyImGui.text(f"Party: {flow.status}")
        PyImGui.separator()

    def draw_host_window(self):
        """The Host-specific UI section: an unassigned client pool plus one child window per
        behavior. Clients are virtually dragged between them (see _resolve_drag)."""
        mgr = self.logic.behaviors
        self.cache.drop_rects = {}        # rebuilt each frame from the windows drawn below

        # Each begin_child below is paired with end_child in a finally so a mid-frame exception
        # (e.g. a behavior's own draw() raising) can never skip a close and leave ImGui's window
        # stack unbalanced -- an unbalanced stack corrupts the next frame and shows as flicker.
        PyImGui.begin_child("Commander Host", (self.cache.WIDTH, 600), True)
        try:
            # --- add a behavior (kept at the top so it's never scrolled out of view once
            # several behavior slots are open) ---
            names = [cls.name() for cls in self.cache.behavior_map.values()]
            self.cache.add_behavior_index = PyImGui.combo(
                "##addbeh", self.cache.add_behavior_index, names)
            PyImGui.same_line(0.0, 6.0)
            if PyImGui.button("Add Behavior"):
                classes = list(self.cache.behavior_map.values())
                if 0 <= self.cache.add_behavior_index < len(classes):
                    mgr.add_slot(classes[self.cache.add_behavior_index])
            PyImGui.separator()

            # --- party-wide commands (call to outpost + join, resign all, interact all) ---
            self._draw_party_controls()

            # --- unassigned pool ---
            PyImGui.begin_child("pool", (self.cache.WIDTH - 24, 110), True)
            try:
                self._capture_rect("pool")
                PyImGui.text("Unassigned clients:")
                for cid in mgr.unassigned_ids():
                    client = self.cache.network_manager.client_list.get(cid)
                    if client is not None:
                        self._draw_client_row(cid, client)
            finally:
                PyImGui.end_child()

            # --- one window per behavior ---
            for slot in list(mgr.slots):
                self._draw_behavior_slot(slot)
        finally:
            PyImGui.end_child()

        # Popped-out behavior settings windows are TOP-LEVEL (must not be nested in the child above).
        self._draw_popped_out_windows()
        self._resolve_drag()

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
        # NOTE: update_thread_manager() (the watchdog keepalive pump) is now driven from main()
        # UNCONDITIONALLY, before the map-ready gate -- it must run even mid-load or the watchdog
        # reaps the network thread. Keep it OUT of here, since update() only runs once the map is
        # fully ready. This method does the GW-touching work the network thread must NOT do:
        #   - build the player_data snapshot (GW reads) and stash it for the network loop to send,
        #   - execute any RPCs the network loop queued (their GW calls run here, on the main thread).
        # Both run only when the map is live, since update() itself is map-gated in main().
        snapshot = Jsonizer.player_data()
        self.local_game_client.update_from_dict(snapshot)
        self.network_manager.pump_outbound_state(snapshot)
        self.network_manager.drain_inbound_rpcs()
        # Self-heal: keep the host visible in its own client list while hosting, even if a prior
        # _cleanup_host (dormancy/map load) wiped it. Idempotent and cheap.
        if self.cache_ui.is_host:
            self.network_manager.ensure_local_client()
            # Advance the call-to-outpost+join flow (travel -> confirm -> invite -> accept). Runs on
            # the map-gated main thread; idle unless a "Call to Outpost + Join" press started it.
            self.logic.party_flow.tick()

    def update_thread_manager(self):
        if self.thread_manager.is_threads_running:
            self.thread_manager.thread_manager.update_all_keepalives()
            # stop all threads when not trying to connect or host, basically go dormant
            if not (self.cache_ui.is_host or self.cache_ui.is_connected):
                self.thread_manager.is_threads_running = False
                self.logic.behaviors.stop_all()
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
MODULE_NAME = "Central Commander"  # Change this Module name
COLLAPSED = "collapsed"
X_POS = "x"
Y_POS = "y"

# load last‐saved window state (fallback to 100,100 / un-collapsed)
window_x = ini_window.read_int(MODULE_NAME, X_POS, 100)
window_y = ini_window.read_int(MODULE_NAME, Y_POS, 100)
window_collapsed = ini_window.read_bool(MODULE_NAME, COLLAPSED, False)

# Bump the hot-reload epoch BEFORE building the new CentralCommander. Behavior worker threads
# from the previous load capture the old epoch and exit once they see it change (see
# _current_epoch / Behavior.is_running), so a reload can't leave a behavior commanding clients
# from an orphaned thread the new manager has no handle to. (Threads started before this guard
# existed won't self-stop -- restart the GW client once to clear those.)
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

    # PyImGui.begin() MUST be paired with PyImGui.end() on EVERY frame, even if drawing the body
    # throws. If an exception escaped draw_in_window() the end() was skipped, leaving ImGui's
    # window stack unbalanced -> the next frame renders corrupted -> visible flicker. The
    # try/finally makes the pairing unconditional so a transient error degrades to one dropped
    # frame instead of a persistent flicker.
    try:
        if is_window_opened:
            central_commander.draw_in_window()
    finally:
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
        # Publish the "is this instance live" flag for background threads, computed HERE on the
        # main thread where the GW frame cache is valid (map_ready() uses frame-cached checks that
        # return stale values / race the cache if called off-thread). The behavior workers and the
        # client network loop read thread_manager.map_live instead of calling map_ready themselves.
        central_commander.thread_manager.publish_map_live(map_ready())

        # Pump the watchdog keepalives EVERY frame, before the map gate. The watchdog only
        # pauses itself while IsMapLoading() is True, but main()'s gate is stricter (MapValid +
        # IsMapReady + IsPartyLoaded). After a zone there's a window where IsMapLoading() has
        # already cleared (watchdog armed) but the party isn't loaded yet (gate still closed) --
        # if we only pumped inside the gate, the network thread's keepalive would expire (2s) and
        # the watchdog would reap it, dropping the connection. The pump touches no GW APIs, so it
        # is safe to run mid-load.
        central_commander.update_thread_manager()

        # Draw the control window whenever the map DATA is loaded -- deliberately NOT gated on a
        # healthy party. The old gate drew only when MapValid() was true, and MapValid() requires
        # Party.IsPartyLoaded(); so when the party went briefly unhealthy (e.g. a member's client
        # crashed mid-zone) the ENTIRE window vanished and never came back until the party reformed
        # (a full reboot) -- a widget reload couldn't fix it because the broken state lives in the
        # GW process, not the widget. The UI only does ImGui + already-synced blackboard reads plus
        # light GW.Party lookups that return empty safely without a party, so it's fine to draw here.
        # IsMapReady() still excludes the hard-load window (map data not loaded / mid-transition),
        # where even those light calls aren't safe.
        if Routines.Checks.Map.IsMapReady():
            draw_widget()

        # The GW-data refresh (player_data reads the local agent) keeps the full, strict gate.
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
        # Catch-all for any other unexpected exceptions
        Py4GW.Console.Log(MODULE_NAME, f"Unexpected error encountered: {str(e)}", Py4GW.Console.MessageType.Error)
        Py4GW.Console.Log(MODULE_NAME, f"Stack trace: {traceback.format_exc()}", Py4GW.Console.MessageType.Error)
    finally:
        pass


if __name__ == "__main__":
    main()
# endregion
