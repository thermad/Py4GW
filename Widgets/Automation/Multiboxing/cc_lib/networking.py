"""ZMQ networking: ROUTER/PUB/PULL host loop + DEALER/SUB/PUSH client loop.

Owns the host<->client wire protocol and the per-client Client list. Depends on the
lower cc_lib leaves (rpc/transport/blackboard/misc_helpers); nothing depends
on it except the behavior + UI layers above."""
import sys, json, time, uuid
from collections import deque
from typing import Dict, Optional

# Inside Py4GW, zmq is vendored under Py4GWCoreLib.ExternalLibs; its internals do some bare imports,
# so its parent dir must be on sys.path before the (fully-qualified) import. A standalone server has
# no Py4GW -- fall back to a normally installed ``zmq``. Idempotent; safe to re-run on every reload.
try:
    _zmq_lib_path = sys.prefix + "\\Py4GWCoreLib\\ExternalLibs"
    if _zmq_lib_path not in sys.path:
        sys.path.insert(0, _zmq_lib_path)
    import Py4GWCoreLib.ExternalLibs.zmq as zmq
except Exception:
    import zmq

from cc_lib.blackboard import GameClient
from cc_lib.rpc import RPC
from cc_lib.transport import Client, LocalTransport
from cc_lib.misc_helpers import ThreadManager, current_epoch
# NOTE (client slice): world_feed / the host-side world-merge lives only in the server codebase.
# This client keeps maintain_host solely for the party-formation convenience, not the decision
# engine, so it does NOT record client world feeds.


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
        # Keep a permanent handle to the local (host's own) client. It must be re-inserted every
        # time we (re)host because _cleanup_host clears the whole client_list on teardown (e.g.
        # when the widget goes dormant on a map load) and this entry is otherwise only created
        # once at construction -- so without re-ensuring it the host silently vanishes from its
        # own client list after the first dormancy cycle.
        self.local_client_id = uuid.uuid4()
        self.local_client = Client(local_game_client, LocalTransport())

        # GW work is kept OFF the network thread (calling GW from a background thread crashes during
        # instance teardown). The MAIN thread produces the outbound state snapshot and executes
        # inbound RPCs; the network thread only moves bytes.
        #   outbound_state: latest player_data() dict, refreshed by the main thread (update());
        #                   the client network loop just sends it -- no GW call on that thread.
        #   inbound_rpcs:   commands received by the network loop, drained+executed on the main
        #                   thread. Bounded so a backlog during a load can't grow without limit;
        #                   deque append/popleft are individually GIL-atomic (no lock needed).
        self.outbound_state: Optional[dict] = None
        self.inbound_rpcs: "deque" = deque(maxlen=64)
        #   outbound_oneshot: ONE-SHOT payloads (e.g. the map-trapezoid bundle) that must be delivered
        #                     EXACTLY as queued, not folded into the per-frame outbound_state. The
        #                     state dict is overwritten every main frame, so a large one-shot key
        #                     stuffed into it races the ~60ms network send loop and is usually clobbered
        #                     before it ships. This queue is drained+sent as its own PUSH message, so
        #                     every queued bundle reaches the server once. GIL-atomic append/popleft.
        self.outbound_oneshot: "deque" = deque(maxlen=16)

        # Host Sockets
        self.router = None  # RPC (Bidirectional)
        self.pub = None     # Broadcast (Host -> All)
        self.pull = None    # State Updates (All -> Host)

        self.context = None
        self.poller = None

        self.ensure_local_client()

    def ensure_local_client(self) -> None:
        """(Re)insert the host's own client into the list if missing. Idempotent. Must ONLY
        touch client_list -- it runs every frame while hosting, so it must never reset the
        live ZMQ sockets/poller (doing so nulls them under the running network thread)."""
        if self.local_client_id not in self.client_list:
            self.client_list[self.local_client_id] = self.local_client

    def queue_oneshot(self, payload: dict) -> None:
        """MAIN THREAD: enqueue a one-shot upstream payload (a plain JSON-able dict, e.g.
        ``{"cc_map_traps": {...}}``) to be PUSHed to the server exactly once, independent of the
        per-frame ``outbound_state``. Use this (not ``outbound_state``) for anything large/one-shot so
        it isn't clobbered by the next frame's state refresh before the network thread sends it."""
        if payload:
            self.outbound_oneshot.append(payload)

    def pump_outbound_state(self, state: dict) -> None:
        """MAIN THREAD: stash the latest local player_data snapshot for the client network loop
        to send. Keeps GW reads (player_data) off the network thread."""
        self.outbound_state = state

    def drain_inbound_rpcs(self) -> None:
        """MAIN THREAD: execute the commands the client network loop queued. Runs here so every
        RPC's GW calls happen on the main thread (frame-cache valid, pause-safe). The caller only
        invokes this when the map is live (it's driven from the map-gated update())."""
        q = self.inbound_rpcs
        while q:
            try:
                method, args, kwargs = q.popleft()
            except IndexError:
                break
            try:
                RPC.call(method, *args, **kwargs)
            except Exception as e:
                print(f"inbound RPC '{method}' error: {e}")

    def setup_zmq_host(self, port):
        # A previous host session's _cleanup_host may have cleared the list; make sure the host
        # is back in its own client list before the new session starts polling.
        self.ensure_local_client()
        self.context = zmq.Context()

        # ROUTER: Handles RPC requests from clients and sends direct replies
        self.router = self.context.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.LINGER, 0)
        self.router.bind(f"tcp://*:{port}")

        # PUB: Broadcasts data to all connected clients
        self.pub = self.context.socket(zmq.PUB)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.bind(f"tcp://*:{port + 1}")

        # PULL: Collects state updates from all clients (Async/Non-blocking)
        self.pull = self.context.socket(zmq.PULL)
        self.pull.setsockopt(zmq.LINGER, 0)
        self.pull.bind(f"tcp://*:{port + 2}")

        self.poller = zmq.Poller()
        self.poller.register(self.router, zmq.POLLIN)
        self.poller.register(self.pull, zmq.POLLIN)

    def maintain_host(self, port, terminal_function):
        self.setup_zmq_host(port)
        # See maintain_client: bail when a hot-reload bumps the epoch so an orphaned host loop
        # (whose thread_manager.is_threads_running stays True) retires instead of running forever.
        epoch = current_epoch()
        try:
            while self.thread_manager.is_threads_running and epoch == current_epoch():
                # A malformed/partial frame from a crashing client (json.loads / recv_json / a
                # bad identity) must not escape and kill the host network thread -- catch per
                # iteration and keep serving the surviving clients.
                try:
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
                                # (client slice) the observed-world "cc_world" payload is consumed
                                # only by the server's world-merge; a client-host ignores it.
                except Exception as e:
                    print(f"maintain_host loop error (continuing): {e}")

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
            dealer.setsockopt(zmq.LINGER, 0)   # never block close()/term() on a dead host
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
            sub.setsockopt(zmq.LINGER, 0)
            sub.connect(f"tcp://{ip}:{port + 1}")
            sub.setsockopt_string(zmq.SUBSCRIBE, "")

            # PUSH: Send state updates to Host PULL
            push = context.socket(zmq.PUSH)
            push.setsockopt(zmq.LINGER, 0)
            push.connect(f"tcp://{ip}:{port + 2}")

            poller = zmq.Poller()
            poller.register(dealer, zmq.POLLIN)
            poller.register(sub, zmq.POLLIN)

            # Capture the hot-reload generation. If the widget reloads, the new CentralCommander
            # builds a fresh ThreadManager + network thread but has NO handle to this one, and our
            # old thread_manager.is_threads_running stays True -- so without this check the stale
            # (pre-fix) loop runs forever and keeps crashing on load screens. Bailing when the epoch
            # moves lets a reload cleanly retire this thread instead of orphaning it.
            epoch = current_epoch()
            while self.thread_manager.is_threads_running and epoch == current_epoch():
                # This thread makes NO GW calls. All GW work is on the main thread:
                #   - outbound: the main thread refreshes self.outbound_state (player_data); here we
                #     just SEND that pre-built dict -- no GW read on this thread, so a hard load that
                #     pauses the main thread can never make us read GW into a dying instance.
                #   - inbound: we ENQUEUE commands; the main thread drains+executes them (gated on
                #     map_ready there). We still always recv to drain the ZMQ buffer, but only
                #     enqueue while is_map_live() so stale orders during a load are discarded rather
                #     than replayed on the new map. is_map_live() is freshness-checked, so a frozen
                #     flag from a paused main thread reads as not-live.
                #
                # A malformed frame / transient ZMQ error must not escape and kill this thread
                # (which would silently drop the connection). Catch per iteration and continue.
                try:
                    # A. Send the latest main-thread-built state snapshot to the host.
                    if self.thread_manager.is_map_live():
                        snapshot = self.outbound_state
                        if snapshot is not None:
                            snapshot = dict(snapshot)        # shallow copy: don't race the main thread
                            snapshot["client_id"] = str(client_id)
                            push.send_json(snapshot)

                    # A2. Drain one-shot payloads (map-trapezoid bundle, ...) as their OWN PUSH
                    # messages so each is delivered intact -- unlike outbound_state, which the main
                    # thread overwrites every frame, a queued one-shot can't be clobbered before it
                    # ships. Pure bytes (built on the main thread), so no map-live gate needed.
                    while self.outbound_oneshot:
                        try:
                            one = self.outbound_oneshot.popleft()
                        except IndexError:
                            break
                        one = dict(one)
                        one["client_id"] = str(client_id)
                        push.send_json(one)

                    # B. Check for incoming messages
                    events = dict(poller.poll(timeout=10))

                    if dealer in events:
                        msg_full = dealer.recv_multipart()       # always drain the frame
                        msg: Dict = json.loads(msg_full[-1].decode('utf-8'))
                        method = msg.get("method", None)
                        # Enqueue for the main thread (no host reply: the host ROUTER drops RPC
                        # return values by design -- client state flows back via PUSH/PULL).
                        if method and self.thread_manager.is_map_live():
                            self.inbound_rpcs.append((method, msg.get("args", []), msg.get("kwargs", {})))

                    if sub in events:
                        broadcast_data = sub.recv_json()         # always drain the frame
                        method = broadcast_data.get("method", None)
                        if method and self.thread_manager.is_map_live():
                            self.inbound_rpcs.append((method, broadcast_data.get("args", []),
                                                      broadcast_data.get("kwargs", {})))
                except Exception as e:
                    print(f"maintain_client loop error (continuing): {e}")

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

