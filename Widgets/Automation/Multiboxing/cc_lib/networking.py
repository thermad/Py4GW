"""ZMQ networking: ROUTER/PUB/PULL host loop + DEALER/SUB/PUSH client loop.

Owns the host<->client wire protocol and the per-client Client list. Depends on the
lower cc_lib leaves (rpc/transport/blackboard/jsonizers/misc_helpers); nothing depends
on it except the behavior + UI layers above."""
import os, sys, json, time, uuid
from typing import Dict

# zmq's vendored internals do some bare imports, so its parent dir must be on sys.path
# before the (fully-qualified) import below. Idempotent; safe to re-run on every reload.
_zmq_lib_path = sys.prefix + "\\Py4GWCoreLib\\ExternalLibs"
if _zmq_lib_path not in sys.path:
    sys.path.insert(0, _zmq_lib_path)
import Py4GWCoreLib.ExternalLibs.zmq as zmq

from cc_lib.jsonizers import Jsonizer
from cc_lib.blackboard import GameClient
from cc_lib.rpc import RPC
from cc_lib.transport import Client, LocalTransport
from cc_lib.misc_helpers import ThreadManager


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

    def setup_zmq_host(self, port):
        # A previous host session's _cleanup_host may have cleared the list; make sure the host
        # is back in its own client list before the new session starts polling.
        self.ensure_local_client()
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

