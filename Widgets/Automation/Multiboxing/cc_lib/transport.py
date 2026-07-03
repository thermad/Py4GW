"""Client + transport abstraction (local in-process vs ZMQ lives in networking) -- hot-reloadable cc_lib."""
import time
from abc import ABC, abstractmethod

from cc_lib.rpc import RPC
from cc_lib.blackboard import GameClient


class Client:
    class Transport(ABC):
        @abstractmethod
        def send(self, method :RPC.CMD, *args: object, **kwargs: object):
            ...

    # Per-client floor between MOVE commands. GW.Player.Move sets a destination the character walks
    # to on its own, so re-issuing every tick (MoveTo/FollowLeader/StandInDoubleDragon all run at a
    # ~0.1s interval) only spams the wire and re-paths the character. ALL movement routes through
    # move() so the whole party honors this throttle regardless of which node issued the order.
    MOVE_THROTTLE = 1.0

    def __init__(self, game: GameClient, transport: Transport):
        self.game_client = game
        self.transport = transport
        self._last_move = 0.0

    def move(self, x: float, y: float) -> bool:
        """Order this client to walk to (x, y), throttled to at most once per ``MOVE_THROTTLE``
        seconds. Returns whether the command was actually sent (False = suppressed by the throttle).
        The character keeps walking toward the last ordered point between sends, so a dropped order
        is not a stall -- the next one just refines the destination."""
        now = time.time()
        if now - self._last_move < self.MOVE_THROTTLE:
            return False
        self._last_move = now
        self.transport.send(RPC.CMD.MOVE, x, y)
        return True


class LocalTransport(Client.Transport):
    def send(self, method: RPC.CMD, *args: object, **kwargs: object) -> None:
        RPC.call(method, *args, **kwargs)

