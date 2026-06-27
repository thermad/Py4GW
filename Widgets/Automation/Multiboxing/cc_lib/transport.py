"""Client + transport abstraction (local in-process vs ZMQ lives in networking) -- hot-reloadable cc_lib."""
from abc import ABC, abstractmethod

from cc_lib.rpc import RPC
from cc_lib.blackboard import GameClient


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
        RPC.call(method, *args, **kwargs)

