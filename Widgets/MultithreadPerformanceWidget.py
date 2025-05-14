import Py4GWCoreLib as GW
import time

class Cache():
    def __init__(self):
        self.timer = time.time()
        self.fps_poll = time.time()
        self.load_delay = time.time()
        self.fps = 60

cache = Cache()

def main():
    global cache
    if not GW.Routines.Checks.Map.MapValid():
        cache.load_delay = time.time()
        return
    current = time.time()
    if current - cache.load_delay < 5: #since this widget exists only to increase single thread performance of other scripts the delay shouldn't be an issue
        return
    if current - cache.fps_poll > 1:
        cache.fps = GW.UIManager.GetFPSLimit()
        cache.fps_poll = current
    if fps != 0: # 0 is uncapped fps
        delta = current - cache.timer
        if delta < 1/fps:
            sleep((1/fps-delta))
            cache.timer = time.time()