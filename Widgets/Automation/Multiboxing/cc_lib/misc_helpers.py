"""Standalone helpers: booster cache, Vec2 math, thread manager -- hot-reloadable cc_lib leaf."""
import time
import math

# GW present only inside Py4GW. Vec2 (pure math) is needed on the headless server; the thread/map
# helpers below are client-side and never run there, so guard the import for GW-free importability.
try:
    import Py4GWCoreLib as GW
    from Py4GWCoreLib import Routines
except Exception:
    GW = None
    Routines = None


def current_epoch() -> int:
    """Hot-reload generation counter, stored on the MultiThreading *class* (loaded once;
    Py4GWCoreLib is not re-imported when this widget hot-reloads, so it survives a reload of
    the cc_lib files). The widget bumps it on every (re)load. A long-lived worker/network
    thread captures it at start and stops when it changes, so a thread orphaned by a reload
    -- one the new CentralCommander has no handle to -- self-terminates instead of running
    its stale code (and crashing on load screens) forever. Mirrors behaviors._current_epoch."""
    if GW is None:
        return 0   # headless server: no hot-reload epoch (the process owns its own lifecycle)
    return getattr(GW.MultiThreading, "_cc_epoch", 0)


def map_ready() -> bool:
    """True only when the local instance is fully live (map valid AND finished loading).

    MAIN-THREAD ONLY. This calls frame-cached GW functions (Checks.Map.*), and the frame cache
    is a single global dict reset by a main-thread PreUpdate callback -- so off the main thread
    it returns stale values and races the cache clear(). Background threads must NOT call this;
    they read ThreadManager.map_live, which the main thread publishes from here every frame.

    Defensive try/except because these checks can themselves throw mid-teardown -> not-ready."""
    if Routines is None:
        return False   # headless server: no local map (it isn't in the game)
    try:
        return Routines.Checks.Map.MapValid() and Routines.Checks.Map.IsMapReady()
    except Exception:
        return False


# --------------------------------------------------------------------------- map trapezoid bundling
# The server asks ONE client (RPC.CMD.DUMP_MAP_TRAPS) to bundle the current instance's pathing
# trapezoids so it can draw the map geometry in its world view. The bundle is LARGE and needed only
# ONCE per map, so it does NOT ride the per-frame cc_world stream: the handler stashes it here and the
# widget attaches it to the very next PUSH as ``cc_map_traps`` (mirrors the one-shot cc_request
# piggyback). Handler runs on the MAIN THREAD (inbound RPCs drain there, map-gated), so the GW map
# reads are legal.
_pending_map_traps = None


def dump_map_trapezoids():
    """CLIENT-SIDE RPC handler: read every pathing trapezoid of the current instance and bundle it
    for upload. Each trapezoid becomes a flat 8-float quad -- its four corners in GW WORLD UNITS,
    ordered TL, TR, BR, BL: ``[XTL,YT, XTR,YT, XBR,YB, XBL,YB]`` (a PathingTrapezoid's top edge spans
    XTL..XTR at YT, its bottom edge XBL..XBR at YB). World units are the same space AgentView.x/y use,
    so the server's world-view Camera renders the geometry under the agents with no extra transform.

    All zplane layers are flattened into one quad list (the top-down view ignores height), matching
    Map.Pathing.GetComputedGeometry. Best-effort: a bad layer/trapezoid is skipped, and a total read
    failure yields an empty bundle rather than raising (the server just gets no geometry this map)."""
    global _pending_map_traps
    if GW is None:
        return
    quads = []
    map_id = 0
    try:
        map_id = int(GW.Map.GetMapID() or 0)
        for layer in (GW.Map.Pathing.GetPathingMaps() or []):
            for t in (getattr(layer, "trapezoids", None) or []):
                try:
                    quads.append([float(t.XTL), float(t.YT), float(t.XTR), float(t.YT),
                                  float(t.XBR), float(t.YB), float(t.XBL), float(t.YB)])
                except Exception:
                    continue
    except Exception as e:
        print(f"[client] dump_map_trapezoids failed: {e}")
    _pending_map_traps = {"map_id": map_id, "quads": quads}
    print(f"[client] bundled {len(quads)} map trapezoids for map {map_id}")


def take_pending_map_traps():
    """MAIN THREAD: pop the one-shot map-trapezoid bundle (or None if none pending). The widget calls
    this each frame and attaches a non-None result to the outbound PUSH exactly once."""
    global _pending_map_traps
    bundle = _pending_map_traps
    _pending_map_traps = None
    return bundle


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
    # A frozen flag must read as NOT live. During a hard load screen Py4GW stops rendering, so the
    # main thread stops calling main() and stops refreshing map_live -- it freezes at its last value
    # (True). A background thread reading that stale True would call GW into the dying instance and
    # crash. So we stamp every publish and trust the flag only if it was refreshed this recently.
    # Must comfortably exceed the main-thread frame interval (even a slow/hitching frame) but be far
    # below a load duration. Bias low: a false "stale" only costs a skipped heartbeat (harmless); a
    # false "live" can crash.
    MAP_LIVE_FRESHNESS = 0.25

    def __init__(self):
        self.thread_manager = GW.MultiThreading(2.0, log_actions=True)
        self.is_threads_running = False
        # Main-thread-published "is the local instance live" signal. Background threads (behavior
        # workers, the client network loop) MUST read it via is_map_live() -- NEVER call map_ready()
        # themselves: map_ready() goes through frame-cached GW functions whose cache is a single
        # global dict reset by a main-thread PreUpdate callback, so off-thread it returns stale
        # values AND races the cache's clear(). The main thread calls publish_map_live() every frame.
        self.map_live = False
        self.map_live_ts = 0.0

    def publish_map_live(self, value: bool) -> None:
        """MAIN THREAD ONLY: publish the current liveness + a freshness timestamp."""
        self.map_live = value
        self.map_live_ts = time.time()

    def is_map_live(self) -> bool:
        """Thread-safe liveness for background threads: the main-thread flag, but only if it was
        refreshed within MAP_LIVE_FRESHNESS. If the main thread stalled (hard load pausing main()),
        the flag is stale -> report NOT live so background work stops touching GW even though the
        last published value was a frozen True. Plain attribute reads are GIL-atomic."""
        return self.map_live and (time.time() - self.map_live_ts) < self.MAP_LIVE_FRESHNESS

    def start_thread(self, name, func):
        self.is_threads_running = True
        # Add sequential threads
        self.thread_manager.add_thread(name, func)
        # Watchdog thread is necessary to async close other running threads
        self.thread_manager.start_watchdog(name)

    def stop_sequential_environment(self):
        self.thread_manager.stop_all_threads()
        self.is_threads_running = False

