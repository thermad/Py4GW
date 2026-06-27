"""Standalone helpers: booster cache, Vec2 math, thread manager -- hot-reloadable cc_lib leaf."""
import time
import math

import Py4GWCoreLib as GW


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

