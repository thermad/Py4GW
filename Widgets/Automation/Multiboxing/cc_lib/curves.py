"""Response curves for the utility scorer -- the IAUS pattern (Dave Mark's Infinite Axis Utility
System): a consideration normalizes some game input to 0..1, then a *curve* shapes how that input
maps to utility. Keeping the curves here (a pure leaf, no deps -- like misc_helpers) means a score
is read as "input through a named curve at a weight" instead of opaque arithmetic, so tuning is
editing a curve, not hunting magic numbers.

Convention: every function takes a 0..1 input and returns a 0..1 output, EXCEPT ``ramp`` (and the
``*_between`` helpers) which map a RAW value range onto 0..1 for you. Compose them:

    s += 0.6 * curves.ramp(hp_fraction, threshold, 0.0)   # deeper HP deficit -> higher utility
    s += 0.3 * curves.quadratic(curves.ramp(dist, near, far))

The scorer stays additive-within-a-tier (the tuned, tested model); curves only make each term
explicit and reshapeable. Pick a curve by the SHAPE you want:
  linear      - proportional.
  quadratic   - ease-in: stays low, ramps hard near the top (use for "only matters when extreme").
  cubic       - sharper ease-in than quadratic.
  inv_quad    - ease-out: jumps early, flattens near the top (use for "any amount matters a lot").
  smoothstep  - S-curve, gentle at both ends.
  logistic    - steep S around a midpoint (use for soft thresholds).
  threshold   - hard step at a cutoff.
"""


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def ramp(value: float, lo: float, hi: float) -> float:
    """Map a RAW ``value`` from the range [lo, hi] onto 0..1 (clamped). ``hi`` MAY be less than
    ``lo`` for an inverse ramp -- the idiom for a deficit/near-expiry term where a LOWER raw value
    means HIGHER utility: ``ramp(hp_fraction, threshold, 0.0)`` is 0 at the threshold and 1 at 0."""
    if hi == lo:
        return 1.0 if value >= lo else 0.0
    return clamp01((value - lo) / (hi - lo))


def linear(x: float) -> float:
    return clamp01(x)


def quadratic(x: float) -> float:
    x = clamp01(x)
    return x * x


def cubic(x: float) -> float:
    x = clamp01(x)
    return x * x * x


def power(x: float, exponent: float) -> float:
    """``x`` raised to ``exponent`` (the generic ease-in knob; exponent>1 stays low longer). This is
    the curve FollowLeader hand-rolls as ``t ** CURVE_EXP``."""
    return clamp01(x) ** exponent


def inv_quad(x: float) -> float:
    """Ease-out: rises fast then flattens -- 1 - (1-x)^2."""
    x = clamp01(x)
    return 1.0 - (1.0 - x) * (1.0 - x)


def smoothstep(x: float) -> float:
    x = clamp01(x)
    return x * x * (3.0 - 2.0 * x)


def logistic(x: float, steepness: float = 10.0, midpoint: float = 0.5) -> float:
    """Steep S-curve around ``midpoint`` -- a soft threshold. ``steepness`` higher == sharper."""
    import math
    try:
        return clamp01(1.0 / (1.0 + math.exp(-steepness * (clamp01(x) - midpoint))))
    except OverflowError:
        return 0.0 if x < midpoint else 1.0


def threshold(x: float, at: float = 0.5) -> float:
    """Hard step: 1.0 at/above ``at``, else 0.0."""
    return 1.0 if x >= at else 0.0


def inverse(x: float) -> float:
    """1 - x: high when the input is low."""
    return clamp01(1.0 - clamp01(x))
