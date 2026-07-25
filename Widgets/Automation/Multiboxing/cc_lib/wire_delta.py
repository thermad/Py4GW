"""Pure, schema-driven (de)serialization + delta/keyframe codec for the wire contract.

Consumes ONLY ``protocol`` -- no game lib, no I/O, no threads. Both the client (encode side) and the
server (decode side) share this. It is driven entirely by the dataclass field types + rate/deadband
metadata in ``protocol``, so adding a field there needs zero changes here.

Two things live here:

  * KEYFRAME (full):   ``to_wire(snapshot)`` -> plain JSON-able dict;  ``from_wire(dict, Snapshot)``
                       rebuilds the dataclass. STATIC fields are structurally excluded.
  * DELTA (changed):   ``diff(prev_wire, cur_wire)`` -> minimal delta;  ``apply(base_wire, delta)``
                       merges it back. Continuous HOT fields are culled by their deadband; keyed
                       collections (world, effects, buffs, skillbar, attributes) carry explicit
                       removal TOMBSTONES so "absent" means "unchanged", never "deleted" -- this is
                       what lets the receiver MERGE instead of clear-and-replace without ever
                       stranding a ghost effect/agent.

Wire representation of a delta for a keyed collection::

    {"c": { key: <sub-delta-or-full>, ... }, "r": [key, ...]}   # c=changed/added, r=removed

Everything operates on the plain-dict ("wire") form; the dataclass instances only exist at the
to_wire / from_wire boundaries.
"""
import dataclasses
from typing import get_origin, get_args, Dict, List

from . import protocol as P


# --------------------------------------------------------------------------- type classification
def _is_dc(tp) -> bool:
    return dataclasses.is_dataclass(tp)


def _kind(tp):
    """Classify a field's annotated type into how the codec treats it."""
    origin = get_origin(tp)
    if _is_dc(tp):
        return ("dc", tp)
    if origin in (dict, Dict):
        _k, v = get_args(tp)
        return ("dict_dc", v) if _is_dc(v) else ("dict_scalar", v)
    if origin in (list, List):
        return ("list", get_args(tp)[0])
    return ("scalar", tp)


# --------------------------------------------------------------------------- keyframe: to/from wire
def to_wire(obj) -> dict:
    """Serialize a schema dataclass instance to a JSON-able dict of its wire (non-STATIC) fields."""
    out = {}
    for name, f in P.iter_wire_fields(type(obj)):
        out[name] = _val_to_wire(f.type, getattr(obj, name))
    return out


def _val_to_wire(tp, val):
    kind, elem = _kind(tp)
    if kind == "dc":
        return to_wire(val) if val is not None else None
    if kind == "dict_dc":
        return {_ikey(k): to_wire(v) for k, v in (val or {}).items()}
    if kind == "dict_scalar":
        return {_ikey(k): v for k, v in (val or {}).items()}
    if kind == "list":
        return list(val or [])
    return val


def from_wire(data: dict, dc_type):
    """Rebuild a schema dataclass from its wire dict. Missing fields fall back to the dataclass
    default. Pure -- produces plain dataclasses, never game-lib objects."""
    kwargs = {}
    for name, f in P.iter_wire_fields(dc_type):
        if data is None or name not in data:
            continue
        kwargs[name] = _val_from_wire(f.type, data[name])
    return dc_type(**kwargs)


def _val_from_wire(tp, val):
    kind, elem = _kind(tp)
    if kind == "dc":
        return from_wire(val, elem) if val is not None else None
    if kind == "dict_dc":
        return {_ikey(k): from_wire(v, elem) for k, v in (val or {}).items()}
    if kind == "dict_scalar":
        return {_ikey(k): v for k, v in (val or {}).items()}
    if kind == "list":
        return list(val or [])
    return val


def _ikey(k):
    """Dict keys in this schema are agent/skill/attribute ids -> ints. JSON stringifies keys, so
    coerce back to int on the way in and out (falls back to the raw key if non-numeric)."""
    try:
        return int(k)
    except (TypeError, ValueError):
        return k


# --------------------------------------------------------------------------- delta: diff
_SENTINEL = object()


def diff(prev_wire: dict, cur_wire: dict, dc_type=P.Snapshot):
    """Minimal delta of two wire dicts of ``dc_type``. Returns {} when nothing changed. ``prev_wire``
    MUST be the last state the sender actually SENT (post-deadband), not the last raw reading --
    otherwise deadband error accumulates and the receiver drifts."""
    delta = {}
    for name, f in P.iter_wire_fields(dc_type):
        pv = (prev_wire or {}).get(name, _SENTINEL)
        cv = (cur_wire or {}).get(name, _SENTINEL)
        if cv is _SENTINEL:
            continue
        sub = _diff_val(f.type, pv, cv, P.field_deadband(f))
        if sub is not _SENTINEL:
            delta[name] = sub
    return delta


def _diff_val(tp, pv, cv, deadband):
    kind, elem = _kind(tp)
    fresh = pv is _SENTINEL

    if kind == "dc":
        d = diff(pv if not fresh else {}, cv or {}, elem)
        return d if d else _SENTINEL

    if kind == "dict_dc":
        return _diff_keyed(pv, cv, lambda p, c: diff(p, c, elem))

    if kind == "dict_scalar":
        return _diff_keyed(pv, cv, lambda p, c: (c if p != c else None))

    if kind == "list":
        return list(cv or []) if (fresh or list(pv or []) != list(cv or [])) else _SENTINEL

    # scalar
    if fresh:
        return cv
    if deadband and _is_number(pv) and _is_number(cv):
        return cv if abs(cv - pv) >= deadband else _SENTINEL
    return cv if pv != cv else _SENTINEL


def _diff_keyed(pv, cv, sub_diff):
    """Diff a keyed collection into {'c': changed/added, 'r': removed-keys}. ``sub_diff(p, c)``
    returns the per-item delta (or None if that item is unchanged)."""
    pv = pv if pv is not _SENTINEL else {}
    cv = cv or {}
    changed = {}
    for k, c in cv.items():
        # Added items are diffed against an EMPTY base so their representation is delta-shaped and
        # uniform with changed items -- crucial for nested collections (a new agent's effects must
        # still come through tombstone-wrapped, or apply() would drop them).
        p = pv[k] if k in pv else {}
        d = sub_diff(p, c)
        if d:
            changed[k] = d
    removed = [k for k in pv if k not in cv]
    if not changed and not removed:
        return _SENTINEL
    out = {}
    if changed:
        out["c"] = changed
    if removed:
        out["r"] = removed
    return out


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# --------------------------------------------------------------------------- delta: apply
def apply(base_wire: dict, delta: dict, dc_type=P.Snapshot) -> dict:
    """Merge ``delta`` onto a COPY of ``base_wire`` (the receiver's last known full state), applying
    tombstones. Returns the new full wire dict. Never mutates ``base_wire``."""
    out = dict(base_wire or {})
    for name, f in P.iter_wire_fields(dc_type):
        if name not in (delta or {}):
            continue
        out[name] = _apply_val(f.type, out.get(name), delta[name])
    return out


def _apply_val(tp, base, d):
    kind, elem = _kind(tp)
    if kind == "dc":
        return apply(base or {}, d or {}, elem)
    if kind == "dict_dc":
        return _apply_keyed(base, d, lambda b, sd: apply(b or {}, sd, elem))
    if kind == "dict_scalar":
        return _apply_keyed(base, d, lambda b, sd: sd)
    if kind == "list":
        return list(d or [])
    return d


def _apply_keyed(base, d, sub_apply):
    out = dict(base or {})
    for k in (d.get("r") or []):
        out.pop(_ikey(k), None)
    for k, sd in (d.get("c") or {}).items():
        k = _ikey(k)
        out[k] = sub_apply(out.get(k), sd)
    return out
