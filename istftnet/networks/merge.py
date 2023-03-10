from __future__ import annotations
from typing import Any, TypeVar
from copy import deepcopy
from dataclasses import field


primitives = (None, bool, int, float, str)

def merge(base: Any, another: Any) -> Any:
    """Merge two objects.

    Args:
        base -
        another -
    """
    type_base, type_another = type(base), type(another)

    # Instance-Dict merge
    if type_base not in (primitives + (dict, list)):
        if type_another is not dict:
            raise RuntimeError(f"{type_base} should be equal to {type_another}")
        else:
            return merge_instance(base, another)

    # Type validation
    if type_base is not type_another:
        raise RuntimeError(f"{type_base} should be equal to {type_another}")

    # Dict-Dict merge
    if type_base is dict:
        return merge_dict(base, another)

    # List-List merge
    elif type_base is list:
        return merge_list(base, another)

    # Primitive-Primitive merge
    else:
        # TODO: null merge
        # TODO: interpolation override
        return another


def merge_dict(base: dict[str, Any], another: dict[str, Any]) -> dict[str, Any]:
    """Merge two dictionaries."""
    # Shallow merge
    merged = {**base, **another}
    # Deep merge
    for k, base_v in base.items():
        if k in another:
            merged[k] = merge(base_v, another[k])
    return merged


def merge_list(base: list[Any], another: list[Any]) -> list[Any]:
    """Merge two Generic-length lists."""
    # Generic-Length validation
    if len(base) is not len(another):
        raise RuntimeError(f"Merged lists should have same length, but {len(base)} != {len(another)}")
    # Element-wise merge
    return [merge(base_i, another_i) for base_i, another_i in zip(base, another)]


def merge_instance(base: Any, another: dict[str, Any]) -> Any:
    """Merge a dictionary `another` into a class instance `base`."""
    merged = base
    for k, another_v in another.items():
        print(base)
        print(k)
        base_v = getattr(base, k)
        type_base_v, type_another_v = type(base_v), type(another_v)
        # Generic-length list in dataclass, initialized with default element
        if type_base_v is list:
            if type_another_v is not list:
                raise RuntimeError(f"another[${k}] should be list, but {type_another_v}")
            setattr(merged, k, merge([deepcopy(base_v[0]) for _ in range(len(another_v))], another_v))
        else:
            setattr(merged, k, merge(base_v, another_v))
    return base


T1 = TypeVar('T1')
def default(instance:T1):
    return field(default_factory=lambda:deepcopy(instance))


T2 = TypeVar('T2')
def list_default(instance:T2):
    return field(default_factory=lambda:[deepcopy(instance)])
