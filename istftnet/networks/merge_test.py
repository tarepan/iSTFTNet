"""Test merge."""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field, asdict

from .merge import merge


def test_merge_no_nest_exclusive():
    merged =   merge({"a": 1,       },
                    {        "b": 2})
    assert merged == {"a": 1, "b": 2}
    print("Tests passed")
    print(merged)

def test_merge_no_nest_full_update():
    merged =   merge({"a": 1},
                    {"a": 2})
    assert merged == {"a": 2}
    print("Tests passed")
    print(merged)

def test_merge_no_nest_partial_update():
    merged =   merge({"a": 1, "b": 3},
                    {"a": 2,       })
    assert merged == {"a": 2, "b": 3}
    print("Tests passed")
    print(merged)

def test_merge_no_nest_partial_new():
    merged =   merge({"a": 1,       },
                    {"a": 2, "b": 3})
    assert merged == {"a": 2, "b": 3}
    print("Tests passed")
    print(merged)

####
def test_merge_nest_exclusive():
    merged = merge(  {"a": {"aa": 1}                },
                    {                "b": {"bb": 2}})
    assert merged == {"a": {"aa": 1}, "b": {"bb": 2}}
    print("Tests passed")
    print(merged)

def test_merge_nest_full_update():
    merged = merge(  {"a": {"aa": 1}},
                    {"a": {"aa": 2}})
    assert merged == {"a": {"aa": 2}}
    print("Tests passed")
    print(merged)

def test_merge_nest_partial_update():
    merged =   merge({"a": {"aa": 1}, "b": {"bb": 2}},
                    {"a": {"aa": 2},              })
    assert merged == {"a": {"aa": 2}, "b": {"bb": 2}}
    print("Tests passed")
    print(merged)

def test_merge_nest_partial_new():
    merged =   merge({"a": {"aa": 1}                },
                    {"a": {"aa": 2}, "b": {"bb": 3}})
    assert merged == {"a": {"aa": 2}, "b": {"bb": 3}}
    print("Tests passed")
    print(merged)

####
def test_merge_nest_child_exclusive():
    merged = merge(  {"a": {"aa": 1,        }},
                    {"a": {         "AA": 2}})
    assert merged == {"a": {"aa": 1, "AA": 2}}
    print("Tests passed")
    print(merged)

def test_merge_nest_child_partial_update():
    merged = merge(  {"a": {"aa": 1, "AA": 3}},
                    {"a": {         "AA": 2}})
    assert merged == {"a": {"aa": 1, "AA": 2}}
    print("Tests passed")
    print(merged)

def test_merge_nest_child_partial_new():
    merged = merge(  {"a": {"aa": 1,        }},
                    {"a": {"aa": 2, "AA": 2}})
    assert merged == {"a": {"aa": 2, "AA": 2}}
    print("Tests passed")
    print(merged)

def test_merge_list_primitive():
    merged = merge(  [1, 2, 3,],
                    [4, 5, 6,])
    assert merged == [4, 5, 6,]
    print("Tests passed")
    print(merged)

def test_merge_list_dict():
    merged = merge(  [{"a": 1}, {"b": 3}, {"a": 2, "b": 2},],
                    [{"a": 2}, {"b": 2}, {"a": 1,       },],)
    assert merged == [{"a": 2}, {"b": 2}, {"a": 1, "b": 2},]
    print("Tests passed")
    print(merged)

def test_merge_list_dict_list():
    merged = merge(  [{"a": [1, 1, 3]}, {"b": [1, {"c": 1}]}],
                    [{"a": [2, 2, 5]}, {"b": [2, {"c": 2}]}],)
    assert merged == [{"a": [2, 2, 5]}, {"b": [2, {"c": 2}]}]
    print("Tests passed")
    print(merged)

def test_merge_instance():

    @dataclass
    class Child:
        attr4: int = 0
        attr5: bool = False

    @dataclass
    class ClsTest:
        attr1: int = 1
        attr2: str = "one"
        child: Child = field(default_factory=lambda: deepcopy(Child(1, False)))
        childlen: list[Child] = field(default_factory=lambda: [Child(attr5=True)])

    conf = {
        "attr1": 2,
        "child": {
            "attr5": True
        },
        "childlen": [
            {"attr4": 3}, {"attr5": False}, {"attr4": 10}
        ]
    }
    gt = ClsTest(
        attr1 = 2,
        attr2 = "one",
        child = Child(
            attr4 = 1,
            attr5 = True),
        childlen = [
            Child(3, True), Child(0, False), Child(10, True)]
        )

    merged = merge(ClsTest(), conf)
    assert merged == gt
    print("Tests passed")
    print(asdict(merged))
