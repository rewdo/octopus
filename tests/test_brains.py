"""Tests for all 7 Octopus brains (runnable without pytest)."""
import asyncio, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from octopus.brains.base import BrainRequest, BrainType, TaskComplexity, TaskRisk
from octopus.brains.cheap_brain import CheapBrain
from octopus.brains.skill_brain import SkillBrain
from octopus.brains.action_brain import ActionBrain
from octopus.brains.memory_brain import MemoryBrain
from octopus.brains.planning_brain import PlanningBrain
from octopus.brains.frontier_brain import FrontierBrain
from octopus.brains.world_brain import WorldBrain


def test_cheap_greeting():
    cb = CheapBrain()
    r = BrainRequest(task_id="t1", user_input="hello")
    resp = asyncio.run(cb.process(r))
    assert resp.success
    assert resp.confidence > 0

def test_cheap_code():
    cb = CheapBrain()
    r = BrainRequest(task_id="t2", user_input="write a python function")
    resp = asyncio.run(cb.process(r))
    assert resp.brain_type == BrainType.CHEAP
    assert len(resp.content) > 0

def test_skill_can_handle():
    sb = SkillBrain()
    assert sb.can_handle(BrainRequest(task_id="t3", user_input="x", relevant_skills=["summarize"])) is True
    assert sb.can_handle(BrainRequest(task_id="t4", user_input="hello")) is False

def test_action_can_handle():
    ab = ActionBrain()
    assert ab.can_handle(BrainRequest(task_id="t5", user_input="ls", allowed_tools=["shell"])) is True
    assert ab.can_handle(BrainRequest(task_id="t6", user_input="hello")) is False

def test_memory_can_handle():
    mb = MemoryBrain()
    assert mb.can_handle(BrainRequest(task_id="t7", user_input="recall yesterday")) is True
    assert mb.can_handle(BrainRequest(task_id="t8", user_input="之前做了什么")) is True
    assert mb.can_handle(BrainRequest(task_id="t9", user_input="hello")) is False

def test_planning_process():
    pb = PlanningBrain()
    r = BrainRequest(task_id="t10", user_input="setup python project", complexity=TaskComplexity.MODERATE)
    assert pb.can_handle(r) is True
    resp = asyncio.run(pb.process(r))
    assert resp.success

def test_frontier_can_handle():
    fb = FrontierBrain()
    assert fb.can_handle(BrainRequest(task_id="t11", user_input="anything")) is True

def test_world_state():
    wb = WorldBrain()
    r = BrainRequest(task_id="t12", user_input="what is the state of the system?")
    assert wb.can_handle(r) is True
    resp = asyncio.run(wb.process(r))
    assert resp.success
    assert isinstance(resp.structured_output, dict)

def test_world_irrelevant():
    wb = WorldBrain()
    assert wb.can_handle(BrainRequest(task_id="t13", user_input="hello")) is False


if __name__ == "__main__":
    tests = [v for k, v in list(locals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} passed")
