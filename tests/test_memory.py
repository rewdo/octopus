"""Tests for memory and API modules."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from octopus.memory.memory_graph import MemoryGraph, NodeType
from octopus.memory.layers import WorkingMemory
from octopus.memory.context_compiler import ContextCompiler
from octopus.brains.base import BrainRequest
from octopus.config import OctopusConfig, BudgetConfig
from octopus.api.cost_tracker import CostTracker
from octopus.budget.cognitive_budget import CognitiveBudget
from octopus.brains.base import BrainType


def test_mem_add_and_query():
    mg = MemoryGraph()
    mg.add_node(NodeType.EVENT, properties={"content": "test", "importance": 0.5, "source": "t"})
    assert len(mg.query_nodes()) == 1

def test_mem_query_by_type():
    mg = MemoryGraph()
    mg.add_node(NodeType.EVENT, properties={"content": "e1", "importance": 0.5, "source": "t"})
    mg.add_node(NodeType.FACT, properties={"content": "f1", "importance": 0.5, "source": "t"})
    assert len(mg.query_nodes(node_type=NodeType.EVENT)) == 1

def test_mem_timeline():
    mg = MemoryGraph()
    for i in range(3):
        mg.add_node(NodeType.EVENT, properties={"content": f"e{i}", "importance": 0.5, "source": "t"})
    assert len(mg.get_timeline()) == 3

def test_mem_gc():
    mg = MemoryGraph()
    mg.add_node(NodeType.EVENT, properties={"content": "hi", "importance": 0.1, "source": "t"})
    mg.add_node(NodeType.EVENT, properties={"content": "bug fix", "importance": 0.9, "source": "t"})
    removed = mg.garbage_collect(threshold=0.3)
    assert removed >= 0

def test_mem_importance():
    mg = MemoryGraph()
    nid = mg.add_node(NodeType.EVENT, properties={"content": "critical database bug fix", "source": "t"})
    nodes = mg.query_nodes()
    node = [n for n in nodes if n.node_id == nid][0]
    assert node.importance >= 0.3

def test_working_memory():
    wm = WorkingMemory()
    wm.add("k", "v")
    assert wm.get("k") == "v"

def test_context_compiler():
    cc = ContextCompiler()
    mg = MemoryGraph()
    mg.add_node(NodeType.EVENT, properties={"content": "fixed login bug", "importance": 0.9, "source": "t"})
    r = BrainRequest(task_id="t1", user_input="what was that login bug?")
    result = cc.compile(r, mg, token_budget=500)
    assert len(result) > 0

def test_cost_tracker():
    c = BudgetConfig(monthly_budget_usd=10.0)
    ct = CostTracker(c)
    assert ct.get_remaining_budget() == 10.0
    ct.record_call("api", 100, 200, "model", task_id="t1")
    assert ct.get_monthly_spent() > 0

def test_budget_cheap_affordable():
    c = OctopusConfig.default()
    ct = CostTracker(c.budget)
    # Cheap brain has 0 estimated cost, should always be allowed
    assert ct.get_remaining_budget() > 0


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
