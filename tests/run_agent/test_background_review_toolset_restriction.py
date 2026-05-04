"""Tests covering the background review agent's toolset wiring.

Original context (issue #15204): the background skill-review agent once
inherited the full default toolset, allowing it to perform non-skill side
effects (terminal, send_message, delegate_task, etc.). The mitigation at
that time was to hard-restrict ``enabled_toolsets`` to ``["memory", "skills"]``.

Updated design (cache + tree-dedup): restricting the toolset ALSO makes the
review agent's system prompt diverge from the parent's (different skills-block,
different tool-guidance layers), which busts the Anthropic prompt cache AND
prevents preprocessing tree-dedup from collapsing the review into the same
conversation leaf as the main session. To preserve both properties, the review
now inherits the parent's full toolset, and the review prompts carry an
explicit "only use memory/skill tools" instruction. Tool-call safety is
enforced at the prompt layer rather than at the toolset layer.
"""

import threading
from unittest.mock import patch


def _make_agent_stub(agent_cls):
    """Create a minimal AIAgent-like object with just enough state for _spawn_background_review."""
    import datetime as _dt
    agent = object.__new__(agent_cls)
    agent.model = "test-model"
    agent.platform = "test"
    agent.provider = "openai"
    agent.session_id = "sess-123"
    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent.quiet_mode = True
    agent._memory_store = None
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._memory_nudge_interval = 5
    agent._skill_nudge_interval = 5
    agent._cached_system_prompt = None
    agent.enabled_toolsets = ["terminal", "web", "memory", "skills"]
    agent.background_review_callback = None
    agent.status_callback = None
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    return agent


class _SyncThread:
    """Drop-in replacement for threading.Thread that runs the target inline."""

    def __init__(self, *, target=None, daemon=None, name=None):
        self._target = target

    def start(self):
        if self._target:
            self._target()


def test_background_review_agent_inherits_parent_toolsets():
    """The review agent must inherit the parent's full enabled_toolsets so
    its system prompt stays bit-identical to the parent's (Anthropic prompt
    cache + preprocessing tree-dedup). Tool-call scoping is enforced by the
    review prompt's "only use memory/skill tools" instruction, not by a
    narrowed toolset.
    """
    import run_agent

    agent = _make_agent_stub(run_agent.AIAgent)
    captured = {}

    def _capture_init(self, *args, **kwargs):
        captured["enabled_toolsets"] = kwargs.get("enabled_toolsets")
        raise RuntimeError("stop after capturing init args")

    with patch.object(run_agent.AIAgent, "__init__", _capture_init), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(
            messages_snapshot=[],
            review_memory=True,
            review_skills=False,
        )

    assert "enabled_toolsets" in captured, "AIAgent.__init__ was not called"
    assert captured["enabled_toolsets"] == agent.enabled_toolsets, (
        "Background review must inherit the parent's full toolset. "
        "Narrowing it would diverge the review's system prompt from the "
        "parent's, breaking Anthropic prompt cache + preprocessing tree-dedup."
    )


def test_background_review_prompts_instruct_tool_restriction():
    """Since the review inherits the full toolset (see above), the prompts
    themselves must carry the scoping instruction — this is the safety layer
    that replaces the old toolset narrowing from issue #15204.
    """
    import run_agent

    for attr in (
        "_MEMORY_REVIEW_PROMPT",
        "_SKILL_REVIEW_PROMPT",
        "_COMBINED_REVIEW_PROMPT",
    ):
        prompt = getattr(run_agent.AIAgent, attr)
        assert "memory/skill tools" in prompt or "memory and skill" in prompt, (
            f"{attr} must instruct the review agent to only use memory/skill "
            f"tools; this is the safety layer now that the toolset itself is "
            f"inherited from the parent (regression guard for #15204)."
        )
        assert "do not invoke any other tool" in prompt, (
            f"{attr} must explicitly forbid calling non-memory/non-skill tools."
        )
