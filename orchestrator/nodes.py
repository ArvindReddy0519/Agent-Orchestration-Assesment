from orchestrator.state import OrchestratorState


def planner(state: OrchestratorState) -> OrchestratorState:
    """Produce an implementation plan from the requirement."""
    _ = state
    return {}


def coder(state: OrchestratorState) -> OrchestratorState:
    """Generate or update code according to the plan."""
    _ = state
    return {}


def reviewer(state: OrchestratorState) -> OrchestratorState:
    """Review code and record errors or approval signals."""
    _ = state
    return {}
