import time
from dataclasses import dataclass, field


MAX_ITERATIONS = 12
COST_CAP_USD = 0.50
WALL_CLOCK_CAP_S = 15 * 60

INPUT_PRICE_PER_MTOK = 3.0
OUTPUT_PRICE_PER_MTOK = 15.0


def estimate_cost(usage) -> float:
    """Estimate USD cost for one response from its token usage.

    Returns 0.0 when usage is absent (e.g. mock clients), so the gate degrades
    gracefully instead of raising.
    """
    if usage is None:
        return 0.0
    in_tok = getattr(usage, "input_tokens", 0) or 0
    out_tok = getattr(usage, "output_tokens", 0) or 0
    return (in_tok * INPUT_PRICE_PER_MTOK + out_tok * OUTPUT_PRICE_PER_MTOK) / 1_000_000


@dataclass
class SafetyState:
    iteration: int = 0
    cost_usd: float = 0.0
    start_time_s: float = field(default_factory=time.time)
    max_iterations: int = MAX_ITERATIONS


@dataclass
class SafetyDecision:
    should_stop: bool
    reason: str


def check(state: SafetyState) -> SafetyDecision:
    if state.iteration >= state.max_iterations:
        return SafetyDecision(True, f"max_iterations ({state.max_iterations}) reached")
    if state.cost_usd >= COST_CAP_USD:
        return SafetyDecision(True, f"cost_cap (${COST_CAP_USD}) exceeded")
    elapsed = time.time() - state.start_time_s
    if elapsed >= WALL_CLOCK_CAP_S:
        return SafetyDecision(True, f"wall_clock_cap ({WALL_CLOCK_CAP_S}s) exceeded")
    return SafetyDecision(False, "")
