from dataclasses import dataclass, field
from typing import List, Optional
from three_d_agent.sad.schema import FeedbackEntry


@dataclass
class FeedbackReport:
    version: int
    sim_violations: List[dict] = field(default_factory=list)
    sim_passed: List[str] = field(default_factory=list)
    sim_failed: List[str] = field(default_factory=list)
    human_feedback: Optional[str] = None
    open_issues: List[str] = field(default_factory=list)


def aggregate(
    version: int,
    scene_results: List[dict],
    human_feedback: Optional[str] = None,
) -> FeedbackReport:
    violations: List[dict] = []
    passed: List[str] = []
    failed: List[str] = []
    for r in scene_results:
        if r["passed"]:
            passed.append(r["scene"])
        else:
            failed.append(r["scene"])
            for v in r.get("violations", []):
                violations.append({**v, "scene": r["scene"]})

    open_issues = [
        f"[{v.get('scene', 'unknown')}] {v['kind']}: {v.get('diagnosis_hint', 'no hint')}"
        for v in violations
    ]
    if human_feedback:
        open_issues.append(f"human: {human_feedback}")

    return FeedbackReport(
        version=version,
        sim_violations=violations,
        sim_passed=passed,
        sim_failed=failed,
        human_feedback=human_feedback,
        open_issues=open_issues,
    )


def to_feedback_entries(report: FeedbackReport) -> List[FeedbackEntry]:
    entries: List[FeedbackEntry] = []
    for v in report.sim_violations:
        entries.append(FeedbackEntry(
            version=report.version,
            source="sim",
            metric=v.get("metric"),
            value=v.get("value"),
            expected_max=v.get("expected_max"),
        ))
    if report.human_feedback:
        entries.append(FeedbackEntry(
            version=report.version,
            source="human",
            msg=report.human_feedback,
        ))
    return entries
