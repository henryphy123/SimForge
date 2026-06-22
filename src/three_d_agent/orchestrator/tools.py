import json
import shutil
from pathlib import Path
from three_d_agent.sad.schema import SAD
from three_d_agent.sad.validator import validate
from three_d_agent.generator.router import route_and_build
from three_d_agent.runner.mujoco_runner import run_scene
from three_d_agent.runner.thumbnail import render_thumbnail
from three_d_agent.artifact.store import ArtifactStore
from three_d_agent.feedback.aggregator import aggregate


class ToolError(Exception):
    pass


def load_inputs(session_id: str, store: ArtifactStore) -> dict:
    inputs_path = store.root / "sessions" / session_id / "inputs.json"
    if not inputs_path.exists():
        raise ToolError(f"session {session_id} has no inputs")
    return json.loads(inputs_path.read_text())


def validate_sad(sad_json: str) -> dict:
    try:
        sad = SAD.model_validate_json(sad_json)
    except Exception as e:
        return {"ok": False, "errors": [str(e)], "warnings": []}
    r = validate(sad)
    return {"ok": r.ok, "errors": r.errors, "warnings": r.warnings}


def build_asset(sad_json: str, version: int, session_id: str, asset_name: str,
                store: ArtifactStore) -> dict:
    try:
        sad = SAD.model_validate_json(sad_json)
    except Exception as e:
        return {"ok": False, "error": f"SAD parse error: {e}"}
    r = validate(sad)
    if not r.ok:
        return {"ok": False, "error": f"SAD invalid: {r.errors}"}

    asset_dir = store.asset_dir(session_id, asset_name)
    if (asset_dir / f"v{version}").exists():
        return {"ok": False, "error": f"version {version} already exists"}

    work_dir = asset_dir / f"v{version}_build"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    try:
        _, builder_name = route_and_build(sad, work_dir)
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        return {"ok": False, "error": str(e)}

    artifact = store.commit_version(session_id, asset_name, version, work_dir)

    thumbnail_path = None
    try:
        thumbnail_path = render_thumbnail(
            str(artifact.path / "asset.mjcf"), str(artifact.path / "thumbnail.png")
        )
    except Exception:
        pass

    return {"ok": True, "artifact_path": str(artifact.path),
            "builder": builder_name, "thumbnail_path": thumbnail_path}


def run_scenes(artifact_path: str, sad_json: str) -> dict:
    sad = SAD.model_validate_json(sad_json)
    artifact = Path(artifact_path)
    mjcf_path = artifact / "asset.mjcf"
    requested = ["gravity_settle"] + [s for s in sad.test_scenes if s != "gravity_settle"]
    scenes_to_run = list(dict.fromkeys(requested))
    results = []
    for scene_name in scenes_to_run:
        try:
            result = run_scene(scene_name, str(mjcf_path), sad=sad)
        except Exception as e:
            result = {"scene": scene_name, "passed": False,
                      "violations": [{"kind": "scene_error", "diagnosis_hint": str(e)}]}
        results.append(result)
    (artifact / "scene_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    report = aggregate(_version_from_path(artifact), results)
    (artifact / "feedback.json").write_text(
        json.dumps({
            "version": report.version,
            "passed": report.sim_passed,
            "failed": report.sim_failed,
            "open_issues": report.open_issues,
        }, indent=2),
        encoding="utf-8",
    )
    return {
        "results": results,
        "passed": report.sim_passed,
        "failed": report.sim_failed,
        "open_issues": report.open_issues,
    }


def _version_from_path(artifact: Path) -> int:
    name = artifact.name
    if name.startswith("v") and name[1:].isdigit():
        return int(name[1:])
    return 0


def present_to_human(
    version_summary: str, human_callback,
    thumbnail_path: str | None = None,
    feedback_path: str | None = None,
    version: int = 0,
) -> dict:
    summary = version_summary
    if thumbnail_path:
        summary = f"{summary}\n[thumbnail] {thumbnail_path}"
    reply = human_callback(summary)

    if feedback_path and reply and reply.strip().lower() != "accept":
        path = Path(feedback_path)
        history = []
        if path.exists():
            try:
                history = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                history = []
        history.append({"version": version, "source": "human", "msg": reply})
        path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return {"human_reply": reply}


def finalize(artifact_path: str) -> dict:
    artifact = Path(artifact_path)
    (artifact / "FINALIZED").write_text("1", encoding="utf-8")
    return {"ok": True, "final_path": str(artifact)}
