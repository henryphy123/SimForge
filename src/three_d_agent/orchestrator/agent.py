import json
import time
from dataclasses import dataclass
from typing import Callable, Any
from three_d_agent.artifact.store import ArtifactStore
from .client import build_client, DEFAULT_MODEL
from .tools import (
    load_inputs, validate_sad, build_asset, run_scenes,
    present_to_human, finalize, ToolError,
)
from .safety import SafetyState, check as safety_check, estimate_cost, MAX_ITERATIONS
from .system_prompt import SYSTEM_PROMPT


TOOL_DEFINITIONS = [
    {"name": "load_inputs", "description": "Load session inputs",
     "input_schema": {"type": "object", "properties": {
         "session_id": {"type": "string"}}, "required": ["session_id"]}},
    {"name": "validate_sad", "description": "Validate SAD JSON",
     "input_schema": {"type": "object", "properties": {
         "sad_json": {"type": "string"}}, "required": ["sad_json"]}},
    {"name": "build_asset", "description": "Build MJCF from SAD",
     "input_schema": {"type": "object", "properties": {
         "sad_json": {"type": "string"}, "version": {"type": "integer"}},
         "required": ["sad_json", "version"]}},
    {"name": "run_scenes", "description": "Run test scenes",
     "input_schema": {"type": "object", "properties": {
         "artifact_path": {"type": "string"}, "sad_json": {"type": "string"}},
         "required": ["artifact_path", "sad_json"]}},
    {"name": "present_to_human", "description": "Show progress and get feedback",
     "input_schema": {"type": "object", "properties": {
         "version_summary": {"type": "string"}},
         "required": ["version_summary"]}},
    {"name": "finalize", "description": "Mark the chosen artifact as final",
     "input_schema": {"type": "object", "properties": {
         "artifact_path": {"type": "string"}}, "required": ["artifact_path"]}},
]


@dataclass
class OrchestratorConfig:
    session_id: str
    asset_name: str
    store: ArtifactStore
    client: Any
    model: str = DEFAULT_MODEL
    human_callback: Callable[[str], str] = lambda s: "accept"
    max_iterations: int = MAX_ITERATIONS


def _execute_tool(name: str, args: dict, cfg: OrchestratorConfig, ctx: dict) -> dict:
    if name == "load_inputs":
        return load_inputs(cfg.session_id, cfg.store)
    if name == "validate_sad":
        return validate_sad(args["sad_json"])
    if name == "build_asset":
        result = build_asset(args["sad_json"], args["version"],
                             cfg.session_id, cfg.asset_name, cfg.store)
        if result.get("ok"):
            ctx["thumbnail_path"] = result.get("thumbnail_path")
            ctx["version"] = args["version"]
        return result
    if name == "run_scenes":
        return run_scenes(args["artifact_path"], args["sad_json"])
    if name == "present_to_human":
        return present_to_human(
            args["version_summary"], cfg.human_callback,
            thumbnail_path=ctx.get("thumbnail_path"),
            feedback_path=ctx.get("feedback_path"),
            version=ctx.get("version", 0),
        )
    if name == "finalize":
        return finalize(args["artifact_path"])
    raise ToolError(f"unknown tool: {name}")


def run(cfg: OrchestratorConfig) -> dict:
    safety = SafetyState(max_iterations=cfg.max_iterations)
    messages = []

    session_dir = cfg.store.root / "sessions" / cfg.session_id
    log_path = session_dir / "session.jsonl"
    ctx = {
        "feedback_path": str(session_dir / "feedback_history.json"),
        "thumbnail_path": None,
        "version": 0,
    }

    def _log(event: dict) -> None:
        event = {"ts": time.time(), **event}
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")
        except (OSError, TypeError):
            pass

    inputs = load_inputs(cfg.session_id, cfg.store)
    _log({"type": "session_start", "session_id": cfg.session_id,
          "asset": cfg.asset_name, "inputs": inputs})
    initial_prompt = (
        f"Generate a SAD for this input. Session: {cfg.session_id}, asset: {cfg.asset_name}.\n"
        f"Inputs: {json.dumps(inputs)}\n"
        f"Start by calling load_inputs, then propose a SAD."
    )
    messages.append({"role": "user", "content": initial_prompt})

    cached_system = [{
        "type": "text", "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }]
    cached_tools = [dict(t) for t in TOOL_DEFINITIONS]
    cached_tools[-1] = {**cached_tools[-1], "cache_control": {"type": "ephemeral"}}

    while True:
        decision = safety_check(safety)
        if decision.should_stop:
            _log({"type": "session_end", "ok": False, "reason": decision.reason,
                  "iterations": safety.iteration})
            return {"ok": False, "reason": decision.reason, "iterations": safety.iteration}

        response = cfg.client.messages.create(
            model=cfg.model,
            max_tokens=8192,
            system=cached_system,
            tools=cached_tools,
            messages=messages,
        )
        safety.iteration += 1
        safety.cost_usd += estimate_cost(getattr(response, "usage", None))

        if response.stop_reason == "tool_use":
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})
            for block in assistant_content:
                if block.type == "tool_use":
                    result = _execute_tool(block.name, block.input, cfg, ctx)
                    _log({"type": "tool_call", "tool": block.name,
                          "input": block.input, "result": result})
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }]
                    })
                    if block.name == "finalize" and result.get("ok"):
                        _log({"type": "session_end", "ok": True,
                              "final_path": result["final_path"],
                              "iterations": safety.iteration})
                        return {"ok": True, "final_path": result["final_path"],
                                "iterations": safety.iteration}
            continue

        if response.stop_reason == "length":
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content":
                "Your previous response was truncated. Continue and call a tool now."})
            continue

        _log({"type": "session_end", "ok": False, "reason": "no tool call",
              "iterations": safety.iteration})
        return {"ok": False, "reason": "no tool call", "iterations": safety.iteration}
