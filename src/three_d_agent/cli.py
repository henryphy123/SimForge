import json
import os
import uuid
from pathlib import Path
import typer
from three_d_agent.artifact.store import ArtifactStore
from three_d_agent.orchestrator.client import build_client, DEFAULT_MODEL
from three_d_agent.orchestrator.agent import OrchestratorConfig, run
from three_d_agent.orchestrator.repl import auto_accept_callback, make_interactive_callback

app = typer.Typer(help="3D Agent: text/image/video to MuJoCo MJCF asset")


def _main():
    """3D Agent CLI."""
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


app.callback()(_main)


def _store() -> ArtifactStore:
    root = os.environ.get("THREE_D_AGENT_ROOT")
    return ArtifactStore(root=Path(root) if root else Path.home() / ".3d-agent")


def _slug(text: str) -> str:
    return (text or "").strip().lower().replace(" ", "_")[:32] or "asset"


def _unique_asset_name(store: ArtifactStore, session_id: str, base: str) -> str:
    """Return base, or base_2/base_3/... if that asset already has versions."""
    name = base
    n = 2
    while store.list_versions(session_id, name):
        name = f"{base}_{n}"
        n += 1
    return name


def _run_session(
    store: ArtifactStore, session_id: str, text: str, asset_name: str,
    model: str, yes: bool,
) -> None:
    """Write this asset's inputs, connect to the model, and drive the loop."""
    session_dir = store.root / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "inputs.json").write_text(json.dumps({"text": text}))

    typer.echo(f"[session {session_id}, asset '{asset_name}']")
    typer.echo(f"  inputs: {json.dumps({'text': text})}")
    typer.echo(f"  model: {model}")
    typer.echo(f"  artifact root: {store.root}")
    typer.echo("Connecting to model provider...")

    try:
        client = build_client()
    except RuntimeError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1)

    human_callback = auto_accept_callback if yes else make_interactive_callback()
    cfg = OrchestratorConfig(
        session_id=session_id,
        asset_name=asset_name,
        store=store,
        client=client,
        model=model,
        human_callback=human_callback,
    )
    result = run(cfg)

    if result["ok"]:
        typer.echo(f"Done! Final asset at: {result['final_path']}")
    else:
        typer.echo(f"Failed: {result['reason']} (iterations: {result['iterations']})")
        raise typer.Exit(code=1)


@app.command()
def new(
    text: str = typer.Option(None, "--text", "-t", help="Text input"),
    session_id: str = typer.Option(None, "--session", "-s", help="Existing session id"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Model name"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-accept (non-interactive)"),
):
    """Start a new 3D asset generation session."""
    if session_id is None:
        session_id = uuid.uuid4().hex[:8]
    store = _store()
    asset_name = _unique_asset_name(store, session_id, _slug(text))
    _run_session(store, session_id, text, asset_name, model, yes)


@app.command()
def add(
    session_id: str = typer.Option(..., "--session", "-s", help="Existing session id"),
    text: str = typer.Option(None, "--text", "-t", help="Text input"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Model name"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-accept (non-interactive)"),
):
    """Add another asset to an existing session."""
    store = _store()
    session_dir = store.root / "sessions" / session_id
    if not session_dir.exists():
        typer.echo(f"Error: session {session_id} does not exist (use 'new' first)")
        raise typer.Exit(code=1)
    asset_name = _unique_asset_name(store, session_id, _slug(text))
    _run_session(store, session_id, text, asset_name, model, yes)


@app.command("list")
def list_assets(
    session_id: str = typer.Option(..., "--session", "-s", help="Session id"),
):
    """List the assets in a session with their latest version and state."""
    store = _store()
    session_dir = store.root / "sessions" / session_id
    if not session_dir.exists():
        typer.echo(f"Error: session {session_id} does not exist")
        raise typer.Exit(code=1)

    typer.echo(f"{session_id}/")
    found = False
    for asset_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
        versions = store.list_versions(session_id, asset_dir.name)
        if not versions:
            continue
        found = True
        latest = versions[-1]
        state = "finalized" if (latest.path / "FINALIZED").exists() else "draft"
        typer.echo(f"  {asset_dir.name}/  (v{latest.version}, {state})")
    if not found:
        typer.echo("  (no assets yet)")


def _flatten(obj, prefix: str = "") -> dict:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _diff_sad(old: dict, new: dict) -> list[str]:
    fo, fn = _flatten(old), _flatten(new)
    lines = []
    for key in sorted(set(fo) | set(fn)):
        a = fo.get(key, "<absent>")
        b = fn.get(key, "<absent>")
        if a != b:
            lines.append(f"{key}: {a} -> {b}")
    return lines


@app.command()
def diff(
    session_id: str = typer.Option(..., "--session", "-s", help="Session id"),
    asset: str = typer.Option(..., "--asset", "-a", help="Asset name"),
    v1: int = typer.Argument(..., help="First version number"),
    v2: int = typer.Argument(..., help="Second version number"),
):
    """Show the field-level SAD diff between two versions of an asset."""
    store = _store()
    a1 = store.get_version(session_id, asset, v1)
    a2 = store.get_version(session_id, asset, v2)
    if a1 is None or a2 is None:
        missing = v1 if a1 is None else v2
        typer.echo(f"Error: version v{missing} of {asset} not found")
        raise typer.Exit(code=1)

    sad1 = json.loads((a1.path / "sad.json").read_text(encoding="utf-8"))
    sad2 = json.loads((a2.path / "sad.json").read_text(encoding="utf-8"))
    lines = _diff_sad(sad1, sad2)

    typer.echo(f"diff {asset} v{v1} -> v{v2}:")
    if not lines:
        typer.echo("  (no SAD changes)")
    for line in lines:
        typer.echo(f"  {line}")


@app.command()
def rollback(
    session_id: str = typer.Option(..., "--session", "-s", help="Session id"),
    asset: str = typer.Option(..., "--asset", "-a", help="Asset name"),
    version: int = typer.Argument(..., help="Version to roll back to"),
):
    """Roll an asset back to an earlier version (copied forward as a new one)."""
    store = _store()
    if store.get_version(session_id, asset, version) is None:
        typer.echo(f"Error: version v{version} of {asset} not found")
        raise typer.Exit(code=1)
    artifact = store.copy_version(session_id, asset, version)
    (artifact.path / "FINALIZED").write_text("1", encoding="utf-8")
    typer.echo(
        f"rolled back {asset} to v{version} -> new v{artifact.version} (finalized)"
    )


if __name__ == "__main__":
    app()
