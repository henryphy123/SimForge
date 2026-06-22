import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VersionArtifact:
    version: int
    path: Path


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def asset_dir(self, session_id: str, asset_name: str) -> Path:
        return self.root / "sessions" / session_id / asset_name

    def commit_version(
        self, session_id: str, asset_name: str, version: int, build_dir: Path,
    ) -> VersionArtifact:
        """Promote an already-built directory to the final v{version} dir.

        Unlike create_version (which auto-numbers and writes files from strings),
        this honors a caller-supplied version so the orchestrator stays in control
        of version numbers across an iteration loop.
        """
        final_dir = self.asset_dir(session_id, asset_name) / f"v{version}"
        if final_dir.exists():
            raise FileExistsError(f"version {version} already exists")
        Path(build_dir).rename(final_dir)
        return VersionArtifact(version=version, path=final_dir)

    def create_version(
        self, session_id: str, asset_name: str,
        sad_json: str, mjcf: str,
        extra_files: dict[str, str] | None = None,
    ) -> VersionArtifact:
        asset_dir = self.asset_dir(session_id, asset_name)
        existing = [p for p in asset_dir.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
        next_version = len(existing) + 1
        v_path = asset_dir / f"v{next_version}"
        v_path.mkdir(parents=True, exist_ok=False)
        (v_path / "sad.json").write_text(sad_json, encoding="utf-8")
        (v_path / "asset.mjcf").write_text(mjcf, encoding="utf-8")
        if extra_files:
            for name, content in extra_files.items():
                (v_path / name).write_text(content, encoding="utf-8")
        return VersionArtifact(version=next_version, path=v_path)

    def copy_version(
        self, session_id: str, asset_name: str, src_version: int,
    ) -> VersionArtifact:
        """Snapshot an existing version into a new auto-numbered version dir.

        History stays append-only: rolling back to v1 produces a fresh vN that
        is a faithful copy of v1, rather than mutating or deleting later work.
        """
        src = self.get_version(session_id, asset_name, src_version)
        if src is None:
            raise FileNotFoundError(f"version {src_version} not found")
        asset_dir = self.asset_dir(session_id, asset_name)
        existing = [p for p in asset_dir.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
        next_version = len(existing) + 1
        dst = asset_dir / f"v{next_version}"
        shutil.copytree(src.path, dst)
        return VersionArtifact(version=next_version, path=dst)

    def list_versions(self, session_id: str, asset_name: str) -> list[VersionArtifact]:
        asset_dir = self.asset_dir(session_id, asset_name)
        if not asset_dir.exists():
            return []
        versions = []
        for p in asset_dir.glob("v*"):
            if p.is_dir() and p.name[1:].isdigit():
                versions.append(VersionArtifact(version=int(p.name[1:]), path=p))
        versions.sort(key=lambda v: v.version)
        return versions

    def latest_version(self, session_id: str, asset_name: str) -> VersionArtifact | None:
        versions = self.list_versions(session_id, asset_name)
        return versions[-1] if versions else None

    def get_version(self, session_id: str, asset_name: str, version: int) -> VersionArtifact | None:
        for v in self.list_versions(session_id, asset_name):
            if v.version == version:
                return v
        return None
