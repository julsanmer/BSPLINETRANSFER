"""Archive grid trajectory CSVs without disturbing resumable solver artifacts.

Each departure/revolution group gets one ZIP containing its trajectory CSVs at
their original relative paths.  An archive is fully written and CRC-checked
before any source CSV is removed.  Repeated calls replace entries from reruns.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np


def archive_path(run_dir: Path) -> tuple[Path, str]:
    """Return the archive and member name for one grid run directory."""
    if run_dir.name.startswith("N") and run_dir.parent.name.startswith("tof_"):
        departure = run_dir.parents[1]
        return departure / "trajectories.zip", str(
            (run_dir / "pythagorean_sundman" / "trajectory.csv").relative_to(departure))
    revolution = run_dir.parents[1]
    departure = revolution.parent
    if not departure.name.startswith("dep_") or not revolution.name.startswith("N"):
        raise ValueError(f"Not a departure/revolution grid run: {run_dir}")
    return departure / f"trajectories_{revolution.name}.zip", str(
        (run_dir / "pythagorean_sundman" / "trajectory.csv").relative_to(departure))


def load_trajectory(run_dir: Path) -> np.ndarray:
    """Read a loose or archived trajectory with the same numeric precision."""
    csv_path = run_dir / "pythagorean_sundman" / "trajectory.csv"
    if csv_path.exists():
        return np.loadtxt(csv_path, delimiter=",", skiprows=1)
    archive, member = archive_path(run_dir)
    with zipfile.ZipFile(archive) as zf:
        with zf.open(member) as stream:
            return np.loadtxt(stream, delimiter=",", skiprows=1)


def compact_group(group: Path, *, min_age_seconds: float = 0) -> tuple[int, int, int]:
    """Return (files removed, original bytes, archive bytes) for one group."""
    if group.name.startswith("dep_"):
        departure = group
        archive = departure / "trajectories.zip"
        sources = sorted(group.glob("tof_*/N*/pythagorean_sundman/trajectory.csv"))
    else:
        departure = group.parent
        archive = departure / f"trajectories_{group.name}.zip"
        sources = sorted(group.glob("tof_*/*/pythagorean_sundman/trajectory.csv"))
    if not sources:
        return 0, 0, 0
    now = time.time()
    if any(now - path.stat().st_mtime < min_age_seconds for path in sources):
        return 0, 0, 0
    replacements = {str(path.relative_to(departure)): path for path in sources}
    original_bytes = sum(path.stat().st_size for path in sources)
    with tempfile.NamedTemporaryFile(prefix=f".{archive.stem}.", suffix=".zip",
                                     dir=departure, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as target:
            if archive.exists():
                with zipfile.ZipFile(archive) as old:
                    for member in old.infolist():
                        if member.filename not in replacements:
                            target.writestr(member, old.read(member.filename))
            for member, path in replacements.items():
                target.write(path, member)
        with zipfile.ZipFile(temporary) as check:
            if check.testzip() is not None:
                raise OSError(f"ZIP CRC verification failed: {temporary}")
            for member, path in replacements.items():
                if check.getinfo(member).file_size != path.stat().st_size:
                    raise OSError(f"ZIP size mismatch: {member}")
        os.replace(temporary, archive)
        for path in sources:
            path.unlink()
        return len(sources), original_bytes, archive.stat().st_size
    finally:
        temporary.unlink(missing_ok=True)


def compact_root(root: Path, *, min_age_seconds: float = 0) -> tuple[int, int]:
    files = bytes_saved = 0
    groups = sorted(root.glob("dep_*/N*")) + sorted(root.glob("dep_*"))
    for group in groups:
        count, original, archived = compact_group(
            group, min_age_seconds=min_age_seconds)
        files += count
        bytes_saved += original - archived
        if count:
            print(f"{group}: {count} CSVs, {original / 1e6:.1f} MB "
                  f"-> {archived / 1e6:.1f} MB", flush=True)
    return files, bytes_saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--min-age-seconds", type=float, default=3600,
                        help="Skip groups with recently modified CSVs (default: 3600)")
    args = parser.parse_args()
    for root in args.roots:
        files, saved = compact_root(root, min_age_seconds=args.min_age_seconds)
        print(f"{root}: archived {files} trajectories; saved {saved / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
