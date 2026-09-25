import subprocess
from importlib.metadata import version
from pathlib import Path

from pier.constants import PYPI_PACKAGE_NAME


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


_base_version = version(PYPI_PACKAGE_NAME)
_commit_hash = _git_commit_hash()
__version__ = f"{_base_version}+{_commit_hash}" if _commit_hash else _base_version
