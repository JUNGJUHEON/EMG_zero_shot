from __future__ import annotations

import os
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = CODE_DIR.parent
CAPSTONE_ROOT = Path(
    os.environ.get("NINAPRO_CAPSTONE_ROOT", REPOSITORY_ROOT / "local_workspace")
).expanduser()
