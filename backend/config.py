from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
# Local development only. Never send credentials to the browser or export them.
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
RUNTIME = Path(os.getenv("PRISM_RUNTIME", str(ROOT / "runtime")))
RUNTIME.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

