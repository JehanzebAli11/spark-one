"""Environment check for the Spark: GPU, packages, model files, teacher server."""

import importlib
import json
import os
import platform
import sys
import urllib.request

out = {"python": sys.version.split()[0], "machine": platform.machine()}
for mod in ("torch", "transformers", "peft", "torchvision", "fastapi", "PIL", "numpy", "httpx"):
    try:
        m = importlib.import_module(mod)
        out[mod] = getattr(m, "__version__", "ok")
    except Exception as e:  # noqa: BLE001
        out[mod] = f"MISSING ({e.__class__.__name__})"
try:
    import torch

    out["cuda"] = torch.cuda.is_available()
    if out["cuda"]:
        out["gpu"] = torch.cuda.get_device_name(0)
        out["bf16"] = torch.cuda.is_bf16_supported()
except Exception:
    pass
base = os.environ.get("TEACHER_BASE_URL", "http://localhost:8000/v1")
try:
    with urllib.request.urlopen(base + "/models", timeout=5) as r:
        out["teacher_models"] = [m["id"] for m in json.load(r)["data"]]
except Exception as e:  # noqa: BLE001
    out["teacher_models"] = f"unreachable ({e.__class__.__name__})"
print(json.dumps(out, indent=2))
bad = [k for k, v in out.items() if isinstance(v, str) and v.startswith("MISSING")]
sys.exit(1 if bad else 0)
