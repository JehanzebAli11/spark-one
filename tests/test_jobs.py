import json
import re
from pathlib import Path

JOBS = sorted((Path(__file__).parent.parent / "jobs").glob("*.json"))
NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def test_job_files_follow_spark_lab_format():
    assert JOBS, "no job files"
    names = set()
    for p in JOBS:
        j = json.loads(p.read_text())
        assert NAME.match(j["name"]), p
        assert j["name"] not in names
        names.add(j["name"])
        assert isinstance(j["cmd"], list) and j["cmd"] and all(isinstance(c, str) for c in j["cmd"])
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in j.get("env", {}).items())
        assert j.get("timeout_min", 720) > 0 and j.get("retries", 1) >= 0
        cwd = Path(j.get("cwd", "."))
        assert not cwd.is_absolute() and ".." not in cwd.parts[1:]
        if "train" in j["name"] and "lora" in j["name"]:
            assert "resume_dir" in j
