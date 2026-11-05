"""Convert teacher-score rows into prediction rows so the eval harness can grade the teacher.

python -m spark_one.teacher.to_preds runs/teacher/test.jsonl runs/teacher/test_preds.jsonl
"""

import json
import math
import sys

from ..schema import read_jsonl


def main(argv=None):
    src, dst = (argv or sys.argv[1:])[:2]
    n = 0
    with open(dst, "w") as f:
        for r in read_jsonl(src):
            lp = [[math.log(max(p, 1e-12)) for p in s["probs"]] for s in r["steps"]]
            f.write(json.dumps({"id": r["id"], "base_id": r["id"], "variant": "base", "logp": lp,
                                "latency_ms": r.get("latency_ms"), "source": "teacher_logprobs",
                                "coverage": [s["coverage"] for s in r["steps"]]}) + "\n")
            n += 1
    print(f"wrote {n} rows to {dst}")


if __name__ == "__main__":
    main()
