#!/usr/bin/env python3
"""Grade predictions and report accuracy.

    python score_voxmembench.py --predictions predictions_32k.jsonl \
        --judge openai:gpt-4o-mini --out metrics_32k.json

The judge sees only the question, the gold and the response, never the audio.
Verdicts are cached back into the predictions file, so re-running does not pay
for the same item twice.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voxmembench import judge as vjudge
from voxmembench import metrics as vmetrics


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--judge", default="exact-match",
                    help="'exact-match' for a cheap smoke test, or "
                         "'openai:<model id>' for the real metric")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--out", default=None, help="where to write the report")
    ap.add_argument("--regrade", action="store_true",
                    help="ignore cached verdicts and judge every item again")
    args = ap.parse_args()

    kwargs = {}
    if args.judge.startswith("openai:"):
        kwargs = {"base_url": args.base_url, "api_key": args.api_key}
    grader = vjudge.build(args.judge, **kwargs)

    rows = load(args.predictions)
    todo = [r for r in rows
            if args.regrade or ("verdict" not in r and r.get("response") is not None)]
    print("%d predictions, %d to grade with %s"
          % (len(rows), len(todo), getattr(grader, "name", args.judge)))

    for n, row in enumerate(todo, 1):
        verdict, raw = grader(row, row.get("response"))
        row["verdict"] = verdict
        row["judge"] = getattr(grader, "name", args.judge)
        row["judge_raw"] = raw
        if n % 25 == 0:
            print("  graded %d/%d" % (n, len(todo)), flush=True)
            save(args.predictions, rows)
    if todo:
        save(args.predictions, rows)

    for row in rows:
        row.setdefault("verdict", None)
    report = vmetrics.aggregate(rows)
    report["predictions"] = os.path.basename(args.predictions)
    report["judge"] = getattr(grader, "name", args.judge)
    models = sorted({r.get("model") for r in rows if r.get("model")})
    report["model"] = models[0] if len(models) == 1 else models
    abstention = sorted({bool(r.get("allow_abstention")) for r in rows})
    report["allow_abstention"] = abstention[0] if len(abstention) == 1 else abstention

    print()
    print(vmetrics.format_report(report))

    if report["by_stratum"].get("answer_refusal") and not report["allow_abstention"]:
        print()
        print("note: these predictions used the always-answer prompt, so the "
              "answer_refusal number is not meaningful. Rerun the model with "
              "--allow-abstention to score that stratum.")

    out = args.out or os.path.splitext(args.predictions)[0].replace(
        "predictions", "metrics") + ".json"
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print()
    print("report written to %s" % out)


if __name__ == "__main__":
    main()
