#!/usr/bin/env python3
"""Run a model over one VoxMemBench config and write its answers.

    python run_voxmembench.py --config 32k --model openai:gpt-4o-audio-preview \
        --out predictions_32k.jsonl

Writes one JSON object per item, carrying the fields the scorer needs so that
scoring never has to touch the audio again. Re-running with the same --out
resumes: items already present are skipped.
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voxmembench import data as vdata
from voxmembench import models as vmodels

KEEP = ["item_id", "question_id", "context_length", "family_id", "evidence_type",
        "memory_operation", "subtype", "question_text", "expected_response",
        "gold_json", "answer_type", "answer_normalization", "session_count",
        "evidence_session_count", "samekey_haystack_session_count"]


def already_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["item_id"])
            except (ValueError, KeyError):
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="e.g. 32k, or 32k_paralinguistic_information")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="abstain",
                    help="'abstain', 'openai:<model id>', 'gemini:<model id>', "
                         "or a local model name used with --adapter")
    ap.add_argument("--adapter",
                    help="module name or path to a .py exposing "
                         "run(messages, model_dir=..., max_new_tokens=...); "
                         "makes --model the local model's name")
    ap.add_argument("--model-dir", help="weights directory passed to the adapter")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--staging-dir",
                    help="where to stage a item's clips for the adapter "
                         "(default: the system temp directory)")
    ap.add_argument("--fold-system", dest="fold_system", action="store_true",
                    default=None,
                    help="put the system prompt in the first user message; the "
                         "default follows the model's known behaviour")
    ap.add_argument("--native-system", dest="fold_system", action="store_false",
                    help="force a real system turn")
    ap.add_argument("--repo", default=vdata.DEFAULT_REPO)
    ap.add_argument("--base-url", default=None,
                    help="for an OpenAI-compatible server other than OpenAI's")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--allow-abstention", action="store_true",
                    help="use the system prompt that permits 'insufficient "
                         "evidence'. Required for the answer-refusal stratum "
                         "to mean anything.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stop-after-seconds", type=int, default=0,
                    help="stop cleanly between items once this much wall clock "
                         "has passed, so a scheduler's time limit cannot kill "
                         "the process mid-item. 0 disables.")
    ap.add_argument("--streaming", action="store_true",
                    help="stream the config instead of downloading it whole")
    ap.add_argument("--token", default=None, help="Hugging Face token")
    args = ap.parse_args()

    if args.adapter:
        from voxmembench import local_adapters

        if args.model in local_adapters.UNSUPPORTED:
            raise SystemExit("%s: %s" % (args.model,
                                         local_adapters.UNSUPPORTED[args.model]))
        model = local_adapters.LocalAdapterModel(
            args.adapter, args.model, model_dir=args.model_dir,
            max_new_tokens=args.max_new_tokens, staging_dir=args.staging_dir)
        delivery = model.delivery
        if args.fold_system is not None:
            delivery = "folded" if args.fold_system else "native"
    else:
        kwargs = {}
        if args.model.startswith(("openai:", "gemini:")):
            kwargs = {"base_url": args.base_url, "api_key": args.api_key,
                      "max_tokens": args.max_new_tokens}
        model = vmodels.build(args.model, **kwargs)
        delivery = "folded" if args.fold_system else "native"

    prompt = vdata.system_prompt(args.allow_abstention)
    fold = delivery == "folded"
    print("system prompt delivery: %s" % delivery)

    done = already_done(args.out)
    if done:
        print("resuming: %d items already in %s" % (len(done), args.out))

    items = vdata.load_items(args.config, repo_id=args.repo, limit=args.limit,
                             streaming=args.streaming, token=args.token)

    written = 0
    failed = 0
    stopped_early = False
    started = time.time()
    with open(args.out, "a") as out:
        for item in items:
            if item["item_id"] in done:
                continue
            if args.stop_after_seconds and \
                    time.time() - started >= args.stop_after_seconds:
                stopped_early = True
                break
            record = {key: item[key] for key in KEEP}
            record["model"] = getattr(model, "name", args.model)
            record["allow_abstention"] = args.allow_abstention
            record["system_prompt_delivery"] = delivery
            try:
                messages = vdata.to_messages(item, prompt, fold_system=fold)
                record["n_audio_clips"] = vdata.count_audio(messages)
                record["response"] = model(messages)
            except Exception as exc:                        # noqa: BLE001
                failed += 1
                record["response"] = None
                record["error"] = "%s: %s" % (type(exc).__name__, exc)
                traceback.print_exc(limit=1)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            written += 1
            if written % 20 == 0:
                rate = written / max(time.time() - started, 1e-6)
                print("  %d done, %d failed, %.2f items/s"
                      % (written, failed, rate), flush=True)

    print("wrote %d items (%d failed) to %s" % (written, failed, args.out))
    if stopped_early:
        print("stopped at the --stop-after-seconds mark with items left; "
              "rerun the same command to continue")
    if failed:
        print("failed items kept a null response and an error field; rerun to "
              "retry them after deleting those lines")


if __name__ == "__main__":
    main()
