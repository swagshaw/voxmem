"""Aggregating graded predictions.

The two strata are reported separately and never averaged together: an
answerable item asks the model to produce the answer, a refusal item asks it to
decline, and a single number over both rewards whichever behaviour a model
happens to default to.
"""
import collections

STRATA = {"answer": "answerable", "insufficient_evidence": "answer_refusal"}


def _rate(rows):
    graded = [r for r in rows if r.get("verdict") is not None]
    if not graded:
        return {"n": len(rows), "graded": 0, "accuracy": None}
    correct = sum(1 for r in graded if r["verdict"])
    return {
        "n": len(rows),
        "graded": len(graded),
        "accuracy": round(correct / len(graded), 4),
    }


def aggregate(rows):
    """Overall, per stratum, per context length, per cell."""
    by_stratum = collections.defaultdict(list)
    for row in rows:
        by_stratum[STRATA.get(row["expected_response"], "unknown")].append(row)

    report = {
        "n_items": len(rows),
        "ungraded": sum(1 for r in rows if r.get("verdict") is None),
        "by_stratum": {k: _rate(v) for k, v in sorted(by_stratum.items())},
    }

    answerable = by_stratum.get("answerable", [])
    refusal = by_stratum.get("answer_refusal", [])

    for name, subset in (("answerable", answerable), ("answer_refusal", refusal)):
        by_length = collections.defaultdict(list)
        for row in subset:
            by_length[row["context_length"]].append(row)
        report.setdefault("by_context_length", {})[name] = {
            k: _rate(v) for k, v in sorted(by_length.items())
        }

    by_cell = collections.defaultdict(list)
    for row in rows:
        by_cell[(row["evidence_type"], row["memory_operation"])].append(row)
    report["by_cell"] = {
        "%s|%s" % key: _rate(value) for key, value in sorted(by_cell.items())
    }

    by_subtype = collections.defaultdict(list)
    for row in answerable:
        by_subtype[row.get("subtype") or "unknown"].append(row)
    report["by_subtype_answerable"] = {
        k: _rate(v) for k, v in sorted(by_subtype.items())
    }
    return report


def format_report(report):
    lines = []
    lines.append("items %d   ungraded %d" % (report["n_items"], report["ungraded"]))
    lines.append("")
    lines.append("stratum                         n   graded   accuracy")
    for name, stats in report["by_stratum"].items():
        lines.append("  %-28s %5d %8d %10s"
                     % (name, stats["n"], stats["graded"],
                        "-" if stats["accuracy"] is None else "%.4f" % stats["accuracy"]))
    for stratum, lengths in report.get("by_context_length", {}).items():
        if not lengths:
            continue
        lines.append("")
        lines.append("%s by context length" % stratum)
        for length, stats in lengths.items():
            lines.append("  %-28s %5d %8d %10s"
                         % (length, stats["n"], stats["graded"],
                            "-" if stats["accuracy"] is None else "%.4f" % stats["accuracy"]))
    lines.append("")
    lines.append("evidence type x memory operation")
    for cell, stats in report["by_cell"].items():
        lines.append("  %-44s %5d %8d %10s"
                     % (cell, stats["n"], stats["graded"],
                        "-" if stats["accuracy"] is None else "%.4f" % stats["accuracy"]))
    return "\n".join(lines)
