"""Grading a response against the gold answer.

Answers are open ended, so equivalence is settled by an LLM judge. The judge
reads only the question, the gold and the response -- never the audio and never
the history -- so it cannot re-answer the item itself. The two prompts under
prompts/ are the contract; they are the ones the reference numbers were
produced with.
"""
import json
import re

from .data import read_prompt

ANSWERABLE_PROMPT = "answerable_judge_prompt.txt"
REFUSAL_PROMPT = "ar_judge_prompt.txt"

_VERDICT = re.compile(r'"verdict"\s*:\s*"(correct|incorrect)"', re.IGNORECASE)


def parse_verdict(text):
    """Pull the verdict out of a judge reply, tolerating fenced JSON."""
    if not text:
        return None
    match = _VERDICT.search(text)
    if match:
        return match.group(1).lower() == "correct"
    stripped = text.strip().strip("`")
    if stripped.startswith("json"):
        stripped = stripped[4:]
    try:
        verdict = json.loads(stripped).get("verdict")
    except ValueError:
        return None
    if isinstance(verdict, str):
        return verdict.lower() == "correct"
    return None


def build_request(item, response):
    """The text the judge is shown for one item."""
    if item["expected_response"] == "insufficient_evidence":
        prompt = read_prompt(REFUSAL_PROMPT)
        body = "\n".join([
            "# Question",
            item["question_text"] or "",
            "",
            "# Model response",
            response or "",
        ])
    else:
        prompt = read_prompt(ANSWERABLE_PROMPT)
        body = "\n".join([
            "# Question",
            item["question_text"] or "",
            "",
            "# Reference answer",
            item["gold_json"] or "",
            "",
            "# Expected answer type",
            "%s (%s)" % (item["answer_type"], item["answer_normalization"]),
            "",
            "# Model response",
            response or "",
        ])
    return prompt, body


class OpenAIJudge:
    """An OpenAI-compatible chat endpoint used as the judge. Text only."""

    def __init__(self, model, base_url=None, api_key=None, max_tokens=200):
        from openai import OpenAI

        self.name = model
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def __call__(self, item, response):
        prompt, body = build_request(item, response)
        reply = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt},
                      {"role": "user", "content": body}],
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        text = (reply.choices[0].message.content or "").strip()
        return parse_verdict(text), text


class ExactMatchJudge:
    """A judge-free fallback for the closed-answer types.

    Only `categorical`, `yes_no` and `number` have answers narrow enough to
    compare without a judge, and even there it is stricter than the real
    metric: a correct answer phrased differently is marked wrong. Use it for
    a cheap smoke test, not for a number you intend to report.
    """

    name = "exact-match"

    def __call__(self, item, response):
        text = (response or "").strip().lower()
        if item["expected_response"] == "insufficient_evidence":
            return ("insufficient" in text or "not enough" in text
                    or "cannot be determined" in text), text
        if item["answer_type"] not in ("categorical", "yes_no", "number"):
            return None, text
        try:
            gold = json.loads(item["gold_json"])
        except (TypeError, ValueError):
            return None, text
        if isinstance(gold, bool):
            gold_text = "yes" if gold else "no"
        else:
            gold_text = str(gold).lower()
        return gold_text in text, text


def build(spec, **kwargs):
    if spec == "exact-match":
        return ExactMatchJudge()
    if spec.startswith("openai:"):
        return OpenAIJudge(spec[len("openai:"):], **kwargs)
    raise ValueError("unknown judge %r; pass 'exact-match' or 'openai:<model>'"
                     % spec)
