"""Driving a locally hosted audio LLM.

A local adapter is one function:

    run(messages, model_dir=None, max_new_tokens=200) -> dict
        {"ok": bool, "response_text": str, "error": str, ...}

taking messages in the shape below, where every clip is a path on disk:

    {"role": "system",    "content": "<text>"}
    {"role": "user",      "content": [{"type": "audio", "audio": "<path>"},
                                      {"type": "text",  "text": "<text>"}]}
    {"role": "assistant", "content": "<text>"}

Point `--adapter` at any module exposing that function. Each model usually
needs its own environment, so the runner drives one model per process rather
than importing several into one.

## The system prompt is not safely native

Many chat templates convert messages to their native format with a line like
`if msg["role"] == "system": continue`. The prompt is then silently gone, and
with it the abstention contract that the 520 answer-refusal items are scored
against -- those items become unanswerable in a way that looks like model
failure rather than a harness bug.

`NATIVE_SYSTEM` lists the models whose adapters were checked to keep a real
system turn. For anything else, fold the prompt into the first user message
(`--fold-system`, the default for unlisted models) and note which happened:
every prediction records `system_prompt_delivery`, because a cross-model table
that mixes the two silently is not comparable.
"""
import io
import os
import shutil
import sys
import tempfile

# Adapters observed to render a real system turn. Anything not listed here is
# assumed to drop it, and gets the folded prompt instead.
NATIVE_SYSTEM = {
    "afnext", "audex", "baichuan", "firered", "phi4",
    "qwen2", "qwen25omni", "qwen3omni", "ultravox",
}

# Known not to accept more than one audio clip in a prompt, which this
# benchmark always does.
UNSUPPORTED = {
    "glm4": "multi-audio input produces NaN generation",
    "minicpm": "multi-audio KV cache mismatch",
}


def delivery_for(model_name):
    """Whether this model can be given a real system turn."""
    return "native" if model_name in NATIVE_SYSTEM else "folded"


def load_adapter(spec):
    """Import an adapter by module name or by path to a .py file."""
    if spec.endswith(".py"):
        import importlib.util

        path = os.path.abspath(spec)
        name = os.path.splitext(os.path.basename(path))[0]
        directory = os.path.dirname(path)
        if directory not in sys.path:
            sys.path.insert(0, directory)
        spec_obj = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
    else:
        import importlib

        module = importlib.import_module(spec)

    for attribute in ("run", "run_messages", "generate"):
        fn = getattr(module, attribute, None)
        if callable(fn):
            return fn
    raise AttributeError(
        "%s exposes no run(messages, ...) entry point" % spec)


def _write_clip(part, directory, index):
    """Materialise one clip to a file, since adapters read audio from disk."""
    path = os.path.join(directory, "%04d_%s" % (index, part.get("path") or "clip.wav"))
    if part.get("bytes") is not None:
        with open(path, "wb") as fh:
            fh.write(part["bytes"])
        return path
    import soundfile as sf

    sf.write(path, part["array"], part["sampling_rate"], format="WAV",
             subtype="PCM_16")
    return path


class LocalAdapterModel:
    """Run one item through a local adapter, clips staged on disk."""

    def __init__(self, adapter, model_name, model_dir=None, max_new_tokens=256,
                 staging_dir=None, keep_clips=False):
        self.run_fn = load_adapter(adapter)
        self.name = model_name
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self.staging_dir = staging_dir
        self.keep_clips = keep_clips
        self.delivery = delivery_for(model_name)
        self.last = None

    def to_adapter_messages(self, messages, directory):
        out = []
        index = 0
        for message in messages:
            if message["role"] == "user":
                content = []
                for part in message["content"]:
                    if part["type"] == "text":
                        content.append({"type": "text", "text": part["text"]})
                    else:
                        index += 1
                        content.append({
                            "type": "audio",
                            "audio": _write_clip(part, directory, index),
                        })
                out.append({"role": "user", "content": content})
            else:
                # system and assistant turns are plain strings for adapters
                text = " ".join(p["text"] for p in message["content"]
                                if p["type"] == "text")
                out.append({"role": message["role"], "content": text})
        return out

    def __call__(self, messages):
        directory = tempfile.mkdtemp(prefix="voxmem_", dir=self.staging_dir)
        try:
            adapter_messages = self.to_adapter_messages(messages, directory)
            kwargs = {"max_new_tokens": self.max_new_tokens}
            if self.model_dir:
                kwargs["model_dir"] = self.model_dir
            result = self.run_fn(adapter_messages, **kwargs)
            self.last = result
            if isinstance(result, str):
                return result
            if not result.get("ok", True):
                raise RuntimeError(result.get("error") or "adapter reported failure")
            return (result.get("response_text") or "").strip()
        finally:
            if not self.keep_clips:
                shutil.rmtree(directory, ignore_errors=True)
