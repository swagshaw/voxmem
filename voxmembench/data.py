"""Loading VoxMemBench items and turning one into a model prompt.

The dataset stores each session's timestamp once, on the session, because the
same session is reused at different points in different histories. The prompt
the model sees puts that timestamp back where it belongs: as text on the first
user turn of the session, ahead of that turn's audio.
"""
import os

DEFAULT_REPO = "AudioMemory/voxmembench"

CONTEXT_LENGTHS = ["8k", "16k", "32k", "64k"]
EVIDENCE_TYPES = [
    "speech_semantics",
    "speaker_information",
    "paralinguistic_information",
    "environmental_sound",
]
MEMORY_OPERATIONS = [
    "information_extraction",
    "multi_session_reasoning",
    "temporal_evolution_tracking",
    "answer_refusal",
]

PROMPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "prompts")
SESSION_TIMESTAMP_PREFIX = "Session timestamp: "


def read_prompt(name):
    with open(os.path.join(PROMPTS, name)) as fh:
        return fh.read().strip()


def system_prompt(allow_abstention):
    """The two candidate prompts.

    Answer refusal is only measurable under the abstaining prompt; under the
    other one the model is told to always answer, so the refusal stratum reads
    about zero by construction.
    """
    return read_prompt("candidate_system_prompt.txt" if allow_abstention
                       else "candidate_system_prompt.no_abstain.txt")


def audio_decoding_available():
    """Whether `datasets` can actually decode audio in this environment.

    The decoder is an optional extra that needs FFmpeg. Rather than failing on
    the first clip, fall back to the raw WAV bytes, which every adapter here
    accepts just as well.
    """
    try:
        from datasets.features._torchcodec import AudioDecoder  # noqa: F401
    except Exception:                                            # noqa: BLE001
        return False
    return True


def load_items(config, repo_id=DEFAULT_REPO, split="train", limit=None,
               streaming=False, token=None, decode_audio=None):
    """Iterate one config, e.g. "32k" or "32k_paralinguistic_information".

    Yields plain dicts. With decode_audio left at None the clips are decoded
    when the environment can and handed over as WAV bytes when it cannot; the
    undecoded path reads the Arrow table directly, which skips the decoder
    rather than casting the dataset.
    """
    from datasets import load_dataset

    if decode_audio is None:
        decode_audio = audio_decoding_available()

    dataset = load_dataset(repo_id, config, split=split, streaming=streaming,
                           token=token)
    if limit is not None:
        dataset = (dataset.take(limit) if streaming
                   else dataset.select(range(min(limit, dataset.num_rows))))

    if decode_audio:
        return dataset

    def rows():
        if streaming:
            for batch in dataset.with_format("arrow").iter(batch_size=1):
                for row in batch.to_pylist():
                    yield row
            return
        table = dataset.with_format("arrow")[:]
        for row in table.to_pylist():
            yield row

    return rows()


def _audio_part(audio):
    """Pass the clip through in whatever form `datasets` handed it over.

    With the audio extra installed the dataset decodes to an array; without it
    the raw WAV bytes come through instead. Both are usable, so both are kept
    rather than forcing one and failing on the other.
    """
    if audio is None:
        return None
    part = {"type": "audio"}
    if audio.get("array") is not None:
        part["array"] = audio["array"]
        part["sampling_rate"] = audio.get("sampling_rate")
    if audio.get("bytes") is not None:
        part["bytes"] = audio["bytes"]
    part["path"] = audio.get("path")
    if "array" not in part and "bytes" not in part:
        raise ValueError("clip %r carries neither an array nor bytes"
                         % part["path"])
    return part


def to_messages(item, prompt):
    """Build the provider-neutral prompt for one item.

    Returns a list of {"role", "content"} where content is a list of parts,
    each either {"type": "text", "text": ...} or {"type": "audio", ...}.
    The model is never given a transcript of the user's speech: `question_text`
    and the session transcripts are for analysis, not for the prompt.
    """
    messages = [{"role": "system",
                 "content": [{"type": "text", "text": prompt}]}]

    for session in item["sessions"]:
        first_user_turn = True
        for turn in session["turns"]:
            if turn["role"] == "assistant":
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": turn["text"] or ""}],
                })
                continue
            content = []
            if first_user_turn:
                content.append({
                    "type": "text",
                    "text": SESSION_TIMESTAMP_PREFIX + (session["timestamp"] or ""),
                })
                first_user_turn = False
            part = _audio_part(turn["audio"])
            if part is not None:
                content.append(part)
            messages.append({"role": "user", "content": content})

    query = [{"type": "text",
              "text": SESSION_TIMESTAMP_PREFIX + (item["query_timestamp"] or "")}]
    question = _audio_part(item["question_audio"])
    if question is None:
        raise ValueError("item %s has no spoken question" % item["item_id"])
    query.append(question)
    messages.append({"role": "user", "content": query})
    return messages


def count_audio(messages):
    return sum(1 for m in messages for p in m["content"] if p["type"] == "audio")
