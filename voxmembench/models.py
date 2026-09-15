"""Model adapters.

A model is any callable that takes the provider-neutral messages from
`data.to_messages` and returns the model's answer as a string. Two adapters
ship here; write your own for anything else and pass it to `run_voxmembench`.
"""
import base64
import io
import os


def to_wav_bytes(part):
    """A prompt part's audio as WAV bytes, whichever form it arrived in."""
    if part.get("bytes") is not None:
        return part["bytes"]
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, part["array"], part["sampling_rate"], format="WAV",
             subtype="PCM_16")
    return buffer.getvalue()


class AbstainModel:
    """Always says the evidence is insufficient.

    Useful for two things: checking the pipeline end to end without paying for
    a real model, and establishing the floor on the refusal stratum, which this
    scores 1.0 on and every answerable item 0.0 on.
    """

    name = "abstain-baseline"

    def __call__(self, messages):
        return "insufficient evidence"


class OpenAIAudioModel:
    """Any OpenAI-compatible chat endpoint that accepts input_audio parts.

    Point `base_url` at another server to use a local or third-party model that
    speaks the same protocol.
    """

    def __init__(self, model, base_url=None, api_key=None, max_tokens=256,
                 temperature=0.0):
        from openai import OpenAI

        self.name = model
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _convert(self, messages):
        converted = []
        for message in messages:
            parts = []
            for part in message["content"]:
                if part["type"] == "text":
                    parts.append({"type": "text", "text": part["text"]})
                else:
                    payload = base64.b64encode(to_wav_bytes(part)).decode("ascii")
                    parts.append({
                        "type": "input_audio",
                        "input_audio": {"data": payload, "format": "wav"},
                    })
            converted.append({"role": message["role"], "content": parts})
        return converted

    def __call__(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._convert(messages),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return (response.choices[0].message.content or "").strip()


class GeminiAudioModel:
    """Google Gemini, which takes audio natively as inline parts.

    Gemini has no system role in the chat turns: the prompt goes in
    `system_instruction`, so no folding is needed.
    """

    def __init__(self, model, api_key=None, max_tokens=256, temperature=0.0):
        from google import genai

        self.name = model
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))

    def _convert(self, messages):
        from google.genai import types

        system = None
        contents = []
        for message in messages:
            if message["role"] == "system":
                system = " ".join(p["text"] for p in message["content"]
                                  if p["type"] == "text")
                continue
            parts = []
            for part in message["content"]:
                if part["type"] == "text":
                    parts.append(types.Part.from_text(text=part["text"]))
                else:
                    parts.append(types.Part.from_bytes(
                        data=to_wav_bytes(part), mime_type="audio/wav"))
            # Gemini names the assistant role "model"
            role = "model" if message["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=parts))
        return system, contents

    def __call__(self, messages):
        from google.genai import types

        system, contents = self._convert(messages)
        config = types.GenerateContentConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            system_instruction=system,
        )
        response = self.client.models.generate_content(
            model=self.model, contents=contents, config=config)
        return (response.text or "").strip()


def build(spec, **kwargs):
    """Resolve a --model string.

    "abstain"            the baseline above
    "openai:<model id>"  an OpenAI-compatible endpoint
    """
    if spec == "abstain":
        return AbstainModel()
    if spec.startswith("openai:"):
        return OpenAIAudioModel(spec[len("openai:"):], **kwargs)
    if spec.startswith("gemini:"):
        kwargs.pop("base_url", None)
        return GeminiAudioModel(spec[len("gemini:"):], **kwargs)
    raise ValueError(
        "unknown model %r; pass 'abstain', 'openai:<model>', 'gemini:<model>', "
        "--adapter for a local model, or import run_voxmembench and hand it "
        "your own callable" % spec)
