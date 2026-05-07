"""Thin VLM clients used by Agent 2 and the simulated user.

We deliberately avoid LangChain-style chat wrappers here so that the prompts
are obvious in the source. The two providers are intentionally kept separate
so that Agent 2 and the simulated user are guaranteed to come from different
model families (project-overview.md §"Node Implementation").
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Protocol


# === Common types ============================================================


@dataclass
class ChatMessage:
    """A single text-only chat turn fed to the VLM."""
    role: Literal["user", "assistant"]
    content: str


@dataclass
class ImageInput:
    """An inline image attachment for the VLM call."""
    b64_data: str
    mime_type: str = "image/png"

    @classmethod
    def from_path(cls, path: str | Path) -> "ImageInput":
        p = Path(path)
        with p.open("rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        return cls(b64_data=b64, mime_type=mime)


class VisionLLMClient(Protocol):
    """Protocol implemented by the OpenAI / Google vision wrappers."""
    def chat(
        self,
        system: str,
        messages: List[ChatMessage],
        image: Optional[ImageInput] = None,
    ) -> str: ...

    @property
    def label(self) -> str: ...


# === OpenAI ==================================================================


class OpenAIVisionClient:
    """Calls a GPT-* multimodal chat model.

    Uses the (responses-compatible) chat-completions API. The image is
    attached to the most recent user message as a data URL.
    """

    def __init__(self, api_key: str, model: str):
        from openai import OpenAI

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required to construct OpenAIVisionClient.")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    @property
    def label(self) -> str:
        return f"openai:{self._model}"

    def chat(
        self,
        system: str,
        messages: List[ChatMessage],
        image: Optional[ImageInput] = None,
    ) -> str:
        api_messages: list[dict] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        for i, m in enumerate(messages):
            attach_image = image is not None and i == len(messages) - 1 and m.role == "user"
            if attach_image:
                api_messages.append(
                    {
                        "role": m.role,
                        "content": [
                            {"type": "text", "text": m.content},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image.mime_type};base64,{image.b64_data}"
                                },
                            },
                        ],
                    }
                )
            else:
                api_messages.append({"role": m.role, "content": m.content})

        # GPT-5 series uses ``max_completion_tokens`` instead of ``max_tokens`` and
        # the reasoning models reject ``temperature`` overrides; we let the model
        # default both. The wrapper still works for older GPT-4o style models.
        response = self._client.chat.completions.create(
            model=self._model,
            messages=api_messages,
        )
        text = response.choices[0].message.content
        if text is None:
            raise RuntimeError(
                f"OpenAI returned no text content for model {self._model}. "
                f"Raw response: {response}"
            )
        return text.strip()


# === Google Gemini ===========================================================


class GoogleVisionClient:
    """Calls a Gemini multimodal model via the unified `google-genai` SDK."""

    def __init__(self, api_key: str, model: str):
        try:
            from google import genai  # type: ignore[import-untyped]
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "google-genai is not installed. Run `pip install google-genai` "
                "(see agent-system/requirements.txt)."
            ) from e

        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required to construct GoogleVisionClient.")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    @property
    def label(self) -> str:
        return f"google:{self._model}"

    def chat(
        self,
        system: str,
        messages: List[ChatMessage],
        image: Optional[ImageInput] = None,
    ) -> str:
        # `google-genai` accepts a list of "contents" where each entry has a
        # role + parts. We collapse the system prompt into the first user
        # message because Gemini's text-completion endpoint expects no
        # standalone system role for some checkpoints; the new SDK supports
        # `system_instruction` separately, which we use when available.
        from google.genai import types  # type: ignore[import-untyped]

        contents: list[types.Content] = []
        for i, m in enumerate(messages):
            parts: list[types.Part] = [types.Part.from_text(text=m.content)]
            attach_image = image is not None and i == len(messages) - 1 and m.role == "user"
            if attach_image:
                parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(image.b64_data),
                        mime_type=image.mime_type,
                    )
                )
            # Gemini's role for assistant turns is ``model``.
            role = "user" if m.role == "user" else "model"
            contents.append(types.Content(role=role, parts=parts))

        config = None
        if system:
            config = types.GenerateContentConfig(system_instruction=system)

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError(
                f"Gemini returned no text for model {self._model}. Raw response: {response}"
            )
        return text.strip()


# === Factory =================================================================


def build_client(provider: str, api_key: str, model: str) -> VisionLLMClient:
    """Construct the right client for the given provider name."""
    provider = provider.lower()
    if provider == "openai":
        return OpenAIVisionClient(api_key=api_key, model=model)
    if provider == "google":
        return GoogleVisionClient(api_key=api_key, model=model)
    raise ValueError(
        f"Unknown LLM provider '{provider}'. Supported: 'openai', 'google'."
    )
