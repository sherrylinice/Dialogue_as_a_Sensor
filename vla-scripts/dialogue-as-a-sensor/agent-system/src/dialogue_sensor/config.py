"""Runtime configuration loaded from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv


def _load_dotenv_once() -> None:
    """Load the project's .env if present.

    We look for `.env` next to the agent-system root, then walk up. This makes
    the CLI work regardless of where it is invoked from.
    """
    here = Path(__file__).resolve()
    for candidate in [here.parent.parent.parent / ".env", *[p / ".env" for p in here.parents]]:
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return
    # Falls through silently if no .env found - env vars may still be set
    # directly in the shell.
    load_dotenv(override=False)


_load_dotenv_once()


Provider = Literal["openai", "google"]


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Config:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))

    visual_inquisitor_provider: Provider = field(
        default_factory=lambda: os.getenv("VISUAL_INQUISITOR_PROVIDER", "openai").lower()  # type: ignore[return-value]
    )
    visual_inquisitor_model: str = field(
        default_factory=lambda: os.getenv("VISUAL_INQUISITOR_MODEL", "gpt-5-mini-2025-08-07")
    )

    simulated_user_provider: Provider = field(
        default_factory=lambda: os.getenv("SIMULATED_USER_PROVIDER", "google").lower()  # type: ignore[return-value]
    )
    simulated_user_model: str = field(
        default_factory=lambda: os.getenv("SIMULATED_USER_MODEL", "gemini-3-flash-preview")
    )

    # Agent 3 (Spatial-to-OSC Grounding / NLU). Defaults to OpenAI's
    # reasoning-class model since Agent 3's job is structured output (JSON of
    # action deltas) rather than open-ended conversation.
    spatial_grounding_provider: Provider = field(
        default_factory=lambda: os.getenv("SPATIAL_GROUNDING_PROVIDER", "openai").lower()  # type: ignore[return-value]
    )
    spatial_grounding_model: str = field(
        default_factory=lambda: os.getenv("SPATIAL_GROUNDING_MODEL", "gpt-5-mini-2025-08-07")
    )
    # When True the graph routes through Agent 3 after the dialogue loop
    # finishes. False short-circuits to END for backward-compat with existing
    # callers that only want the dialogue trace.
    enable_spatial_grounding: bool = field(
        default_factory=lambda: _bool_env("ENABLE_SPATIAL_GROUNDING", True)
    )

    max_turns: int = field(default_factory=lambda: int(os.getenv("MAX_DIALOGUE_TURNS", "1")))
    verbose: bool = field(default_factory=lambda: _bool_env("DIALOGUE_VERBOSE", True))

    def __post_init__(self) -> None:
        if self.visual_inquisitor_provider == self.simulated_user_provider:
            # Project overview.md explicitly requires different VLM families to
            # avoid the inquisitor and the simulated user echoing each other.
            print(
                f"[config] WARNING: visual_inquisitor_provider and simulated_user_provider are both "
                f"'{self.visual_inquisitor_provider}'. The project overview recommends different families."
            )

    def validate(self) -> None:
        """Raise if a required API key for one of the selected providers is missing."""
        providers = {self.visual_inquisitor_provider, self.simulated_user_provider}
        if self.enable_spatial_grounding:
            providers.add(self.spatial_grounding_provider)
        for provider in providers:
            self._require_key(provider)

    def _require_key(self, provider: Provider) -> None:
        if provider == "openai" and not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set but the OpenAI provider is selected. "
                "Add it to your .env file (see .env.example)."
            )
        if provider == "google" and not self.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set but the Google provider is selected. "
                "Add it to your .env file (see .env.example)."
            )


def load_config() -> Config:
    cfg = Config()
    return cfg
