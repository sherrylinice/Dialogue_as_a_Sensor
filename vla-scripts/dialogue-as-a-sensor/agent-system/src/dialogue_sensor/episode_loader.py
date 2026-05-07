"""Load an episode (image + metadata) from disk into a ``DialogueState``.

By default we render the planned-trajectory overlay onto the agentview image
before sending it to the agents — A1/A2/A3 axis frames in red/green/blue
matching the data-generation visualiser. The overlay is computed from the
``waypoints`` + camera matrices in the episode's ``metadata.json``, with
fall-backs to the constants captured from the PickPlaceClutter agentview
camera (which is rigidly mounted, so the matrices are stable across resets).
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .overlays import (
    DEFAULT_K,
    DEFAULT_T_WC,
    encode_png,
    project_waypoints_summary,
    render_overlay,
)
from .state import DialogueState, TriggerReason


# Names we recognise for the "stuck" image, in order of preference. We fall
# through to the initial scene image so the loader works on episodes produced
# by the legacy ``generate_vla_dataset.py`` (which only saved ``image_rgb.png``)
# as well as on the new ``generate_stuck_dataset.py`` outputs.
STUCK_IMAGE_CANDIDATES = ["image_stuck.png", "image_rgb.png", "image_initial.png"]


def _find_image(episode_dir: Path, override: Optional[str] = None) -> Path:
    if override:
        cand = (episode_dir / override) if not Path(override).is_absolute() else Path(override)
        if cand.is_file():
            return cand
        raise FileNotFoundError(f"Specified image not found: {cand}")

    for name in STUCK_IMAGE_CANDIDATES:
        cand = episode_dir / name
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"No image found in {episode_dir}. Looked for: {STUCK_IMAGE_CANDIDATES}"
    )


def _resolve_camera_matrices(
    metadata: dict,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], bool]:
    """Return ``(K, T_wc, (H, W), used_default)``.

    Episodes from the new generator store both matrices under
    ``camera_intrinsics`` / ``camera_extrinsics_wc``. Legacy episodes
    don't, in which case we fall back to the agentview defaults.
    """
    K_meta = metadata.get("camera_intrinsics")
    T_meta = metadata.get("camera_extrinsics_wc")
    size_meta = metadata.get("camera_image_size") or [256, 256]
    used_default = False
    if K_meta is not None and T_meta is not None:
        K = np.asarray(K_meta, dtype=np.float64)
        T_wc = np.asarray(T_meta, dtype=np.float64)
    else:
        K, T_wc = DEFAULT_K.copy(), DEFAULT_T_WC.copy()
        used_default = True
    H, W = int(size_meta[0]), int(size_meta[1])
    return K, T_wc, (H, W), used_default


def load_episode_state(
    episode_dir: str | Path,
    *,
    image_filename: Optional[str] = None,
    trigger_reason: Optional[TriggerReason] = None,
    max_turns: int = 1,
    overlay: bool = True,
    save_overlay_to: Optional[str | Path] = None,
) -> DialogueState:
    """Read an episode directory and produce an initial ``DialogueState``.

    Parameters
    ----------
    episode_dir : path
        Directory containing ``metadata.json`` and at least one of
        ``image_stuck.png`` / ``image_rgb.png``.
    image_filename : str, optional
        Override which file to load as the stuck image. Useful for picking
        a specific phase frame (e.g. ``image_phase_grasp_descend.png``).
    trigger_reason : str, optional
        Override the trigger reason. Defaults to the metadata's
        ``trigger_reason`` if present, else ``"unknown"``.
    max_turns : int
        Maximum dialogue rounds to allow.
    overlay : bool
        When True (default) render the planned-trajectory overlay onto the
        loaded image and feed the annotated PNG to the agents. When False
        the raw image is sent.
    save_overlay_to : path, optional
        If given, also save the annotated PNG to this path (for inspection).
    """
    ep = Path(episode_dir)
    if not ep.is_dir():
        raise NotADirectoryError(f"Episode dir does not exist: {ep}")

    meta_path = ep / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata.json missing under {ep}")
    metadata = json.loads(meta_path.read_text())

    img_path = _find_image(ep, override=image_filename)
    raw_bytes = img_path.read_bytes()

    waypoints = metadata.get("waypoints") or {}
    K, T_wc, (H, W), used_default_matrices = _resolve_camera_matrices(metadata)

    overlay_used = False
    overlay_pixel_summary = []
    if overlay and waypoints:
        annotated = render_overlay(raw_bytes, waypoints, K=K, T_wc=T_wc)
        # Resize back to the source size so we don't change the aspect
        # ratio (PIL's render_overlay preserves it, but defensive resize).
        if annotated.size != (W, H):
            # Reproject overlay rather than naively resizing - we want sharp
            # axis lines. So instead, rerender at the actual loaded size.
            actual_size = Image.open(io.BytesIO(raw_bytes)).size
            annotated = render_overlay(raw_bytes, waypoints, K=K, T_wc=T_wc)
            actual_w, actual_h = actual_size
            H, W = actual_h, actual_w
        png_bytes = encode_png(annotated)
        overlay_used = True
        overlay_pixel_summary = project_waypoints_summary(
            waypoints, K=K, T_wc=T_wc, image_size=(H, W)
        )
        if save_overlay_to is not None:
            out = Path(save_overlay_to)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(png_bytes)
    else:
        png_bytes = raw_bytes

    img_b64 = base64.b64encode(png_bytes).decode("utf-8")

    instruction: str = metadata.get("instruction", "")
    resolved_trigger: TriggerReason
    if trigger_reason is not None:
        resolved_trigger = trigger_reason
    else:
        resolved_trigger = metadata.get("trigger_reason") or "unknown"
        # Episodes from the new generator can carry "none" when the trial
        # actually succeeded; the dialogue agents still want a real trigger
        # tag so they know how to frame their question/answer.
        if resolved_trigger == "none":
            resolved_trigger = "semantic_ambiguity"

    state: DialogueState = {
        "image_path": str(img_path),
        "image_b64": img_b64,
        "image_mime": "image/png",
        "task_description": instruction,
        "trigger_reason": resolved_trigger,
        "kinematic_state": metadata.get("kinematic_state") or {},
        "osc_delta": None,
        "dialogue_history": [],
        "clarifying_question": None,
        "clarified_instruction": None,
        "turn_count": 0,
        "max_turns": max_turns,
        "done": False,
        "metadata": {
            **metadata,
            # Convenience fields the agents inspect at prompt-build time.
            "_overlay_used": overlay_used,
            "_overlay_used_default_camera": used_default_matrices,
            "_overlay_waypoint_pixels": overlay_pixel_summary,
        },
    }
    return state
