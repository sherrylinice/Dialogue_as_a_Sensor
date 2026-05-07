"""Project planned waypoints into pixel space and draw the action plan onto
the agentview image.

Mirrors the conventions of
``data-generation/visualize_dataset_and_affordances.py`` so the dialogue
agents see the same coordinate-frame visualization the rest of the project
uses for inspection.

Conventions (matching ``action_correction_prompt.py`` and the saved
``vis_episode_*.png`` files):

* Each waypoint pose is 7-DoF: ``[x, y, z, roll, pitch, yaw, gripper]`` in
  the world frame, with rotations expressed as ``scipy`` ``'xyz'`` Euler
  angles in radians.
* At each waypoint we draw a 5 cm coordinate frame:

    - **X-axis (RED)**: gripper "pinch" direction. Forward (+) moves away
      from the camera.
    - **Y-axis (GREEN)**: perpendicular to the pinch direction. Right (+)
      on screen, left (−).
    - **Z-axis (BLUE)**: vertical, up (+).

* Only ``A1_pregrasp``, ``A2_grasp``, ``A3_release`` are drawn (A4_home is
  not informative for the dialogue).
"""

from __future__ import annotations

import io
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Default camera intrinsics + extrinsics for the PickPlaceClutter agentview
# camera at 256x256. These were captured once from a fresh ``env.reset()`` and
# are constant across episodes (the camera is rigidly mounted on the arena),
# so we use them as a fallback whenever the episode metadata does not include
# camera matrices (the legacy ``my_data/`` episodes do not).
DEFAULT_K = np.array(
    [
        [309.01933598375615, 0.0, 128.0],
        [0.0, 309.01933598375615, 128.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

DEFAULT_T_WC = np.array(
    [
        [0.0, 0.706147844353306, -0.7080644193257978, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -0.7080644193257978, -0.706147844353306, 1.75],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

DEFAULT_IMAGE_SIZE = (256, 256)  # (height, width)

# Pillow uses RGB tuples; the colours below match the BGR ones used by the
# data-generation visualiser when re-interpreted in RGB.
AXIS_COLOURS_RGB: Dict[str, Tuple[int, int, int]] = {
    "x": (255, 0, 0),    # Red
    "y": (0, 255, 0),    # Green
    "z": (0, 0, 255),    # Blue
}
LABEL_COLOUR_RGB: Tuple[int, int, int] = (255, 255, 255)
LABEL_OUTLINE_RGB: Tuple[int, int, int] = (0, 0, 0)

DEFAULT_AXIS_LENGTH_M: float = 0.05  # 5 cm long axes
DEFAULT_LINE_WIDTH: int = 2

DRAW_WAYPOINTS = ("A1_pregrasp", "A2_grasp", "A3_release")
WAYPOINT_LABEL = {
    "A1_pregrasp": "A1",
    "A2_grasp": "A2",
    "A3_release": "A3",
}


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


def _euler_xyz_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """SciPy-equivalent ``R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()``.

    Implemented locally so that the agent-system has no dependency on SciPy.
    Convention: extrinsic rotations applied in the order X, then Y, then Z, so
    the resulting matrix is ``Rz @ Ry @ Rx`` (which is what
    ``scipy.spatial.transform.Rotation.from_euler('xyz', ...)`` produces).
    """
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def project_world_to_pixel(
    p_world: np.ndarray,
    T_wc: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> Optional[Tuple[int, int]]:
    """Project a 3D world point into 2D pixel coordinates.

    Returns ``None`` if the point is behind the camera, projects outside the
    image, or any matrix is non-finite. Mirrors the function in
    ``visualize_dataset_and_affordances.py``.
    """
    if not np.isfinite(p_world).all() or not np.isfinite(T_wc).all() or not np.isfinite(K).all():
        return None
    try:
        T_cw = np.linalg.inv(T_wc)
    except np.linalg.LinAlgError:
        return None

    p_h = np.array([p_world[0], p_world[1], p_world[2], 1.0], dtype=np.float64)
    p_cam = (T_cw @ p_h)[:3]
    if not np.isfinite(p_cam).all():
        return None
    z_c = p_cam[2]
    if z_c <= 1e-6:
        return None
    p_img = K @ p_cam
    u = p_img[0] / p_img[2]
    v = p_img[1] / p_img[2]
    if not (0 <= u < width and 0 <= v < height):
        return None
    return int(round(u)), int(round(v))


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str) -> None:
    """Render a small label with a 1-pixel black outline for legibility."""
    x, y = xy[0] + 4, xy[1] + 2
    # 1-pixel outline so the label is readable on bright backgrounds.
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + dx, y + dy), text, fill=LABEL_OUTLINE_RGB)
    draw.text((x, y), text, fill=LABEL_COLOUR_RGB)


def render_overlay(
    image: Image.Image | bytes | np.ndarray,
    waypoints: Dict[str, Sequence[float]],
    *,
    K: Optional[np.ndarray] = None,
    T_wc: Optional[np.ndarray] = None,
    axis_length_m: float = DEFAULT_AXIS_LENGTH_M,
    line_width: int = DEFAULT_LINE_WIDTH,
    draw_keys: Iterable[str] = DRAW_WAYPOINTS,
    instruction_text: Optional[str] = None,
) -> Image.Image:
    """Return a copy of ``image`` with the planned waypoints drawn on top.

    Parameters
    ----------
    image : PIL.Image, raw bytes (PNG/JPEG), or HxWx3 ndarray.
    waypoints : mapping waypoint-name -> 7-DoF pose
        ``[x, y, z, roll, pitch, yaw, gripper]``. Only keys in ``draw_keys``
        are rendered.
    K, T_wc : optional camera matrices. Default to the constants captured
        from the PickPlaceClutter agentview camera (constant across resets).
    axis_length_m : world-frame axis length in meters (default 0.05 = 5 cm).
    line_width : axis-line thickness in pixels.
    instruction_text : if given, render it as a black footer bar with white
        text at the bottom of the image (matches the existing
        ``vis_episode_*.png`` look).
    """
    pil = _coerce_to_pil_rgb(image)
    width, height = pil.size

    K_eff = np.asarray(K if K is not None else DEFAULT_K, dtype=np.float64)
    T_wc_eff = np.asarray(T_wc if T_wc is not None else DEFAULT_T_WC, dtype=np.float64)

    canvas = pil.copy()
    draw = ImageDraw.Draw(canvas)

    for key in draw_keys:
        pose = waypoints.get(key)
        if pose is None or len(pose) < 6:
            continue
        origin = np.asarray(pose[:3], dtype=np.float64)
        roll, pitch, yaw = float(pose[3]), float(pose[4]), float(pose[5])
        Rmat = _euler_xyz_to_matrix(roll, pitch, yaw)
        end_x = origin + Rmat @ np.array([axis_length_m, 0.0, 0.0])
        end_y = origin + Rmat @ np.array([0.0, axis_length_m, 0.0])
        end_z = origin + Rmat @ np.array([0.0, 0.0, axis_length_m])

        pix_origin = project_world_to_pixel(origin, T_wc_eff, K_eff, width, height)
        pix_x = project_world_to_pixel(end_x, T_wc_eff, K_eff, width, height)
        pix_y = project_world_to_pixel(end_y, T_wc_eff, K_eff, width, height)
        pix_z = project_world_to_pixel(end_z, T_wc_eff, K_eff, width, height)

        # Draw whichever of the three axes are entirely in-frame; the
        # data-generation visualiser only draws when ALL four points are
        # in-frame, but for the dialogue agents partial overlays are still
        # informative — for instance, A1 sometimes sits just above the image.
        if pix_origin is not None:
            for endpoint, colour in (
                (pix_x, AXIS_COLOURS_RGB["x"]),
                (pix_y, AXIS_COLOURS_RGB["y"]),
                (pix_z, AXIS_COLOURS_RGB["z"]),
            ):
                if endpoint is not None:
                    draw.line([pix_origin, endpoint], fill=colour, width=line_width)
            _draw_label(draw, pix_origin, WAYPOINT_LABEL.get(key, key[:2]))

    if instruction_text:
        _draw_instruction_bar(canvas, instruction_text)

    return canvas


def _coerce_to_pil_rgb(image: Image.Image | bytes | np.ndarray) -> Image.Image:
    """Accept any of the three input shapes and return an RGB PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            return Image.fromarray(image, mode="L").convert("RGB")
        if image.shape[2] == 4:
            return Image.fromarray(image, mode="RGBA").convert("RGB")
        return Image.fromarray(image, mode="RGB")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _draw_instruction_bar(canvas: Image.Image, text: str) -> None:
    """Draw a black footer bar with white text - matches vis_episode_*.png."""
    w, h = canvas.size
    bar_h = 16
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([(0, h - bar_h), (w, h)], fill=(0, 0, 0))
    # Pillow's default font is small enough to fit at 256 px width.
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((4, h - bar_h + 2), text, fill=(255, 255, 255), font=font)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def encode_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def render_overlay_png(
    image_bytes_or_path,
    waypoints: Dict[str, Sequence[float]],
    *,
    K: Optional[np.ndarray] = None,
    T_wc: Optional[np.ndarray] = None,
    instruction_text: Optional[str] = None,
) -> bytes:
    """Convenience: take a PNG path or raw bytes, return overlay-PNG bytes."""
    if isinstance(image_bytes_or_path, (str, bytes, bytearray)) and not (
        isinstance(image_bytes_or_path, (bytes, bytearray))
    ):
        with open(image_bytes_or_path, "rb") as f:
            data = f.read()
    elif isinstance(image_bytes_or_path, (bytes, bytearray)):
        data = bytes(image_bytes_or_path)
    else:
        raise TypeError("render_overlay_png expects a path or bytes")
    img = render_overlay(
        data,
        waypoints,
        K=K,
        T_wc=T_wc,
        instruction_text=instruction_text,
    )
    return encode_png(img)


def project_waypoints_summary(
    waypoints: Dict[str, Sequence[float]],
    *,
    K: Optional[np.ndarray] = None,
    T_wc: Optional[np.ndarray] = None,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    draw_keys: Iterable[str] = DRAW_WAYPOINTS,
) -> List[Dict[str, object]]:
    """Compute the in-image pixel position of each waypoint origin.

    Useful for the agent prompts: lets the inquisitor reason about where each
    overlay is on the image even without seeing the rendered pixels.
    """
    K_eff = np.asarray(K if K is not None else DEFAULT_K, dtype=np.float64)
    T_wc_eff = np.asarray(T_wc if T_wc is not None else DEFAULT_T_WC, dtype=np.float64)
    H, W = image_size
    out: List[Dict[str, object]] = []
    for key in draw_keys:
        pose = waypoints.get(key)
        if pose is None or len(pose) < 6:
            continue
        origin = np.asarray(pose[:3], dtype=np.float64)
        pix = project_world_to_pixel(origin, T_wc_eff, K_eff, W, H)
        out.append(
            {
                "key": key,
                "label": WAYPOINT_LABEL.get(key, key[:2]),
                "world_xyz": [float(x) for x in origin.tolist()],
                "pixel_uv": list(pix) if pix is not None else None,
                "in_frame": pix is not None,
            }
        )
    return out
