# agent-system

Implementation of **Agent 2 (Visual Inquisitor / NLG)**, the **simulated
human-in-the-loop user**, and **Agent 3 (Spatial-to-OSC Grounding / NLU)** for
the *Dialogue as a Sensor* multi-agent robotic manipulation system. See
`../README.md` for the full architecture (Agents 1–4).

The three nodes are wired together as a small **LangGraph** state machine,
so the missing pieces (Agent 1 dual-trigger monitor, Agent 4 VLA executor)
can be plugged in later without restructuring the dialogue + grounding
loop.

```
START ──► visual_inquisitor ──► simulated_user ──┬─► (loop) ─► visual_inquisitor
                                                 └─► (done) ─► spatial_grounding ─► END
```

| Node                | VLM family    | Default model           | Role                                             |
| ------------------- | ------------- | ----------------------- | ------------------------------------------------ |
| `visual_inquisitor` | OpenAI        | `gpt-5-mini-2025-08-07` | Asks one clarifying question per turn            |
| `simulated_user`    | Google Gemini | `gemini-3-flash-preview`| Plays the human; returns clarified instruction   |
| `spatial_grounding` | OpenAI        | `gpt-5-mini-2025-08-07` | Converts dialogue → JSON of A1/A2/A3 OSC deltas  |

The inquisitor and the simulated user are intentionally instantiated from
**different model families** to avoid them echoing each other. Agent 3
defaults to OpenAI because its job is structured-JSON emission rather than
open-ended conversation; you can override any of the three via `.env`.

The image fed to all three agents is the agentview RGB **with the planned
A1/A2/A3 waypoints overlaid as 3D coordinate frames** (RED = X / pinch,
GREEN = Y / left-right, BLUE = Z / vertical). The overlay is rendered just
in time by `dialogue_sensor.overlays`, using the camera matrices from each
episode's `metadata.json` (or default agentview constants for legacy
episodes).

---

## 1. Install

```bash
# Either reuse the convai env that runs data-generation,
# or create a fresh env. Python 3.10+ required.
conda activate convai

# Install the package + its deps
cd /home/jc166/dialogue-as-a-sensor/agent-system
pip install -e .
```

## 2. Configure API keys

```bash
cp .env.example .env
# then edit .env and fill in OPENAI_API_KEY and GOOGLE_API_KEY
```

The `.env` is loaded automatically by every entry-point. Model selection
(provider + checkpoint) for each of the three agents is also controlled via
`.env` so swapping families is a one-line change.

## 3. Run

The CLI consumes any episode directory produced by the
`data-generation/` pipeline. It works on:

* legacy successful episodes from `generate_vla_dataset.py`
  (just `image_rgb.png` + `metadata.json`), and
* new stuck-state episodes from `generate_stuck_dataset.py`
  (`image_stuck.png` / `image_phase_*.png` + richer metadata that includes
  `camera_intrinsics` / `camera_extrinsics_wc`).

```bash
# Dry run - renders the prompts for all three agents without hitting any API.
python -m dialogue_sensor.cli \
    --episode_dir /home/jc166/dialogue-as-a-sensor/data-generation/my_data/episode_00000 \
    --dry_run

# Default live run: Agent 2 -> Sim User -> Agent 3 -> END.
python -m dialogue_sensor.cli \
    --episode_dir /home/jc166/dialogue-as-a-sensor/data-generation/my_data/episode_00000 \
    --trigger_reason semantic_ambiguity \
    --output_json runs/episode_00000.json
# Writes runs/episode_00000.json (full state + corrections) plus
# runs/episode_00000.overlay.png (original A1/A2/A3 overlay) and
# runs/episode_00000.corrected.overlay.png (overlay rendered with the
# corrected waypoints from Agent 3).

# Multi-turn dialogue on a stuck-state episode.
python -m dialogue_sensor.cli \
    --episode_dir /home/jc166/dialogue-as-a-sensor/data-generation/my_stuck_data/episode_00002 \
    --image_filename image_phase_grasp_descend.png \
    --trigger_reason kinematic_collision \
    --max_turns 2 \
    --output_json runs/stuck_episode_00002.json

# Skip Agent 3 (ablation / debugging Agent 2).
python -m dialogue_sensor.cli --episode_dir <dir> --no_grounding

# Skip the overlay (raw image only).
python -m dialogue_sensor.cli --episode_dir <dir> --no_overlay
```

### Agent 3 output schema

After the dialogue loop terminates, Agent 3 emits a single JSON envelope:

```json
{
  "reasoning": "<concise chain of thought>",
  "corrections": {
    "A1": {"d_yaw": 1.5708},
    "A2": {"dy": -0.03, "dz": 0.02}
  }
}
```

Allowed delta keys per waypoint:

| Key                              | Units                              | Notes                                        |
| -------------------------------- | ---------------------------------- | -------------------------------------------- |
| `dx`, `dy`, `dz`                 | METRES                             | +x away from camera, +y viewer-right, +z up. Hard cap ±0.10 m. |
| `d_roll`, `d_pitch`, `d_yaw`     | RADIANS (scipy `'xyz'` Euler)      | Same units as the absolute waypoint angles. Common values: ±π/4 ≈ ±0.785, ±π/2 ≈ ±1.571, ±π ≈ ±3.142. Hard cap ±π. |
| `d_gripper`                      | dimensionless                      | Absolute gripper is in [-1=open, +1=closed]. |

Rotation deltas are radians **everywhere** in the pipeline — no degree↔radian
conversion happens between the LLM's JSON, the validator, the
`apply_corrections()` step, or the `osc_delta` forwarded to Agent 4.

The CLI / state expose three derived fields:

* `corrections` — sanitised, clipped per-waypoint deltas.
* `corrected_waypoints` — original waypoints with the deltas applied
  (m + rad — same units as `metadata.waypoints`).
* `osc_delta` — the A2 delta as a single 7-DoF vector
  `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgripper]` in m + rad + dimensionless,
  ready to forward to Agent 4. `None` if A2 is unchanged.

## 4. Generating stuck-state episodes

The data-generation pipeline ships a sibling script that always saves
trials (success or failure), captures per-phase images, and writes camera
matrices into the metadata so the overlay always renders correctly:

```bash
cd /home/jc166/dialogue-as-a-sensor/data-generation
bash run_stuck_generation.sh
# or directly:
python generate_stuck_dataset.py \
    --output_dir ./my_stuck_data --num_trials 30 --save_all_phases
```

Each `episode_*` directory then contains:

```
image_initial.png                # before motion (matches legacy image_rgb.png)
image_rgb.png                    # alias for image_initial.png
image_stuck.png                  # selected stuck/representative frame
image_phase_<phase>.png          # per-phase frames (when --save_all_phases)
metadata.json                    # instruction + waypoints + camera matrices + trigger info
```

`metadata.json` adds `success`, `trigger_reason` (`kinematic_collision` |
`semantic_ambiguity` | `none`), `stuck_phase`, `kinematic_state`,
`scene_objects`, `camera_intrinsics`, `camera_extrinsics_wc`,
`camera_image_size` on top of the legacy fields.

---

## Layout

```
agent-system/
├── .env.example                 # Template for API keys + model choice
├── pyproject.toml
├── requirements.txt
├── README.md
├── tests/
│   └── test_smoke.py            # 16 offline tests (no API calls)
└── src/
    └── dialogue_sensor/
        ├── __init__.py
        ├── config.py            # Loads .env -> Config dataclass
        ├── state.py             # LangGraph DialogueState schema
        ├── llm_clients.py       # OpenAI + Gemini vision clients
        ├── overlays.py          # A1/A2/A3 axis-frame renderer
        ├── episode_loader.py    # Episode dir -> initial DialogueState
        ├── graph.py             # LangGraph wiring (Agents 2 + sim + 3)
        ├── cli.py               # `python -m dialogue_sensor.cli ...`
        └── agents/
            ├── visual_inquisitor.py   # Agent 2 (NLG)
            ├── simulated_user.py      # HITL simulator
            └── spatial_grounding.py   # Agent 3 (NLU / OSC grounding)
```

## Smoke test

```bash
# No API keys required.
python -m pytest tests/ -q
# or directly:
python -m dialogue_sensor.cli --episode_dir <any_episode> --dry_run
```
