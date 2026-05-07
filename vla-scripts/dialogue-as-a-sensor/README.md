# Dialogue as a Sensor: Resolving Navigational Uncertainty through Human-Robot Conversations with Multi-Agent System

**Sherry Li**  
*xuehail2@illinois.edu*  

**Jingming Chen**  
*jc166@illinois.edu*  

**Mathew Ishaq**  
*mrishaq2@illinois.edu*  

**University of Illinois Urbana-Champaign**

## Abstract

Current embodied manipulation agents, specifically Vision-Language-Action (VLA) models, often operate as open-loop systems that fail silently in ambiguous environments. These agents lack the communicative grounding necessary to actively query a human expert when low-level physical execution fails. To address this gap, we propose an interactive multi-agent system (MAS) that utilizes natural language dialogue as an active, bimodal spatial sensor to resolve both perceptual and physical uncertainty. Orchestrated via LangGraph, our framework introduces a cognitive pause driven by a Dual-Trigger mechanism, monitoring semantic ambiguity and kinematic collisions, to initiate Human-in-the-Loop (HITL) recovery. Furthermore, we ground the Natural Language Understanding (NLU) agent entirely in Operational Space Control (OSC), mapping unstructured human text directly into Cartesian action deltas. This approach effectively bridges the gap between high-level reasoning and low-level continuous control to enable reliable error recovery trajectories in robotic manipulation.

## Introduction

Current embodied manipulation agents typically operate as “silent executors”. While Vision-Language-Action (VLA) models excel at semantic generalization, they function as open-loop systems that blindly execute high-probability actions. When encountering ambiguous tabletop environments, such as heavily cluttered workspaces or novel occlusions, they fail silently. They either attempt an unsafe trajectory that results in a collision, or they freeze due to low internal confidence. The core deficiency is a lack of communicative grounding; current systems cannot actively query a human expert when their physical execution fails. Standard VLA failures in environments like a cluttered workspace result in the robot colliding with an obstacle, its end-effector stalling, but the model continuously pumping out forward-motion commands until the episode timer expires.

This research asks: How can a manipulation robot utilize natural language dialogue not just to receive initial instructions, but as a high-fidelity spatial sensor to resolve physical, kinematic, and perceptual uncertainty via Human-in-the-Loop (HITL) intervention?

We expect to make the following contributions with this project:

- **Dual-Trigger Bimodal Sensor:** We propose treating human dialogue as an active, bimodal spatial sensor that resolves both perceptual and physical uncertainty. This includes a Semantic Ambiguity Trigger that halts the simulation if grasp confidence drops, and a Kinematic Collision Trigger that intercepts the continuous control loop when it is mathematically certain that the robot is physically obstructed.

- **Actionable OSC Grounding:** We propose grounding the Natural Language Understanding (NLU) agent entirely in Operational Space Control (OSC) to map human spatial intent directly into Cartesian action deltas in the robot’s base coordinate frame. This transforms unstructured text into precise recovery trajectories.

- **Conversational Multi-Agent Architecture:** We propose an interactive, conversational multi-agent system (MAS) utilizing LangGraph or LangChain agents to orchestrate dialogue, state monitoring, and spatial grounding. Unlike standard VLA pipelines that operate strictly as Directed Acyclic Graphs (DAGs), this allows us to model the system as a state machine with cyclic graphs for multi-turn error recovery.

## Proposed Approach

We propose an interactive, conversational multi-agent system (MAS) utilizing LangGraph to orchestrate dialogue, state monitoring, and spatial grounding. Unlike standard Vision-Language-Action (VLA) pipelines that operate as Directed Acyclic Graphs (DAGs) flowing strictly from input to output, LangGraph allows us to model the system as a state machine with cyclic graphs for multi-turn error recovery. The environment will be simulated in Robosuite 1.5, controlling the manipulator via Operational Space Control (OSC).

### The LangGraph State Definition

To ensure seamless passing of variables between nodes, the shared LangGraph State tracks:

- `image_context`: The current `agentview_image` from MuJoCo.
- `kinematic_state`: End-effector velocity, current OSC command, and `R_C^B` Camera-to-Base rotation matrix.
- `trigger_reason`: A string flag, such as `semantic_ambiguity` or `kinematic_collision`.
- `dialogue_history`: The conversational loop between the robot and the human.
- `osc_delta`: The resulting 7D vector calculated by the NLU agent.

### Node Implementation

The architecture operates in four distinct phases, or nodes, conditionally routed based on the State:

- **Node 1: The Dual-Trigger Monitor (Agent 1 - State Monitor):** This intercepts the `env.step()` function to monitor the Robosuite simulation state. If a threshold is breached, it freezes the simulation. It fires if the VLA’s internal grasp confidence drops below a predefined safety threshold, or if the end-effector velocity drops to approximately 0 while the OSC command remains non-zero. This pauses the simulation, then sends the agent's current view (as an image) overlayed with 7-DoF waypoints for key actions(A1, A2, A3) in the action plan and the agent's task description (as text) to agent 2.

- **Node 2: The Visual Inquisitor (Agent 2 - NLG):** Triggered by Agent 1, this node assesses the paused simulation and asks the human for help. It sees the current "stuck" state (the `agentview_image` with overlayed waypoints and task description) and utilizes a Vision-Language Model (VLM) to generate a clarifying question based on the state, such as asking whether to push a blocking object or attempt a different grasp. This node then asks the human-in-the-loop user (in our case, a user simulated using a VLM) the clarifying question, expecting an answer as a clarified instruction to send to agent 3.

    - For our project, we will be simulating a human-user using a VLM. The simulated user will receive the "stuck" state (the `agentview_image` with overlays and task description) and the clarifying question (from agent 2) as input, and return an answer to the clarifying question, such as "reach around the box to grab the cup."

- **Node 3: The Spatial-to-OSC Grounding Agent (Agent 3 - NLU):** This agent parses the human-robot conversation and maps the semantic intent into precise OSC action deltas. For instance, it can map “grasp from the top” to a positive `Δz` and a downward pitch `Δβ` rotation to the camera frame. The input for agent 3 will be the "stuck" state (overlayed image and state information), and the human-robot conversation. The output of agent 3 will be a JSON object with deltas to adjust each 7-DoF numbers for each actions A1, A2, and A3 if needed in order to correct the action plan. The output action deltas will be sent to agent 4 for action plan correction and execution.

- **Node 4: The VLA Executor (Agent 4 - Action):** This node resumes the simulation physics and recalculates its trajectory based on the newly injected OSC constraints (from agent 3). We utilize OpenVLA, fine-tuned to predict specific affordance waypoints, converting the sequence into the 7D OSC action vector: `[Δx, Δy, Δz, Δα, Δβ, Δγ, gripper]`.

### 3.3 Conditional Routing (Edges)

The flow between the nodes is dictated by conditional routing logic inherent to the LangGraph architecture:

- **Monitor → NLG:** If the State Monitor (Node 1) detects a breach, it updates the `trigger_reason` and routes the state to the Visual Inquisitor (Node 2).

- **Monitor → Action:** If the simulation remains within safety thresholds, the state bypasses the dialogue loop entirely and routes directly to the VLA Executor (Node 4) for continuous operation.

- **NLU → Action:** Once the Spatial Grounding Agent (Node 3) successfully populates the `osc_delta` vector, the state is routed to Node 4 to resume the simulation physics.

## Experiment Plan

To effectively train and evaluate our proposed architecture, our experiment plan is divided into dataset curation and automated metric evaluation.

### Datasets

We will utilize three primary datasets tailored to the specific nodes of our multi-agent system:

- **Clutter Awareness Dataset:** To train the Visual Inquisitor (Agent 2), we will construct a dataset consisting of 1,000+ paired examples of ambiguous or cluttered Robosuite scenes mapped to ground-truth failure captions and generated clarifying questions.

- **Dialogue-to-Coordinate Dataset:** To train the Spatial-to-OSC Grounding Agent (Agent 3), we will utilize a synthetic dataset of “Dialogue-to-Coordinate” pairs. This will be used for few-shot prompting or lightweight fine-tuning to ensure accurate mapping of semantic intent to 7D Cartesian constraints.

- **Robosuite Teleoperation Dataset:** To train the VLA Executor (Agent 4), we require a robust teleoperation dataset of cluttered tabletop manipulations. This dataset will be used to fine-tune the OpenVLA model via LoRA/qLoRA, training it to predict specific spatial affordance waypoints, such as pre-grasp, grasp, and release, rather than just imitating end-to-end trajectories.

### Evaluation Metrics

To rigorously evaluate the system’s progress and effectiveness, we will conduct automated batch testing. Our primary quantitative metrics will focus on comparing our multi-agent framework against a standard, non-conversational baseline:

- **Success Rate Weighted by Path Length (SPL):** We will measure the SPL of the dialogue-enabled multi-agent system against a silent OpenVLA baseline to evaluate task efficiency and successful completion.

- **Collision Rates:** We will compare the collision rates between our system and the baseline to quantify the safety improvements gained through the Kinematic Collision Trigger and human-in-the-loop recovery interventions.

### Baselines and Ablation Studies

To isolate the impact of our Dual-Trigger mechanism and OSC grounding, we will evaluate our system against the following baselines and ablations:

- **Baseline 1 (Silent OpenVLA):** A standard, open-loop OpenVLA model fine-tuned on the same dataset, operating without any human-in-the-loop intervention.

- **Ablation 1 (Visual Trigger Only):** Our architecture with the Kinematic Collision Trigger disabled, relying solely on the Semantic Ambiguity threshold to pause the simulation.

- **Ablation 2 (Kinematic Trigger Only):** Our architecture with the VLM grasp confidence monitor disabled, meaning the system will only ask for help after a physical collision has already occurred.

This ablation structure will allow us to quantify the preventative safety value of the visual trigger versus the deterministic recovery value of the kinematic trigger.

## Implementation Status

| Component                                     | Where it lives                                                  | Status                                                                                                                                                                                                                  |
| --------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Agent 1 — Dual-Trigger Monitor**            | (offline analogue in `data-generation/generate_stuck_dataset.py`) | Partially implemented. The offline data generator captures the same kinematic-stall signal (EEF speed ≈ 0 with non-zero command + position error > 5 cm) that an online Agent 1 would fire on, and writes per-phase frames + `kinematic_state` into each episode for the rest of the pipeline to consume. The semantic ambiguity trigger from the VLA model still has to be implemented (because the current codebase does not run on a VLA model). |
| **Agent 2 — Visual Inquisitor (NLG)**         | `agent-system/src/dialogue_sensor/agents/visual_inquisitor.py`  | Implemented. Default model `gpt-5-mini-2025-08-07` (OpenAI). Sees the agentview image with A1/A2/A3 overlay + 7-DoF waypoints (m + rad) and asks one focused clarifying question per turn.                                |
| **Simulated Human user**                       | `agent-system/src/dialogue_sensor/agents/simulated_user.py`     | Implemented. Default model `gemini-3-flash-preview` (Google). Different VLM family from Agent 2 to avoid the two echoing each other. Returns a clarified instruction grounded in the same image + overlay.              |
| **Agent 3 — Spatial-to-OSC Grounding (NLU)**  | `agent-system/src/dialogue_sensor/agents/spatial_grounding.py`  | Implemented. Default model `gpt-5-mini-2025-08-07`. Parses the dialogue and emits a JSON envelope `{"reasoning": ..., "corrections": {...}}` of per-waypoint deltas (m + rad), then applies them to produce the corrected waypoints + a 7-DoF OSC Δ vector for Agent 4. |
| **Agent 4 — VLA Executor (Action)**           | (not implemented yet)                                    | Planned. Will consume `osc_delta` / `corrected_waypoints` from Agent 3 and resume the simulation through OpenVLA.                                                                                                       |

Agents 2/3 + the simulated user are wired together as a single LangGraph
state machine:

```
START ──► visual_inquisitor ──► simulated_user ──┬─► (loop) ─► visual_inquisitor
                                                 └─► (done) ─► spatial_grounding ─► END
```

The `(loop)` edge is the cyclic HITL recovery path described in §"Conditional Routing (Edges)".

## Quick Start

This project has two top-level components, each with its own setup notes:

- `data-generation/` — Robosuite 1.5 simulator that produces episode datasets (success and stuck-state). See `data-generation/README.md` (full pipeline) and `data-generation/SETUP.md` (server/env workarounds).
- `agent-system/` — LangGraph implementation of Agent 2 (Visual Inquisitor), the simulated HITL user, and Agent 3 (Spatial-to-OSC Grounding). See `agent-system/README.md`.

The agent system consumes episode directories produced by data generation, so set up data generation first.

### 1. Conda environment

Tested on Ubuntu 24.04, Python 3.10, robosuite 1.5.1, mujoco 3.8.0, NVIDIA GPU.

```bash
conda create -n convai python=3.10 -y
conda activate convai
```

### 2. Install data-generation dependencies

```bash
cd /home/jc166/dialogue-as-a-sensor/data-generation

pip install tensorflow-datasets tensorflow opencv-python apache-beam \
            mlcroissant mujoco numba scipy h5py

# GLVND wrappers needed by MuJoCo's EGL backend on headless servers
conda install -y -c conda-forge libegl libgl

touch __init__.py   # required by `tfds build` import system
```

If MuJoCo rendering crashes with `'NoneType' object has no attribute 'eglQueryString'`, see `data-generation/SETUP.md` §3. The repo already includes the `macros_private.py`, depth NaN guard, and observable-toggle workarounds described in that file.

### 3. Generate a dataset

`run_generation.sh` and `run_stuck_generation.sh` set the required env vars (`MUJOCO_GL=egl`, `LD_LIBRARY_PATH`) and call the generation scripts.

```bash
# Successful-trajectory dataset (legacy)
bash run_generation.sh
# -> ./my_data/episode_00000/ ... with image_rgb.png, image_depth.npy, metadata.json

# Stuck-state dataset (used by the agent system for HITL recovery)
bash run_stuck_generation.sh
# -> ./my_stuck_data/episode_00000/ ... with image_stuck.png, image_phase_*.png,
#    richer metadata.json (instruction, waypoints, success, trigger_reason,
#    stuck_phase, kinematic_state, scene_objects, camera_intrinsics,
#    camera_extrinsics_wc, camera_image_size).
```

Tune `--num_trials`, `--num_videos`, `--start_index` inside the shell scripts. On a memory-constrained GPU keep `--num_videos 0` (see SETUP.md §6).

The stuck-state generator additionally writes `camera_intrinsics` (3×3 K) and `camera_extrinsics_wc` (4×4 T_wc) into each episode's `metadata.json`. The agent system uses these to project the planned waypoints into pixel space and overlay the action plan onto the agentview image (A1/A2/A3 axis frames in red/green/blue) before feeding it to the VLMs. Legacy `my_data/` episodes lack the matrices but still get an overlay rendered from baked-in defaults captured from the same fixed agentview camera.

### 4. Install the agent system

```bash
cd /home/jc166/dialogue-as-a-sensor/agent-system
pip install -e .   # reuses the `convai` env
```

### 5. Configure API keys

Agent 2 (Visual Inquisitor), the simulated user, and Agent 3 (Spatial-to-OSC Grounding) all need API access. Agent 2 and the simulated user are intentionally instantiated from different VLM families (OpenAI + Google Gemini) so they don't echo each other; Agent 3 defaults to OpenAI because its job is structured-JSON emission rather than open-ended conversation.

```bash
cp .env.example .env
# then edit .env and fill in:
#   OPENAI_API_KEY=sk-...
#   GOOGLE_API_KEY=...
```

Defaults (override any of these via `.env`):

| Agent              | env vars                                            | Default model               |
| ------------------ | --------------------------------------------------- | --------------------------- |
| Visual Inquisitor  | `VISUAL_INQUISITOR_PROVIDER` / `*_MODEL`            | `openai` / `gpt-5-mini-2025-08-07` |
| Simulated user     | `SIMULATED_USER_PROVIDER` / `*_MODEL`               | `google` / `gemini-3-flash-preview` |
| Spatial Grounding  | `SPATIAL_GROUNDING_PROVIDER` / `*_MODEL`            | `openai` / `gpt-5-mini-2025-08-07` |

Use `ENABLE_SPATIAL_GROUNDING=false` (or `--no_grounding` on the CLI) to skip Agent 3 — useful for ablations or when debugging Agent 2 alone.

### 6. Run the agent system

```bash
# No-API smoke test — renders ALL three agents' prompts (Agent 2,
# simulated user, Agent 3), hits no provider.
python -m dialogue_sensor.cli \
    --episode_dir /home/jc166/dialogue-as-a-sensor/data-generation/my_data/episode_00000 \
    --dry_run

# Live run: Agent 2 -> Sim User -> Agent 3 -> END.
python -m dialogue_sensor.cli \
    --episode_dir /home/jc166/dialogue-as-a-sensor/data-generation/my_stuck_data/episode_00003 \
    --image_filename image_phase_grasp_descend.png \
    --trigger_reason kinematic_collision \
    --max_turns 2 \
    --output_json runs/episode_00003.json
```

`--trigger_reason` accepts `semantic_ambiguity` or `kinematic_collision`. For successful (non-stuck) episodes from `generate_vla_dataset.py` you can omit `--image_filename`; the loader falls back to `image_rgb.png`.

When `--output_json runs/episode_00003.json` is passed, the CLI saves three artifacts:

```
runs/episode_00003.json                    # full state: dialogue history, Agent 3
                                           # corrections + reasoning, corrected
                                           # waypoints, OSC Δ vector
runs/episode_00003.overlay.png             # agentview image with the ORIGINAL
                                           # planned A1/A2/A3 axis frames overlaid
runs/episode_00003.corrected.overlay.png   # same image with the CORRECTED waypoints
                                           # overlaid - visualises Agent 3's edit
```

The corrected-overlay PNG is only written if Agent 3 actually emitted any corrections.

#### Output schema (Agent 3)

Agent 3 emits a single JSON envelope:

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
| `dx`, `dy`, `dz`                 | METERS                             | +x away from camera, +y viewer-right, +z up. Hard cap ±0.10 m. |
| `d_roll`, `d_pitch`, `d_yaw`     | RADIANS (scipy `'xyz'` Euler)      | Same units as the absolute waypoint angles. Common values ±π/4, ±π/2, ±π. Hard cap ±π. |
| `d_gripper`                      | dimensionless                      | Absolute gripper is in [-1=open, +1=closed]. Hard cap ±2. |

The CLI / state expose three derived fields:

- `corrections` — sanitised, clipped per-waypoint deltas.
- `corrected_waypoints` — original waypoints with the deltas applied (m + rad — same units as `metadata.waypoints`).
- `osc_delta` — the A2 delta as a single 7-DoF vector `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgripper]` in m + rad + dimensionless, ready to forward to Agent 4. `None` if A2 is unchanged.

### 7. Verify

```bash
# Offline smoke + Agent 3 unit tests (no API keys required, 16 tests).
cd /home/jc166/dialogue-as-a-sensor/agent-system
python -m pytest tests/ -q
```

A successful end-to-end run prints, in order:

1. The inquisitor's clarifying question (Agent 2).
2. The simulated user's clarified instruction.
3. Agent 3's reasoning, the per-waypoint corrections JSON, the OSC Δ vector for A2, and a side-by-side `original → corrected` table for A1 / A2 / A3.

It also writes the JSON trace + both overlay PNGs listed above so the corrected plan can be inspected visually.