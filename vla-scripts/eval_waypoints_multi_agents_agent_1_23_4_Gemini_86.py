import os
os.environ["MUJOCO_GL"] = "egl"

import robosuite as suite
from robosuite.utils import camera_utils
from scipy.spatial.transform import Rotation as R
import cv2

import json
import base64
import re
import json_numpy
json_numpy.patch()

from google import genai
from google.genai import types
from PIL import Image
import os
import json

# Initialize the NEW SDK Client
client = genai.Client(api_key="AIzaSyCWN2r4ajumw0vYbMvwbt_HhbVzFYPf0Qo")


# --- Import your new environment to register it ---
try:
    import robosuite.environments.manipulation.pick_place_clutter
except ImportError:
    print("=" * 80)
    print("ERROR: Could not import PickPlaceClutter environment.")
    exit()

import sys
import time
import requests
import numpy as np
from pathlib import Path
import draccus
import torch

from generate_vla_dataset_visualize import VLADataGenerator
from visualize_dataset_and_affordances import visualize_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 0. HELPER FUNCTIONS
# ==========================================
def get_base64_image(image_path):
    """Helper to load and encode images"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def check_safety_strict(a1, a2, pre_q, grasp_q):
    """The strict, original Agent 1 logic"""
    if a1[1] < -0.31:
        return False, f"Target is near the workspace edge (Y={a1[1]:.3f}). Potential center-bias distortion."
    
    xy_drift = np.linalg.norm(a1[:2] - a2[:2])
    if xy_drift > 0.041:
        return False, f"Kinematic inconsistency detected. Pre-grasp and Grasp XY drift is {xy_drift*100:.1f}cm."
    
    yaw_diff = abs(R.from_quat(pre_q).as_euler('xyz')[2] - R.from_quat(grasp_q).as_euler('xyz')[2])
    phys_yaw = min(yaw_diff % np.pi, np.pi - (yaw_diff % np.pi))
    if phys_yaw > 0.42:
        return False, f"Orientation inconsistency detected. Wrist yaw physically shifts {np.degrees(phys_yaw):.1f} degrees."
    
    return True, "Trajectory looks semantically safe."

# ==========================================
# ENVIRONMENT & EVALUATION HELPERS
# ==========================================
class VideoEnv():
    def __init__(self, env, init_obs, cfg):
        self.env = env
        self.cfg = cfg
        self.obs = init_obs
        self.robot_pos = self.obs["robot0_eef_pos"].copy()
        self.robot_quat = self._get_current_quat()
        self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)
        self._rec = {}
        self.cam_width = self.env.camera_widths[0]
        self.cam_height = self.env.camera_heights[0]

    def _get_current_quat(self):
        return self.obs["robot0_eef_quat_site"].copy()

    def video_frame(self, text=None):
        if not getattr(self, "_rec", None) or not self._rec["on"]: return
        H, W = self._rec["H"], self._rec["W"]
        cam = self._rec["camera"]
        rgb = self.env.sim.render(camera_name=cam, height=H, width=W, depth=False)
        frame = cv2.flip(rgb, 0)
        if frame.dtype != np.uint8: frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        if text:
            cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bgr = np.ascontiguousarray(bgr)
        self._rec["frames"] += 1
        self._rec["writer"].write(bgr)
        
    def video_start(self, path="eval.mp4", fps=30, H=256, W=256, camera_name="agentview"):
        self._rec = {"on": False, "path": path, "fps": fps, "H": H, "W": W, "camera": camera_name, "frames": 0}
        for fourcc_str in ("avc1", "mp4v", "XVID", "MJPG"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(path, fourcc, fps, (W, H), True)
            if writer.isOpened():
                self._rec.update({"writer": writer, "fourcc": fourcc_str, "on": True})
                break
        if not self._rec.get("on", False):
            print(f"⚠️ WARNING: OpenCV failed to open VideoWriter for {path}. Skipping video recording for this trial.")
            return
            
        _ = self.env.sim.render(camera_name=camera_name, height=H, width=W, depth=False)
        for _ in range(3): self.video_frame("start")

    def video_stop(self):
        if getattr(self, "_rec", None) and self._rec.get("on", False):
            self._rec["writer"].release()
            print(f"[VIDEO] Saved to {self._rec['path']} (codec={self._rec.get('fourcc')}, frames={self._rec['frames']})")
            self._rec["on"] = False
    
    def step(self, action):
        action_np = np.array(action, dtype=np.float64, copy=True)
        if not action_np.flags.writeable: action_np = action_np.copy()
        self.obs, _, _, _ = self.env.step(action_np)
        self.video_frame()

from dataclasses import dataclass

from copy import deepcopy

@dataclass
class EvalConfig:
    exp_id: str = None
    exp_tag: str = None
    output_dir: Path = Path("eval_output")
    num_times: int = 50

@draccus.wrap()
def eval(cfg: EvalConfig) -> None:
    seed = 40
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"🌱 Random seed set to {seed} for reproducible evaluations.")
    
    task_generator = VLADataGenerator("eval_task")
    total_successes = 0
    failed_trials = []
    
    # --- NEW: Master Log for Analysis ---
    trial_logs = []
    
    
    SYSTEM_RULES = '''System Role: You are an expert robotic spatial reasoning agent correcting hallucinated waypoints.
Coordinate System Rules:
- X-axis (Red line): Red line represents the Pinch direction. Forward-Backward. Positive dx moves away from camera.
- Y-axis (Green line): Left/Right across the screen from the viewer's perspective. Positive dy moves to the RIGHT. Negative dy moves to the LEFT.
- Z-axis (Blue line): Up/Down. Positive dz moves higher (lifts).

Visual Geometry Rule for Yaw (Gripper Orientation):
The robot's gripper pinches along the X-axis (Red line). The robot relies heavily on A1's orientation to begin its descent. For a successful grasp, the predicted axes MUST align with the box's geometry, regardless of how the box is rotated on the table:
- GOOD ALIGNMENT: The Red line (X-axis) points perpendicularly INTO the broad, flat face of the box. The Green line (Y-axis) runs parallel to the long edge of the box.
- BAD ALIGNMENT (90-deg error): The Red line runs parallel to the long edge of the box (smash directly into the narrow top or side edges, pushing the box over and failing the grasp.).
- BAD ALIGNMENT (45-deg error): The Red and Green lines are diagonal/skewed relative to the flat faces of the box.

Correction Strategies (In Order of Priority):
1. "A1 Yaw Override": If the predicted orientation has BAD ALIGNMENT, the robot will fail its descent. Apply a clean `d_yaw` to A1 (e.g., 45.0, 90.0, or -90.0) to force the Red line perpendicular to the box's broad, flat face (and the long axis of the box).
2. "Micro-Nudge XY": If orientation is GOOD, but A2 is missing the edge or not overlapping with the box, apply a tiny `dy` (e.g., 0.02 or -0.02, 0.03 or -0.03) to A2.
3. "Micro-Nudge Z": If orientation is GOOD and centered, but A2 clipping too deeply into the box, lift slightly with `dz` (e.g., 0.02) to A2.
4. "Do Not Touch": If alignment is GOOD and A2 safely overlaps the object without clipping, do nothing, output an empty JSON: {}.

Task Protocol:
You must "think out loud" and explicitly answer this 3-step checklist before outputting your JSON:
1. Look at the coordinate axes. Does the A1 Red line (X-axis) point perpendicularly into the broad face of the box (Good), or is it misaligned parallel/diagonal to the long edge (Bad)? (If Bad -> use A1 Yaw Override).
2. Is A2 slightly missing the left or right edge of the cereal box from the viewer's perspective? (If hovering to the Right -> Needs Negative dy. If hovering to the Left -> Needs Positive dy).
3. Is A2 perfectly centered but clipping too deeply into the object in Z? (If yes -> use Micro-Nudge dz).

If Step 1 is "Good", and Steps 2 and 3 are "No", you MUST output an empty JSON: {}.
'''

    # --- THE EXAMPLES ---

    EXAMPLE_1_TRIGGER = '''"Orientation inconsistency detected. Wrist yaw physically shifts 42.1 degrees."'''
    EXAMPLE_1_RESPONSE = '''Reasoning:
1. Look at the coordinate axes. Does the A1 Red line (X-axis) point perpendicularly into the broad face of the box? No. The Red line is pointing parallel to the long edge of the box (Bad Alignment).
2. Is A2 slightly missing the left or right edge of the cereal box from the viewer's perspective? N/A.
3. Is A2 perfectly centered but clipping too deeply into the object in Z? N/A.

Because the axes are misaligned by 90 degrees, I will use the "A1 Yaw Override" strategy. I will adjust A1 by 90 degrees to force the Red line to be perpendicular to the broad face for a safe descent.

```json
{"A1": {"d_yaw": 90.0}}
```'''

    EXAMPLE_2_TRIGGER = '''"Reason: Kinematic inconsistency detected. Pre-grasp and Grasp XY drift is 3.5cm."'''
    EXAMPLE_2_RESPONSE = '''Reasoning:
1. Look at the coordinate axes. Does the A1 Red line (X-axis) point perpendicularly into the broad face of the box? Yes, the alignment is Good.
2. Is A2 slightly missing the left or right edge of the cereal box from the viewer's perspective? Yes, A2 is hovering just slightly to the Right of the cereal box. Therefore, it Needs Negative dy.
3. Is A2 perfectly centered but clipping too deeply into the object in Z? No.

Because the orientation is good but it barely misses the right edge, I will use the "Micro-Nudge XY" strategy. A tiny shift left (-0.02) will center A2.

```json
{"A2": {"dy": -0.02}}
```'''

    EXAMPLE_3_TRIGGER = '''"Reason: Kinematic inconsistency detected. Pre-grasp and Grasp XY drift is 2.8cm."'''
    EXAMPLE_3_RESPONSE = '''Reasoning:
1. Look at the coordinate axes. Does the A1 Red line (X-axis) point perpendicularly into the broad face of the box? I don't see A1. Move to the next question.
2. Is A2 slightly missing the left or right edge of the cereal box from the viewer's perspective? No, it is roughly centered/overlapped over the box.
3. Is A2 perfectly centered but clipping too deeply into the object in Z? Yes, the A2 origin is sinking too deeply into the top of the box, risking a hard collision.

Because the orientation is fine but the depth is dangerous, I will use the "Micro-Nudge Z" strategy to lift A2 slightly upward.

```json
{"A2": {"dz": 0.02}}
```'''

    EXAMPLE_4_TRIGGER = '''"Target is near the workspace edge (Y=-0.314). Potential center-bias distortion."'''
    EXAMPLE_4_RESPONSE = '''Reasoning:
1. Look at the coordinate axes. Does the A1 Red line (X-axis) point perpendicularly into the broad face of the box? Yes. Even though the box is diagonal on the table, the Red line is correctly perpendicular to the box's broad face (Good Alignment).
2. Is A2 slightly missing the left or right edge of the cereal box from the viewer's perspective? No, it safely overlaps.
3. Is A2 perfectly centered but clipping too deeply into the object in Z? No, the A2 origin is right just slightly below the upper edge.

Because all three checks indicate a safe trajectory, I will do nothing and output an empty JSON: {}

```json
{}
```'''

    EXAMPLE_5_TRIGGER = '''"Orientation inconsistency detected. Wrist yaw physically shifts 45.3 degrees."'''
    EXAMPLE_5_RESPONSE = '''Reasoning:
1. Look at the coordinate axes. Does the A1 Red line (X-axis) point perpendicularly into the broad face of the box? Yes. Even though the box is diagnonal on the table, the Red Line of A1 is correctly perpendicular to the box's broad face (Good Alignment).
2. Is A2 slightly missing the left or right edge of the cereal box from the viewer's perspective? No. The A2 origin is centered on the box.
3. Is A2 perfectly centered but clipping too deeply into the object in Z? No. The A2 origin is correctly positioned slightly below the top edge of the box.

Because all three checks indicate a safe trajectory, I will do nothing and output an empty JSON: {}

```json
{}
```'''

    for i in range(cfg.num_times):
        task_generator.instruction = task_generator.generate_instruction("cereal box", "target bin")
        print("\n" + "-"*40)
        print(f"🚀 STARTING TRIAL {i} / {cfg.num_times - 1}")
        print("-"*40)
        
        task_env = task_generator.env
        video_env = VideoEnv(task_env, task_env.reset(), cfg)
        neutral_pos = video_env.robot_pos.copy()
        neutral_quat = video_env.robot_quat.copy()

        print(f"Instruction: {task_generator.instruction}")
        img = video_env.obs["agentview_image"]
        img = cv2.flip(img, 0).astype(np.uint8)

        try:
            waypoint_1_action = requests.post("http://0.0.0.0:8001/act", json={"image": img.copy(), "instruction": task_generator.instruction, "predict_mode": "pregrasp"}).json()
            waypoint_2_action = requests.post("http://0.0.0.0:8001/act", json={"image": img.copy(), "instruction": task_generator.instruction, "predict_mode": "grasp"}).json()
            waypoint_3_action = requests.post("http://0.0.0.0:8001/act", json={"image": img.copy(), "instruction": task_generator.instruction, "predict_mode": "release"}).json()
        except Exception as e:
            print(f"Network error communicating with server: {e}")
            failed_trials.append(i)
            continue

        a1_pos = np.array(waypoint_1_action[:3], dtype=np.float64, copy=True)
        a2_pos = np.array(waypoint_2_action[:3], dtype=np.float64, copy=True)
        a3_pos = np.array(waypoint_3_action[:3], dtype=np.float64, copy=True)

        pregrasp_quat = R.from_euler('xyz', np.array(waypoint_1_action[3:6], dtype=np.float64, copy=True)).as_quat()
        grasp_quat = R.from_euler('xyz', np.array(waypoint_2_action[3:6], dtype=np.float64, copy=True)).as_quat()
        standard_place_quat = R.from_euler('xyz', np.array(waypoint_3_action[3:6], dtype=np.float64, copy=True)).as_quat()

        trajectory_goals_3d = {
            "A1_pregrasp": (a1_pos.tolist(), pregrasp_quat.tolist()),
            "A2_grasp": (a2_pos.tolist(), grasp_quat.tolist()),
            "A3_release": (a3_pos.tolist(), standard_place_quat.tolist()),
        }

        # --- SAVE BASELINE FOR LOGGING ---
        baseline_trajectory = deepcopy(trajectory_goals_3d)
        vlm_intervened = False
        vlm_raw_output = "N/A"

        # ==========================================
        # MULTI-AGENT RECOVERY SYSTEM (ONE-SHOT)
        # ==========================================
        def check_safety_strict(a1, a2, pre_q, grasp_q):
            if a1[1] < -0.31: return False, f"Target is near the workspace edge (Y={a1[1]:.3f}). Potential center-bias distortion."
            xy_drift = np.linalg.norm(a1[:2] - a2[:2])
            if xy_drift > 0.041: return False, f"Kinematic inconsistency detected. Pre-grasp and Grasp XY drift is {xy_drift*100:.1f}cm."
            yaw_diff = abs(R.from_quat(pre_q).as_euler('xyz')[2] - R.from_quat(grasp_q).as_euler('xyz')[2])
            phys_yaw = min(yaw_diff % np.pi, np.pi - (yaw_diff % np.pi))
            if phys_yaw > 0.42: return False, f"Orientation inconsistency detected. Wrist yaw physically shifts {np.degrees(phys_yaw):.1f} degrees."
            return True, "Trajectory looks semantically safe."

        is_safe, trigger_reason = check_safety_strict(a1_pos, a2_pos, pregrasp_quat, grasp_quat)

        if not is_safe:
            print("\n" + "!"*50)
            print("🛑 AGENT 1 TRIGGER: SIMULATION PAUSED")
            print(f"Reason: {trigger_reason}")
            print("!"*50)
            
            print("📸 Generating visualization for VLM review...")
            cam_name = "agentview"
            K = camera_utils.get_camera_intrinsic_matrix(task_generator.env.sim, cam_name, task_generator.cam_height, task_generator.cam_width)
            T_wc = camera_utils.get_camera_extrinsic_matrix(task_generator.env.sim, cam_name)
            
            waypoint_labels_predicted = {
                "A1_pregrasp": task_generator._get_7dof_pose(a1_pos, pregrasp_quat, -1.0),
                "A2_grasp":    task_generator._get_7dof_pose(a2_pos, grasp_quat,  1.0),
                "A3_release":  task_generator._get_7dof_pose(a3_pos, standard_place_quat, -1.0),
                "A4_home":     task_generator._get_7dof_pose(neutral_pos, neutral_quat, -1.0)
            }
            
            debug_dir = Path("debug_intervention")
            debug_dir.mkdir(exist_ok=True)
            task_generator.output_dir = debug_dir
            
            task_generator.save_sample(
                trial_idx=0, rgb_img=img, depth_img=task_generator.obs["agentview_depth"],
                instruction=task_generator.instruction, waypoint_labels=waypoint_labels_predicted,
                K=K, T_wc=T_wc
            )
            visualize_dataset(debug_dir, debug_dir, num_samples=1)
            
            # --- ONE SHOT VLM CALL ---
            # --- ONE SHOT GEMINI VLM CALL ---
            print("\n🤖 Requesting Single-Shot Gemini Correction...")
            
            try:
                # Gemini takes PIL Images directly! No Base64 needed.
                live_img = Image.open("debug_intervention/vis_episode_00000.png")
                ex1_img = Image.open("eval_failed_cases/trial_33_FAILED_prediction_plot.png")
                ex2_img = Image.open("eval_failed_cases/trial_23_FAILED_prediction_plot.png")
                ex3_img = Image.open("eval_failed_cases/trial_19_FAILED_prediction_plot.png")
                ex4_img = Image.open("eval_failed_cases/trial_4_SUCCESS_prediction_plot.png")
                ex5_img = Image.open("eval_failed_cases/trial_35_SUCCESS_prediction_plot.png")

                # Build the interleaved text-and-image prompt
                prompt_contents = [
                    SYSTEM_RULES,
                    "\n--- EXAMPLE 1 ---",
                    f"Trigger: {EXAMPLE_1_TRIGGER}\nDiagnose the image and provide the JSON correction.",
                    ex1_img,
                    f"Assistant Response:\n{EXAMPLE_1_RESPONSE}",
                    
                    "\n--- EXAMPLE 2 ---",
                    f"Trigger: {EXAMPLE_2_TRIGGER}\nDiagnose the image and provide the JSON correction.",
                    ex2_img,
                    f"Assistant Response:\n{EXAMPLE_2_RESPONSE}",
                    
                    "\n--- EXAMPLE 3 ---",
                    f"Trigger: {EXAMPLE_3_TRIGGER}\nDiagnose the image and provide the JSON correction.",
                    ex3_img,
                    f"Assistant Response:\n{EXAMPLE_3_RESPONSE}",
                    
                    "\n--- EXAMPLE 4 ---",
                    f"Trigger: {EXAMPLE_4_TRIGGER}\nDiagnose the image and provide the JSON correction.",
                    ex4_img,
                    f"Assistant Response:\n{EXAMPLE_4_RESPONSE}",
                    
                    "\n--- EXAMPLE 5 ---",
                    f"Trigger: {EXAMPLE_5_TRIGGER}\nDiagnose the image and provide the JSON correction.",
                    ex5_img,
                    f"Assistant Response:\n{EXAMPLE_5_RESPONSE}",
                    
                    "\n--- REAL SCENARIO ---",
                    "Based on the rules and examples above, analyze this final image and provide ONLY the JSON response.",
                    f"Trigger: {trigger_reason}",
                    live_img
                ]

                # Call the 3.1-pro-preview model
                response = client.models.generate_content(
                    model='gemini-3.1-pro-preview',
                    contents=prompt_contents,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json", # Forces strict JSON
                    )
                )

                # --- NEW ADDITION: PACING TO PREVENT API HANGS ---
                print("⏳ Pacing API to avoid RPM limits...")
                time.sleep(15)
                
                vlm_raw_output = response.text
                print(f"\n[Gemini Reasoning]:\n{vlm_raw_output}\n")
                
                corrections = json.loads(vlm_raw_output)
                
                # FIX: If Gemini wrapped the dictionary in a list [ { ... } ], extract the dictionary
                if isinstance(corrections, list):
                    corrections = corrections[0] if len(corrections) > 0 else {}
                
                # The model will output an empty JSON {} if it decides no correction is needed
                if not corrections:
                    print("✅ Gemini determined trajectory is actually safe (False Positive Override).")
                    vlm_intervened = "False Positive Override"
                else:
                    vlm_intervened = True
                    
                    # Look for A2
                    osc_delta = corrections.get("A2", corrections.get("A2_grasp", {}))
                    
                    if osc_delta:
                        # --- 1. UPDATE A2 (GRASP) ---
                        orig_a2_pos = np.array(trajectory_goals_3d["A2_grasp"][0])
                        orig_a2_quat = trajectory_goals_3d["A2_grasp"][1]
                        
                        delta_pos = np.array([osc_delta.get('dx', 0.0), osc_delta.get('dy', 0.0), osc_delta.get('dz', 0.0)])
                        
                        new_a2_pos = (orig_a2_pos + delta_pos).tolist()
                        
                        d_roll = np.radians(osc_delta.get('d_roll', 0.0))
                        d_pitch = np.radians(osc_delta.get('d_pitch', 0.0))
                        d_yaw = np.radians(osc_delta.get('d_yaw', 0.0))
                        
                        if np.any([d_roll, d_pitch, d_yaw]):
                            current_euler = R.from_quat(orig_a2_quat).as_euler('xyz')
                            new_a2_quat = R.from_euler('xyz', current_euler + [d_roll, d_pitch, d_yaw]).as_quat().tolist()
                        else:
                            new_a2_quat = orig_a2_quat
                            
                        trajectory_goals_3d["A2_grasp"] = (new_a2_pos, new_a2_quat)
                        
                        # --- 2. UPDATE A1 (PRE-GRASP) TO MATCH A2's X/Y ---
                        orig_a1_pos = np.array(trajectory_goals_3d["A1_pregrasp"][0])
                        new_a1_pos = [new_a2_pos[0], new_a2_pos[1], orig_a1_pos[2]]
                        trajectory_goals_3d["A1_pregrasp"] = (new_a1_pos, new_a2_quat)

            except Exception as e:
                print(f"⚠️ Gemini API Error: {e}")

        # ==========================================
        # EXECUTE AND EVALUATE (Agent 4)
        # ==========================================
        debug_video_dir = Path("debug_videos")
        debug_video_dir.mkdir(exist_ok=True)
        video_path = str(debug_video_dir / f"trial_{i}_intervention.mp4")
        video_env.video_start(path=video_path)

        original_step = task_generator.env.step
        def recording_step(action):
            step_results = original_step(action)
            video_env.obs = step_results[0]
            video_env.video_frame()
            return step_results
        task_generator.env.step = recording_step

        is_success = task_generator.execute_trajectory(trajectory_goals_3d)

        task_generator.env.step = original_step
        video_env.video_stop()

        if is_success:
            total_successes += 1
            print(f"\n✅ >>> TRIAL {i} RESULT: SUCCESS <<<")
        else:
            failed_trials.append(i)
            print(f"\n❌ >>> TRIAL {i} RESULT: FAILED <<<")

        # --- LOG THE RESULTS FOR THIS TRIAL ---
        trial_logs.append({
            "trial_idx": i,
            "success": is_success,
            "intervened": vlm_intervened,
            "trigger_reason": trigger_reason if not is_safe else "N/A",
            "baseline": baseline_trajectory,
            "final": trajectory_goals_3d,
            "vlm_raw": vlm_raw_output
        })

    # ==========================================
    # FORENSIC DELTA REPORT
    # ==========================================
    print("\n" + "="*80)
    print("  🔍 FORENSIC DELTA REPORT: ORIGINAL VS CORRECTED WAYPOINTS")
    print("="*80)
    
    for log in trial_logs:
        idx = log["trial_idx"]
        status = "✅ SUCCESS" if log["success"] else "❌ FAILED"
        
        if log["intervened"] == False:
            intervention_str = "No Intervention (Agent 1 Passed)"
        elif log["intervened"] == "False Positive Override":
            intervention_str = "VLM False Positive Override (Empty JSON)"
        else:
            intervention_str = "VLM Applied Corrections"
            
        print(f"\n--- TRIAL {idx} | {status} | {intervention_str} ---")
        
        if log["intervened"] == True:
            print(f"Trigger: {log['trigger_reason']}")
            # Compare A1 and A2
            for wp in ["A1_pregrasp", "A2_grasp"]:
                orig_pos = np.array(log["baseline"][wp][0])
                final_pos = np.array(log["final"][wp][0])
                
                orig_yaw = R.from_quat(log["baseline"][wp][1]).as_euler('xyz')[2]
                final_yaw = R.from_quat(log["final"][wp][1]).as_euler('xyz')[2]
                
                # Calculate Deltas (in centimeters for readability)
                d_pos = (final_pos - orig_pos) * 100
                d_yaw = np.degrees(final_yaw - orig_yaw)
                
                # Only print if something actually changed
                if np.any(np.abs(d_pos) > 0.1) or abs(d_yaw) > 1.0:
                    print(f"  {wp} Corrections:")
                    print(f"    Original : X={orig_pos[0]:.3f}, Y={orig_pos[1]:.3f}, Z={orig_pos[2]:.3f} | Yaw={np.degrees(orig_yaw):.1f}°")
                    print(f"    Final    : X={final_pos[0]:.3f}, Y={final_pos[1]:.3f}, Z={final_pos[2]:.3f} | Yaw={np.degrees(final_yaw):.1f}°")
                    print(f"    Delta    : dx={d_pos[0]:+.1f}cm, dy={d_pos[1]:+.1f}cm, dz={d_pos[2]:+.1f}cm | d_yaw={d_yaw:+.1f}°")

    print("\n" + "="*50)
    print("  FINAL MAS EVALUATION RESULTS")
    print("="*50)
    print(f"Total Trials: {cfg.num_times}")
    print(f"Successes:    {total_successes}")
    print(f"Failures:     {len(failed_trials)}")
    success_rate = (total_successes / cfg.num_times) * 100
    print(f"Success Rate: {success_rate:.2f}%")
    print("-" * 50)
    if failed_trials: print(f"🚨 FAILED TRIAL NUMBERS: {failed_trials}")
    else: print("🎉 ALL TRIALS SUCCESSFUL!")
    print("="*50 + "\n")

if __name__ == "__main__":
    eval()
