import os
os.environ["MUJOCO_GL"] = "egl"

import robosuite as suite
from robosuite.utils import camera_utils
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import cv2

# --- Import your new environment to register it ---
try:
    import robosuite.environments.manipulation.pick_place_clutter
except ImportError:
    print("=" * 80)
    print("ERROR: Could not import PickPlaceClutter environment.")
    print("Please make sure 'pick_place_clutter.py' is in 'robosuite/environments/manipulation/'")
    print("and you have added 'from . import pick_place_clutter' to that folder's __init__.py")
    print("=" * 80)
    exit()

import sys
import time
import json
import pprint
import signal
import datetime
import threading
import requests
import ast
import json_numpy
json_numpy.patch()

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from copy import deepcopy

import draccus
import torch
import torch.distributed as dist
import numpy as np
import tqdm
import contextlib

from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from merge import merge_lora

from generate_vla_dataset_visualize import VLADataGenerator

from visualize_dataset_and_affordances import visualize_dataset


# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VideoEnv():
    """
    A robosuite env with video saving
    """
    def __init__(self, env, init_obs, cfg):
        self.env = env
        self.cfg = cfg

        self.obs = init_obs
        self.robot_pos = self.obs["robot0_eef_pos"].copy()
        self.robot_quat = self._get_current_quat()
        self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)
        self._rec = {} # For video saving

        # 5. Camera parameters
        self.cam_width = self.env.camera_widths[0]
        self.cam_height = self.env.camera_heights[0]

    def _get_current_quat(self):
        return self.obs["robot0_eef_quat_site"].copy()  # SciPy-friendly [x,y,z,w]

    def video_frame(self, text=None):
        if not getattr(self, "_rec", None) or not self._rec["on"]:
            return

        H, W = self._rec["H"], self._rec["W"]
        cam = self._rec["camera"]

        # Render
        rgb = self.env.sim.render(camera_name=cam, height=H, width=W, depth=False)
        frame = cv2.flip(rgb, 0)

        # Ensure dtype + contiguity for VideoWriter
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

        # Optional overlays
        if text:
            cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)


        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bgr = np.ascontiguousarray(bgr)

        # Track frames-written to catch “empty” files
        self._rec["frames"] += 1
        self._rec["writer"].write(bgr)
        
    def video_start(self, path="eval.mp4", fps=30, H=256, W=256, camera_name="agentview"):
        self._rec = {"on": False, "path": path, "fps": fps, "H": H, "W": W, "camera": camera_name, "frames": 0}
        for fourcc_str in ("avc1", "mp4v", "XVID"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(path, fourcc, fps, (W, H), True)
            if writer.isOpened():
                self._rec.update({"writer": writer, "fourcc": fourcc_str, "on": True})
                break
        if not self._rec["on"]:
            raise RuntimeError(f"Failed to open VideoWriter for {path}. Install H.264 support or try .avi with XVID.")

        # Warm up the renderer once after previous resets
        _ = self.env.sim.render(camera_name=camera_name, height=H, width=W, depth=False)

        # Seed file with a few frames so it’s never near-empty
        for _ in range(3):
            self.video_frame("start")

    def video_stop(self):
        if getattr(self, "_rec", None) and self._rec.get("on", False):
            self._rec["writer"].release()
            print(f"[VIDEO] Saved to {self._rec['path']} (codec={self._rec.get('fourcc')}, frames={self._rec['frames']})")
            self._rec["on"] = False
    
    def step(self, action):
        """
        step the environment and record a video frame
        """
        # Ensure action is a writable numpy array before passing to robosuite
        action_np = np.array(action, dtype=np.float64, copy=True)
        if not action_np.flags.writeable:
            action_np = action_np.copy()

        self.obs, _, _, _ = self.env.step(action_np)
        self.video_frame()


@dataclass
class EvalConfig:
    # fmt: off
    exp_id: str = None                                              # Unique experiment ID (will be initialized if left None)
    exp_tag: str = None                                             # Extra tag to end onto the end of experiment ID string

    # Directory Paths
    output_dir: Path = Path("eval_output")                               # Path to directory to store model output
    
    # eval config
    num_times: int = 50


def agent_1_state_monitor(a1_pos, a2_pos, pregrasp_quat, grasp_quat):
    """
    Agent 1: The Semantic Ambiguity Trigger.
    Analyzes the VLA's predicted waypoints for safety thresholds before execution.
    Returns: (is_safe: bool, trigger_reason: str)
    """
    # 1. Edge of Vision Check (Y-axis distortion zone)
    if a1_pos[1] < -0.31:
        return False, f"Target is near the workspace edge (Y={a1_pos[1]:.3f}). Potential center-bias distortion."

    # 2. Kinematic Inconsistency Check (XY drift)
    # A1 and A2 should be a straight vertical drop. If they drift, the model is confused by clutter.
    xy_drift = np.linalg.norm(a1_pos[:2] - a2_pos[:2])
    if xy_drift > 0.02: # 2 cm threshold
        return False, f"Kinematic inconsistency detected. Pre-grasp and Grasp XY drift is {xy_drift*100:.1f}cm."

    # 3. Kinematic Inconsistency Check (Yaw drift - Symmetric Gripper)
    # Convert Quaternions to Euler to check wrist rotation
    a1_euler = R.from_quat(pregrasp_quat).as_euler('xyz')
    a2_euler = R.from_quat(grasp_quat).as_euler('xyz')
    
    # Calculate absolute difference
    yaw_diff = abs(a1_euler[2] - a2_euler[2])
    
    # Account for 180-degree (pi radian) gripper symmetry
    # This finds the shortest physical rotational distance (0 to 90 degrees)
    physical_yaw_diff = min(yaw_diff % np.pi, np.pi - (yaw_diff % np.pi))
    
    if physical_yaw_diff > 0.35: # Roughly 20 degrees
        return False, f"Orientation inconsistency detected. Wrist yaw physically shifts {np.degrees(physical_yaw_diff):.1f} degrees between hover and grasp."

    return True, "Trajectory looks semantically safe."


@draccus.wrap()
def eval(cfg: EvalConfig) -> None:
    # Set deterministic seeds for reproducibility
    seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"🌱 Random seed set to {seed} for reproducible evaluations.")
    
    task_generator = VLADataGenerator("eval_task")


    total_successes = 0
    failed_trials = []  # Keep track of exactly which trials fail
    
    for i in range(cfg.num_times):
        task_generator.instruction = task_generator.generate_instruction("cereal box", "target bin")
        print("\n" + "-"*40)
        print(f"🚀 STARTING TRIAL {i} / {cfg.num_times - 1}")
        print("-"*40)
        
        # save env to video env and eval model
        task_env = task_generator.env
        video_env = VideoEnv(task_env, task_env.reset(), cfg)
        
        # Grab the proper neutral pose BEFORE any movement
        neutral_pos = video_env.robot_pos.copy()
        neutral_quat = video_env.robot_quat.copy()

        print(f"Instruction: {task_generator.instruction}")

        img = video_env.obs["agentview_image"]
        img = cv2.flip(img, 0).astype(np.uint8)
        # save the image
        cv2.imwrite("eval_image_waypoint.png", img)

        try:
            waypoint_1_action = requests.post(
                "http://0.0.0.0:8000/act",
                json={"image": img.copy(), "instruction": task_generator.instruction, "predict_mode": "pregrasp"}
            ).json()

            waypoint_2_action = requests.post(
                "http://0.0.0.0:8000/act",
                json={"image": img.copy(), "instruction": task_generator.instruction, "predict_mode": "grasp"}
            ).json()

            waypoint_3_action = requests.post(
                "http://0.0.0.0:8000/act",
                json={"image": img.copy(), "instruction": task_generator.instruction, "predict_mode": "release"}
            ).json()
        except Exception as e:
            print(f"Network error communicating with server: {e}")
            failed_trials.append(i)
            continue

        a1_pos = np.array(waypoint_1_action[:3], dtype=np.float64, copy=True)
        a2_pos = np.array(waypoint_2_action[:3], dtype=np.float64, copy=True)
        a3_pos = np.array(waypoint_3_action[:3], dtype=np.float64, copy=True)

        # Convert predicted rotvec -> quaternion (robosuite expects quat for slerp)
        pregrasp_quat = R.from_euler('xyz', np.array(waypoint_1_action[3:6], dtype=np.float64, copy=True)).as_quat()
        grasp_quat = R.from_euler('xyz', np.array(waypoint_2_action[3:6], dtype=np.float64, copy=True)).as_quat()
        standard_place_quat = R.from_euler('xyz', np.array(waypoint_3_action[3:6], dtype=np.float64, copy=True)).as_quat()

        trajectory_goals_3d = {
            "A1_pregrasp": (a1_pos.tolist(), pregrasp_quat.tolist()),
            "A2_grasp": (a2_pos.tolist(), grasp_quat.tolist()),
            "A3_release": (a3_pos.tolist(), standard_place_quat.tolist()),
        }

        # ==========================================
        # MULTI-AGENT RECOVERY SYSTEM
        # ==========================================
        is_safe, trigger_reason = agent_1_state_monitor(a1_pos, a2_pos, pregrasp_quat, grasp_quat)

        if not is_safe:
            print("\n" + "!"*50)
            print("🛑 AGENT 1 TRIGGER: SIMULATION PAUSED")
            print(f"Reason: {trigger_reason}")
            print("!"*50)
            
            # ==========================================
            # GENERATE LIVE VISUALIZATION FOR HUMAN
            # ==========================================
            print("📸 Generating live visualization for human review...")
            
            # 1. Get Camera Matrices from MuJoCo
            cam_name = "agentview"
            K = camera_utils.get_camera_intrinsic_matrix(task_generator.env.sim, cam_name, task_generator.cam_height, task_generator.cam_width)
            T_wc = camera_utils.get_camera_extrinsic_matrix(task_generator.env.sim, cam_name)
            
            # 2. Format the 7-DoF waypoints exactly how the visualizer expects them
            waypoint_labels_predicted = {
                "A1_pregrasp": task_generator._get_7dof_pose(a1_pos, pregrasp_quat, -1.0),
                "A2_grasp":    task_generator._get_7dof_pose(a2_pos, grasp_quat,  1.0),
                "A3_release":  task_generator._get_7dof_pose(a3_pos, standard_place_quat, -1.0),
                "A4_home":     task_generator._get_7dof_pose(neutral_pos, neutral_quat, -1.0)
            }
            
            # 3. Save to a temporary debug folder and run the visualizer
            debug_dir = Path("debug_intervention")
            debug_dir.mkdir(exist_ok=True)
            task_generator.output_dir = debug_dir
            
            task_generator.save_sample(
                trial_idx=0,
                rgb_img=img,
                depth_img=task_generator.obs["agentview_depth"],
                instruction=task_generator.instruction,
                waypoint_labels=waypoint_labels_predicted,
                K=K,
                T_wc=T_wc
            )
            
            visualize_dataset(debug_dir, debug_dir, num_samples=1)
            
            print("👉 Open 'debug_intervention/vis_episode_00000.png' in your IDE to see the predicted waypoints!")
            # ==========================================
            
            # --- HUMAN API (Dialogue as a Sensor) ---
            print("\n👤 Human Intervention Required.")
            print("Target specific waypoints. Format: {'A2': {'dy': -0.05, 'd_yaw': -45.0}, 'A1': {'dy': -0.05, 'dz': 0.1}}")
            print("Or press Enter to bypass and let the robot attempt the unsafe trajectory.")
            
            human_input = input("Correction: ")
            
            if human_input.strip():
                try:
                    # Parse the nested dictionary
                    user_corrections = ast.literal_eval(human_input)
                    
                    # Map shortcuts to actual dictionary keys
                    wp_map = {"A1": "A1_pregrasp", "A2": "A2_grasp", "A3": "A3_release"}
                    
                    for wp_key, osc_delta in user_corrections.items():
                        actual_wp = wp_map.get(wp_key, wp_key)
                        
                        if actual_wp not in trajectory_goals_3d:
                            print(f"⚠️ Warning: Unknown waypoint '{wp_key}'. Skipping.")
                            continue
                            
                        # Extract current position and quaternion
                        current_pos = np.array(trajectory_goals_3d[actual_wp][0])
                        current_quat = trajectory_goals_3d[actual_wp][1]
                            
                        # 1. Calculate New Translation
                        delta_pos = np.array([osc_delta.get('dx', 0.0), osc_delta.get('dy', 0.0), osc_delta.get('dz', 0.0)])
                        new_pos = (current_pos + delta_pos).tolist()
                        
                        # 2. Calculate New Rotation
                        d_roll = np.radians(osc_delta.get('d_roll', 0.0))
                        d_pitch = np.radians(osc_delta.get('d_pitch', 0.0))
                        d_yaw = np.radians(osc_delta.get('d_yaw', 0.0))
                        delta_euler = np.array([d_roll, d_pitch, d_yaw])
                        
                        if np.any(delta_euler): # Only calculate if a rotation was requested
                            current_euler = R.from_quat(current_quat).as_euler('xyz')
                            new_euler = current_euler + delta_euler
                            new_quat = R.from_euler('xyz', new_euler).as_quat().tolist()
                        else:
                            new_quat = current_quat

                        # 3. ASSIGN THE NEW TUPLE (This fixes the bug!)
                        trajectory_goals_3d[actual_wp] = (new_pos, new_quat)

                        print(f"✅ Corrections applied to {actual_wp}: Shifted pos by {delta_pos}m, rotated by {[osc_delta.get('d_roll', 0), osc_delta.get('d_pitch', 0), osc_delta.get('d_yaw', 0)]} degrees.")
                except Exception as e:
                    print(f"⚠️ Failed to parse human input, resuming original trajectory. Error: {e}")
            else:
                print("⏩ Bypassing correction. Executing original trajectory...")
                
# ==========================================
        # 3. EXECUTE AND EVALUATE (Agent 4)
        # ==========================================
        # Ensure debug videos directory exists
        debug_video_dir = Path("debug_videos")
        debug_video_dir.mkdir(exist_ok=True)
        
        # Start recording
        video_path = str(debug_video_dir / f"trial_{i}_intervention.mp4")
        video_env.video_start(path=video_path)

        # --- THE FIX: Hijack the step function to force recording ---
        original_step = task_generator.env.step

        def recording_step(action):
            # 1. Run the actual physics step
            step_results = original_step(action)
            # 2. Update the observation and take a picture
            video_env.obs = step_results[0]
            video_env.video_frame()
            return step_results

        # Apply the patch
        task_generator.env.step = recording_step
        # -------------------------------------------------------------

        # Execute the corrected trajectory
        is_success = task_generator.execute_trajectory(trajectory_goals_3d)

        # Remove the patch so we don't break future trials
        task_generator.env.step = original_step

        # Stop recording and save
        video_env.video_stop()

        # Clearer per-trial results
        if is_success:
            total_successes += 1
            print(f"\n✅ >>> TRIAL {i} RESULT: SUCCESS <<<")
            print(f"🎥 Video saved to: {video_path}")
        else:
            failed_trials.append(i)
            print(f"\n❌ >>> TRIAL {i} RESULT: FAILED <<<")
            print(f"🎥 Debug Video saved to: {video_path}")

    # ==========================================
    # --- END OF RUN SUMMARY ---
    # ==========================================
    print("\n" + "="*50)
    print("  FINAL MAS EVALUATION RESULTS")
    print("="*50)
    print(f"Total Trials: {cfg.num_times}")
    print(f"Successes:    {total_successes}")
    print(f"Failures:     {len(failed_trials)}")
    
    success_rate = (total_successes / cfg.num_times) * 100
    print(f"Success Rate: {success_rate:.2f}%")
    print("-" * 50)
    
    if failed_trials:
        print(f"🚨 FAILED TRIAL NUMBERS: {failed_trials}")
    else:
        print("🎉 ALL TRIALS SUCCESSFUL!")
    print("="*50 + "\n")

if __name__ == "__main__":
    eval()
    
