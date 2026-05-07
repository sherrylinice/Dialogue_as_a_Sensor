import os
os.environ["MUJOCO_GL"] = "egl"

import robosuite as suite
from robosuite.utils import camera_utils
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import cv2
# --- 1. IMPORT camera_utils ---
from robosuite.utils import camera_utils

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

import os
import sys
import time
import json
import pprint
import signal
import datetime
import threading
import requests
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

# from generate_vla_dataset_trajectories import VLADataGenerator
from generate_vla_dataset_visualize import VLADataGenerator
from visualize_dataset_and_affordances import visualize_dataset

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VideoEnv():
    """
    A robosuite env with video saving
    """
    def __init__(self, env, obs, cfg):
        self.env = env
        self.cfg = cfg

        self.obs = obs
        self.robot_pos = self.obs["robot0_eef_pos"].copy()
        self.robot_quat = self._get_current_quat()
        self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)
        self._rec = {} # For video saving

        # 5. Camera parameters
        self.cam_width = self.env.camera_widths[0]
        self.cam_height = self.env.camera_heights[0]

    def _get_current_quat(self):
#        """Helper to consistently get the correct quaternion from observations."""
#        if "robot0_eef_quat_site" in self.obs:
#            return self.obs["robot0_eef_quat_site"].copy()
#        else:
#            return self.obs["robot0_eef_quat"].copy()

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

@draccus.wrap()
def eval(cfg: EvalConfig) -> None:
    # ==========================================
    # 0. SET DETERMINISTIC SEEDS
    # ==========================================
    seed = 34
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"🌱 Random seed set to {seed} for reproducible evaluations.")

    task_generator = VLADataGenerator("eval_task")
    num_trials = 50
    total_successes = 0
    
    # Create a directory to store our failure analysis
    fail_dir = Path("eval_failed_cases")
    fail_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*50)
    print(f"  STARTING FAILURE ANALYSIS EVALUATION: {num_trials} TRIALS")
    print("="*50 + "\n")

    for i in range(num_trials):
        print(f"\n--- TRIAL {i} / {num_trials - 1} ---")

        task_generator.instruction = task_generator.generate_instruction("cereal box", "target bin")
        print(f"Instruction: {task_generator.instruction}")

        task_env = task_generator.env
        video_env = VideoEnv(task_env, task_env.reset(), cfg)
        neutral_pos = task_generator.robot_pos.copy()
        neutral_quat = task_generator.robot_quat.copy()

        # Start Video Recording for this specific trial
        video_path = str(fail_dir / f"trial_{i}_video.mp4")
        video_env.video_start(path=video_path)

        img = video_env.obs["agentview_image"]
        img = cv2.flip(img, 0).astype(np.uint8)

        # ==========================================
        # 1. CALCULATE GROUND TRUTH (Before moving!)
        # ==========================================
        cereal_body_id = task_generator.env.obj_body_id["Cereal"]
        cereal_pos = task_generator.env.sim.data.body_xpos[cereal_body_id]
        cereal_mj_quat = task_generator.env.sim.data.body_xquat[cereal_body_id]
        target_bin_pos = task_generator.env.target_bin_placements[2]
        
        cereal_scipy_quat = np.array([cereal_mj_quat[1], cereal_mj_quat[2], cereal_mj_quat[3], cereal_mj_quat[0]])
        cereal_rotation = R.from_quat(cereal_scipy_quat)
        box_long_side_vector = cereal_rotation.as_matrix()[:, 0]
        x_axis_gripper = np.array([-box_long_side_vector[1], box_long_side_vector[0], 0])
        norm = np.linalg.norm(x_axis_gripper)
        x_axis_gripper = np.array([1., 0., 0.]) if norm < 1e-5 else x_axis_gripper / norm
        z_axis_gripper = np.array([0., 0., -1.])
        y_axis_gripper = np.cross(z_axis_gripper, x_axis_gripper)
        
        grasp_rotation_matrix = np.array([x_axis_gripper, y_axis_gripper, z_axis_gripper]).T
        grasp_quat = R.from_matrix(grasp_rotation_matrix).as_quat()
        standard_place_quat = R.from_euler('xyz', [180, 0, 90], degrees=True).as_quat()
        
        a1_pos_gt = cereal_pos + np.array([0, 0, 0.30])
        a2_pos_gt = cereal_pos + np.array([0, 0, 0.03])
        a3_pos_gt = target_bin_pos + np.array([0, 0, 0.10])
        
        waypoint_labels_gt = {
            "A1_pregrasp": task_generator._get_7dof_pose(a1_pos_gt, grasp_quat, -1.0),
            "A2_grasp":    task_generator._get_7dof_pose(a2_pos_gt, grasp_quat,  1.0),
            "A3_release":  task_generator._get_7dof_pose(a3_pos_gt, standard_place_quat, -1.0),
            "A4_home":     task_generator._get_7dof_pose(neutral_pos, neutral_quat, -1.0)
        }

        # ==========================================
        # 2. GET PREDICTIONS
        # ==========================================
        try:
            waypoint_1_action = requests.post("http://0.0.0.0:8001/act", json={"image": img.copy(), "instruction": task_generator.instruction, "predict_mode": "pregrasp"}).json()
            waypoint_2_action = requests.post("http://0.0.0.0:8001/act", json={"image": img.copy(), "instruction": task_generator.instruction, "predict_mode": "grasp"}).json()
            waypoint_3_action = requests.post("http://0.0.0.0:8001/act", json={"image": img.copy(), "instruction": task_generator.instruction, "predict_mode": "release"}).json()
        except Exception as e:
            print(f"Network error: {e}")
            video_env.video_stop()
            continue

        a1_pos_pred = np.array(waypoint_1_action[:3], dtype=np.float64, copy=True)
        a2_pos_pred = np.array(waypoint_2_action[:3], dtype=np.float64, copy=True)
        a3_pos_pred = np.array(waypoint_3_action[:3], dtype=np.float64, copy=True)

        pregrasp_quat_pred = R.from_euler('xyz', np.array(waypoint_1_action[3:6], dtype=np.float64, copy=True)).as_quat()
        grasp_quat_pred = R.from_euler('xyz', np.array(waypoint_2_action[3:6], dtype=np.float64, copy=True)).as_quat()
        standard_place_quat_pred = R.from_euler('xyz', np.array(waypoint_3_action[3:6], dtype=np.float64, copy=True)).as_quat()

        trajectory_goals_3d = {
            "A1_pregrasp": (a1_pos_pred.tolist(), pregrasp_quat_pred.tolist()),
            "A2_grasp": (a2_pos_pred.tolist(), grasp_quat_pred.tolist()),
            "A3_release": (a3_pos_pred.tolist(), standard_place_quat_pred.tolist()),
        }

        waypoint_labels_predicted = {
            "A1_pregrasp": task_generator._get_7dof_pose(a1_pos_pred, pregrasp_quat_pred, -1.0),
            "A2_grasp":    task_generator._get_7dof_pose(a2_pos_pred, grasp_quat_pred,  1.0),
            "A3_release":  task_generator._get_7dof_pose(a3_pos_pred, standard_place_quat_pred, -1.0),
            "A4_home":     task_generator._get_7dof_pose(neutral_pos, neutral_quat, -1.0)
        }

        # ==========================================
        # 3. EXECUTE AND EVALUATE (WITH VIDEO PATCH)
        # ==========================================
        # --- THE FIX: Hijack the step function to force recording ---
        original_step = task_generator.env.step

        def recording_step(action):
            step_results = original_step(action)
            video_env.obs = step_results[0]
            video_env.video_frame()
            return step_results

        task_generator.env.step = recording_step
        # -------------------------------------------------------------
        
        is_success = task_generator.execute_trajectory(trajectory_goals_3d)
        
        # Remove the patch so we don't break future trials
        task_generator.env.step = original_step
        video_env.video_stop()
        
        if is_success:
            print(">>> Trajectory SUCCESSFUL - Saving visualization logs <<<")
            total_successes += 1
            status_tag = "SUCCESS"
        else:
            print(">>> Trajectory FAILED - Saving visualization logs <<<")
            status_tag = "FAILED"
            
        # ==========================================
        # 4. SAVE VISUALIZATIONS FOR ALL TRIALS
        # ==========================================
        # Get camera matrices
        K = camera_utils.get_camera_intrinsic_matrix(task_generator.env.sim, "agentview", task_generator.cam_height, task_generator.cam_width)
        T_wc = camera_utils.get_camera_extrinsic_matrix(task_generator.env.sim, "agentview")
        
        # Create a directory for this specific trial's data (keeping it in fail_dir as requested)
        trial_data_dir = fail_dir / f"trial_data_{i}"
        trial_data_dir.mkdir(parents=True, exist_ok=True)
        task_generator.output_dir = trial_data_dir
        
        # Save Ground Truth (0) and Predicted (1)
        task_generator.save_sample(0, img, task_generator.obs["agentview_depth"], task_generator.instruction, waypoint_labels_gt, K, T_wc)
        task_generator.save_sample(1, img, task_generator.obs["agentview_depth"], task_generator.instruction, waypoint_labels_predicted, K, T_wc)
        
        # Generate the visual plots
        visualize_dataset(trial_data_dir, fail_dir, num_samples=2)
        
        # Rename the generated plot with the success/fail tag so it doesn't get overwritten
        try:
            os.rename(fail_dir / "vis_episode_00001.png", fail_dir / f"trial_{i}_{status_tag}_prediction_plot.png")
            os.remove(fail_dir / "vis_episode_00000.png") # Delete GT plot, just need the comparison
        except FileNotFoundError:
            pass

    # ==========================================
    # --- PRINT FINAL AVERAGE SUCCESS RATE ---
    # ==========================================

#        if is_success:
#            print(">>> Trajectory SUCCESSFUL <<<")
#            total_successes += 1
#            os.remove(video_path) # Delete successful videos to save space
#        else:
#            print(">>> Trajectory FAILED - Saving visualization logs <<<")
#            
#            # Save the visualization strictly for the failed run
#            K = camera_utils.get_camera_intrinsic_matrix(task_generator.env.sim, "agentview", task_generator.cam_height, task_generator.cam_width)
#            T_wc = camera_utils.get_camera_extrinsic_matrix(task_generator.env.sim, "agentview")
#            
#            # Save GT as trial 0, Predicted as trial 1 inside a specific fail folder
#            trial_fail_dir = fail_dir / f"fail_data_{i}"
#            trial_fail_dir.mkdir(parents=True, exist_ok=True)
#            task_generator.output_dir = trial_fail_dir
#            
#            task_generator.save_sample(0, img, task_generator.obs["agentview_depth"], task_generator.instruction, waypoint_labels_gt, K, T_wc)
#            task_generator.save_sample(1, img, task_generator.obs["agentview_depth"], task_generator.instruction, waypoint_labels_predicted, K, T_wc)
#            
#            visualize_dataset(trial_fail_dir, fail_dir, num_samples=2)
#            
#            # Rename the generated plot so it doesn't get overwritten
#            try:
#                os.rename(fail_dir / "vis_episode_00001.png", fail_dir / f"trial_{i}_prediction_plot.png")
#                os.remove(fail_dir / "vis_episode_00000.png") # Delete GT plot, just need the comparison
#            except FileNotFoundError:
#                pass

    # ==========================================
    # --- PRINT FINAL AVERAGE SUCCESS RATE ---
    # ==========================================
    success_rate = (total_successes / num_trials) * 100

    print("\n" + "="*50)
    print("  FINAL EVALUATION RESULTS")
    print("="*50)
    print(f"Total Trials   : {num_trials}")
    print(f"Total Successes: {total_successes}")
    print(f"Success Rate   : {success_rate:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    eval()
