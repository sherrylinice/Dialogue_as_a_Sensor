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

from generate_vla_dataset_trajectories import VLADataGenerator


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
    
    # eval config
    num_times: int = 50


@draccus.wrap()
def eval(cfg: EvalConfig) -> None:

    seed = 34
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"🌱 Random seed set to {seed} for reproducible evaluations.")
    task_generator = VLADataGenerator("eval_task")

    total_successes = 0
    failed_trials = []  # Keep track of exactly which trials fail
    
    # --- NEW: Create a directory for baseline videos ---
    video_dir = Path("eval_baseline_videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(cfg.num_times):
        task_generator.instruction = task_generator.generate_instruction("cereal box", "target bin")
        print("\n" + "-"*40)
        print(f"🚀 STARTING TRIAL {i} / {cfg.num_times - 1}")
        print("-"*40)
        
        # save env to video env and eval model
        task_env = task_generator.env
        video_env = VideoEnv(task_env, task_env.reset(), cfg)
        neutral_quat = task_generator.robot_quat.copy()

        # ==========================================
        # 1. START VIDEO IMMEDIATELY
        # ==========================================
        video_path = str(video_dir / f"trial_{i}_baseline.mp4")
        video_env.video_start(path=video_path)

        print(f"Instruction: {task_generator.instruction}")

        img = video_env.obs["agentview_image"]
        img = cv2.flip(img, 0).astype(np.uint8)
        
        # --- DEBUG CODE: Prove the seed works ---
        debug_frames_dir = Path("debug_initial_frames")
        debug_frames_dir.mkdir(exist_ok=True)
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(debug_frames_dir / f"trial_{i}_seed_check.png"), bgr_img)
        # ----------------------------------------------

        try:
            waypoint_1_action = requests.post(
                #"https://unifiedvla498.ngrok.dev/act",
                "http://0.0.0.0:8001/act",
                json={"image": img.copy(), "instruction": task_generator.instruction, "predict_mode": "pregrasp"}
            ).json()

            waypoint_2_action = requests.post(
                #"https://unifiedvla498.ngrok.dev/act",
                "http://0.0.0.0:8001/act",
                json={"image": img.copy(), "instruction": task_generator.instruction, "predict_mode": "grasp"}
            ).json()

            waypoint_3_action = requests.post(
                #"https://unifiedvla498.ngrok.dev/act",
                "http://0.0.0.0:8001/act",
                json={"image": img.copy(), "instruction": task_generator.instruction, "predict_mode": "release"}
            ).json()
        except Exception as e:
            print(f"Network error communicating with server: {e}")
            failed_trials.append(i)
            video_env.video_stop() # Clean up video on failure
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

        # ==========================================
        # 2. MONKEY-PATCH THE STEP FUNCTION
        # ==========================================
        original_step = task_generator.env.step
        
        def recording_step(action):
            step_results = original_step(action)
            video_env.obs = step_results[0]
            video_env.video_frame()
            return step_results
            
        task_generator.env.step = recording_step
        # ------------------------------------------

        # Execute
        is_success, trajectory = task_generator.execute_trajectory(trajectory_goals_3d)

        # ==========================================
        # 3. UNPATCH AND STOP VIDEO
        # ==========================================
        task_generator.env.step = original_step
        video_env.video_stop()
        # ------------------------------------------

        if is_success:
            total_successes += 1
            print(f"\n✅ >>> TRIAL {i} RESULT: SUCCESS <<<")
        else:
            failed_trials.append(i)
            print(f"\n❌ >>> TRIAL {i} RESULT: FAILED <<<")

    # ==========================================
    # --- END OF RUN SUMMARY ---
    # ==========================================
    print("\n" + "="*50)
    print("  FINAL BASELINE EVALUATION RESULTS")
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
