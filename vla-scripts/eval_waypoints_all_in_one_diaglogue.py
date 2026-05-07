import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["MPLBACKEND"] = "Agg" # Protects the MuJoCo EGL context from Matplotlib

import robosuite as suite
from robosuite.utils import camera_utils
from scipy.spatial.transform import Rotation as R
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
import requests
import subprocess
import gc
import matplotlib.pyplot as plt
import json_numpy
json_numpy.patch()

from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy

import draccus
import torch
import numpy as np

from generate_vla_dataset_trajectories import VLADataGenerator
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

        # Camera parameters
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

        # Track frames-written to catch "empty" files
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

        # Seed file with a few frames so it's never near-empty
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
        action_np = np.array(action, dtype=np.float64, copy=True)
        if not action_np.flags.writeable:
            action_np = action_np.copy()

        self.obs, _, _, _ = self.env.step(action_np)
        self.video_frame()


@dataclass
class EvalConfig:
    exp_id: str = None
    exp_tag: str = None
    output_dir: Path = Path("eval_output")
    num_times: int = 50


@draccus.wrap()
def eval(cfg: EvalConfig) -> None:

    seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"🌱 Random seed set to {seed} for reproducible evaluations.")
    
    task_generator = VLADataGenerator("eval_task")
    total_successes = 0
    failed_trials = []
    trial_logs = []
    
    video_dir = Path("eval_videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(cfg.num_times):
        task_generator.instruction = task_generator.generate_instruction("cereal box", "target bin")
        print("\n" + "-"*40)
        print(f"🚀 STARTING TRIAL {i} / {cfg.num_times - 1}")
        print("-"*40)
        
        task_env = task_generator.env
        video_env = VideoEnv(task_env, task_env.reset(), cfg)
        neutral_quat = task_generator.robot_quat.copy()

        # Only Start video here, rendering handled by patched step function
        video_path = str(video_dir / f"trial_{i}_video.mp4")
        video_env.video_start(path=video_path)

        print(f"Instruction: {task_generator.instruction}")

        img = video_env.obs["agentview_image"]
        img = cv2.flip(img, 0).astype(np.uint8)

        try:
            # ADDED TIMEOUTS TO PREVENT FREEZING!
            waypoint_1_action = requests.post(
                "http://0.0.0.0:8001/act",
                json={"image": img.copy(), "instruction": task_generator.instruction, "predict_mode": "pregrasp"},
                timeout=45
            ).json()

            waypoint_2_action = requests.post(
                "http://0.0.0.0:8001/act",
                json={"image": img.copy(), "instruction": task_generator.instruction, "predict_mode": "grasp"},
                timeout=45
            ).json()

            waypoint_3_action = requests.post(
                "http://0.0.0.0:8001/act",
                json={"image": img.copy(), "instruction": task_generator.instruction, "predict_mode": "release"},
                timeout=45
            ).json()
        except Exception as e:
            print(f"Network error communicating with server: {e}")
            failed_trials.append(i)
            video_env.video_stop()
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
        
        baseline_trajectory = deepcopy(trajectory_goals_3d)
        vlm_intervened = False
        vlm_raw_output = "N/A"

        # ==========================================
        # AGENT 1: SAFETY MONITOR
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
            
            # Generate visual for agents
            cam_name = "agentview"
            K = camera_utils.get_camera_intrinsic_matrix(task_generator.env.sim, cam_name, task_generator.cam_height, task_generator.cam_width)
            T_wc = camera_utils.get_camera_extrinsic_matrix(task_generator.env.sim, cam_name)
            
            waypoint_labels_predicted = {
                "A1_pregrasp": task_generator._get_7dof_pose(a1_pos, pregrasp_quat, -1.0),
                "A2_grasp":    task_generator._get_7dof_pose(a2_pos, grasp_quat,  1.0),
                "A3_release":  task_generator._get_7dof_pose(a3_pos, standard_place_quat, -1.0),
                "A4_home":     task_generator._get_7dof_pose(video_env.robot_pos, video_env.robot_quat, -1.0)
            }
            
            debug_dir = Path("debug_intervention")
            debug_dir.mkdir(exist_ok=True)
            task_generator.output_dir = debug_dir
            
            task_generator.save_sample(
                trial_idx=0, trajectory=[], rgb_img=img, depth_img=task_generator.obs["agentview_depth"],
                instruction=task_generator.instruction, waypoint_labels=waypoint_labels_predicted,
                #K=K, T_wc=T_wc
            )
            visualize_dataset(debug_dir, debug_dir, num_samples=1)

            # ==========================================
            # AGENTS 2 & 3: LANGGRAPH DIALOGUE SENSOR
            # ==========================================
            print("\n🤖 Triggering Dialogue-as-a-Sensor LangGraph...")
            try:
                episode_path = "debug_intervention/episode_00000"
                output_json_path = "debug_intervention/langgraph_output.json"
                
                #safe_trigger = trigger_reason.replace(" ", "_").replace(".", "")
                # Map Agent 1's detailed reason to the strict CLI choices
                if "Target is near" in trigger_reason:
                    safe_trigger = "semantic_ambiguity"
                elif "Kinematic" in trigger_reason or "Orientation" in trigger_reason:
                    safe_trigger = "kinematic_collision"
                else:
                    safe_trigger = "unknown"

                cmd = [
                    sys.executable, "-m", "dialogue_sensor.cli",
                    "--episode_dir", episode_path,
                    "--trigger_reason", safe_trigger,
                    "--output_json", output_json_path,
                    "--max_turns", "2"
                ]
                
                # Run the multi-agent conversation
                subprocess.run(cmd, check=True)
                
                with open(output_json_path, 'r') as f:
                    langgraph_result = json.load(f)
                
                vlm_raw_output = json.dumps(langgraph_result, indent=2)
                print(f"\n[Agent 3 Final Output]:\n{vlm_raw_output}\n")
                
                corrections = langgraph_result.get("corrections", {})
                
                if not corrections:
                    print("✅ Agent 3 determined trajectory is actually safe (False Positive Override).")
                    vlm_intervened = "False Positive Override"
                else:
                    vlm_intervened = True
                    a1_delta = corrections.get("A1", {})
                    a2_delta = corrections.get("A2", {})
                    
                    if a1_delta or a2_delta:
                        # --- ISOLATE A2 CORRECTIONS ---
                        orig_a2_pos = np.array(trajectory_goals_3d["A2_grasp"][0])
                        orig_a2_quat = trajectory_goals_3d["A2_grasp"][1]
                        
                        a2_dx = a2_delta.get('dx', 0.0)
                        a2_dy = a2_delta.get('dy', 0.0)
                        a2_dz = a2_delta.get('dz', 0.0)
                        new_a2_pos = (orig_a2_pos + np.array([a2_dx, a2_dy, a2_dz])).tolist()
                        
                        a2_d_roll, a2_d_pitch, a2_d_yaw = a2_delta.get('d_roll', 0.0), a2_delta.get('d_pitch', 0.0), a2_delta.get('d_yaw', 0.0)
                        if np.any([a2_d_roll, a2_d_pitch, a2_d_yaw]):
                            current_a2_euler = R.from_quat(orig_a2_quat).as_euler('xyz')
                            new_a2_quat = R.from_euler('xyz', current_a2_euler + [a2_d_roll, a2_d_pitch, a2_d_yaw]).as_quat().tolist()
                        else:
                            new_a2_quat = orig_a2_quat
                            
                        trajectory_goals_3d["A2_grasp"] = (new_a2_pos, new_a2_quat)

                        # --- ISOLATE A1 CORRECTIONS ---
                        orig_a1_pos = np.array(trajectory_goals_3d["A1_pregrasp"][0])
                        orig_a1_quat = trajectory_goals_3d["A1_pregrasp"][1]
                        
                        # Only apply alignment if A1 wasn't explicitly given a correction by the VLM
                        if not a1_delta:
                            new_a1_pos = [new_a2_pos[0], new_a2_pos[1], orig_a1_pos[2]]
                            new_a1_quat = new_a2_quat
                            print("📏 Enforced Kinematic Alignment for A1.")
                        else:
                            a1_dx = a1_delta.get('dx', 0.0)
                            a1_dy = a1_delta.get('dy', 0.0)
                            a1_dz = a1_delta.get('dz', 0.0)
                            new_a1_pos = (orig_a1_pos + np.array([a1_dx, a1_dy, a1_dz])).tolist()
                            
                            a1_d_roll, a1_d_pitch, a1_d_yaw = a1_delta.get('d_roll', 0.0), a1_delta.get('d_pitch', 0.0), a1_delta.get('d_yaw', 0.0)
                            if np.any([a1_d_roll, a1_d_pitch, a1_d_yaw]):
                                current_a1_euler = R.from_quat(orig_a1_quat).as_euler('xyz')
                                new_a1_quat = R.from_euler('xyz', current_a1_euler + [a1_d_roll, a1_d_pitch, a1_d_yaw]).as_quat().tolist()
                            else:
                                new_a1_quat = orig_a1_quat
                            print("🔧 Applied explicit VLM Nudges to A1.")

                        trajectory_goals_3d["A1_pregrasp"] = (new_a1_pos, new_a1_quat)
                        print(f"🔧 Applied Agent 3 Nudges successfully.")

                # (Make sure to DELETE your old "ALWAYS ENFORCE ALIGNMENT" block below this!)
#                    if a1_delta or a2_delta:
#                        orig_a2_pos = np.array(trajectory_goals_3d["A2_grasp"][0])
#                        orig_a2_quat = trajectory_goals_3d["A2_grasp"][1]
#                        
#                        dx = a1_delta.get('dx', 0.0) + a2_delta.get('dx', 0.0)
#                        dy = a1_delta.get('dy', 0.0) + a2_delta.get('dy', 0.0)
#                        dz = a1_delta.get('dz', 0.0) + a2_delta.get('dz', 0.0)
#                        delta_pos = np.array([dx, dy, dz])
#                        new_a2_pos = (orig_a2_pos + delta_pos).tolist()
#                        
#                        # 🚨 NOTE: Agent 3 outputs RADIANS natively, no np.radians() needed here
#                        d_roll = a1_delta.get('d_roll', 0.0) + a2_delta.get('d_roll', 0.0)
#                        d_pitch = a1_delta.get('d_pitch', 0.0) + a2_delta.get('d_pitch', 0.0)
#                        d_yaw = a1_delta.get('d_yaw', 0.0) + a2_delta.get('d_yaw', 0.0)
#                        
#                        if np.any([d_roll, d_pitch, d_yaw]):
#                            current_euler = R.from_quat(orig_a2_quat).as_euler('xyz')
#                            new_quat = R.from_euler('xyz', current_euler + [d_roll, d_pitch, d_yaw]).as_quat().tolist()
#                        else:
#                            new_quat = orig_a2_quat
#                            
#                        trajectory_goals_3d["A2_grasp"] = (new_a2_pos, new_quat)
#                        print(f"🔧 Applied Agent 3 Nudges to A2.")

                # ALWAYS ENFORCE ALIGNMENT
                final_a2_pos = np.array(trajectory_goals_3d["A2_grasp"][0])
                final_a2_quat = trajectory_goals_3d["A2_grasp"][1]
                orig_a1_pos = np.array(trajectory_goals_3d["A1_pregrasp"][0])
                new_a1_pos = [final_a2_pos[0], final_a2_pos[1], orig_a1_pos[2]]
                trajectory_goals_3d["A1_pregrasp"] = (new_a1_pos, final_a2_quat)
                print("📏 Enforced Kinematic Alignment: A1 XY and Yaw synced to A2.")

            except subprocess.CalledProcessError as e:
                print(f"⚠️ LangGraph CLI Execution Failed: {e}")
            except Exception as e:
                print(f"⚠️ Error parsing Agent 3 output: {e}")

        # ==========================================
        # 2. MONKEY-PATCH THE STEP FUNCTION & EXECUTE
        # ==========================================
        original_step = task_generator.env.step
        
        def recording_step(action):
            step_results = original_step(action)
            video_env.obs = step_results[0]
            video_env.video_frame()
            return step_results
            
        task_generator.env.step = recording_step

        # Execute
        #is_success = task_generator.execute_trajectory(trajectory_goals_3d)
        # Execute
        is_success, _ = task_generator.execute_trajectory(trajectory_goals_3d)

        # 3. UNPATCH AND STOP VIDEO
        task_generator.env.step = original_step
        video_env.video_stop()

        if is_success:
            total_successes += 1
            print(f"\n✅ >>> TRIAL {i} RESULT: SUCCESS <<<")
        else:
            failed_trials.append(i)
            print(f"\n❌ >>> TRIAL {i} RESULT: FAILED <<<")

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
        # AGGRESSIVE GARBAGE COLLECTION
        # ==========================================
        plt.close('all')
        gc.collect()

    # ==========================================
    # --- END OF RUN SUMMARY ---
    # ==========================================
    print("\n" + "="*80)
    print("  🔍 FORENSIC DELTA REPORT")
    print("="*80)
    
    for log in trial_logs:
        idx = log["trial_idx"]
        status = "✅ SUCCESS" if log["success"] else "❌ FAILED"
        
        if log["intervened"] == False:
            intervention_str = "No Intervention"
        elif log["intervened"] == "False Positive Override":
            intervention_str = "VLM False Positive Override (Empty JSON)"
        else:
            intervention_str = "VLM Applied Corrections"
            
        print(f"\n--- TRIAL {idx} | {status} | {intervention_str} ---")
        
        if log["intervened"] == True:
            print(f"Trigger: {log['trigger_reason']}")
            for wp in ["A1_pregrasp", "A2_grasp"]:
                orig_pos = np.array(log["baseline"][wp][0])
                final_pos = np.array(log["final"][wp][0])
                
                orig_yaw = R.from_quat(log["baseline"][wp][1]).as_euler('xyz')[2]
                final_yaw = R.from_quat(log["final"][wp][1]).as_euler('xyz')[2]
                
                d_pos = (final_pos - orig_pos) * 100
                d_yaw = np.degrees(final_yaw - orig_yaw)
                
                if np.any(np.abs(d_pos) > 0.1) or abs(d_yaw) > 1.0:
                    print(f"  {wp} Deltas: dx={d_pos[0]:+.1f}cm, dy={d_pos[1]:+.1f}cm, dz={d_pos[2]:+.1f}cm | d_yaw={d_yaw:+.1f}°")

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
