"""
Stuck-State Dataset Generation Script for Robosuite

This script extends the standard VLA dataset generator to capture episodes
where the robot's hard-coded trajectory either fails or visibly gets stuck.
The output is consumed by the multi-agent dialogue system (Agent 2 / NLG +
the simulated user) to test conversational disambiguation.

What this script saves per episode (success or failure):
    - image_initial.png         : RGB scene at t=0 (before motion).
    - image_stuck.png           : RGB frame captured at the most relevant
                                  "stuck" moment of the trajectory (or, if
                                  no kinematic stuck event was detected, the
                                  frame at the grasp-descend phase, which is
                                  where occlusion / collision is most likely).
    - image_phase_<phase>.png   : Optional per-phase frames (when
                                  --save_all_phases is set).
    - metadata.json             : {
            "instruction": str,                  (same as parent generator)
            "waypoints": {...},                  (same as parent generator)
            "success": bool,                     (did the parent's success
                                                  check pass?)
            "trigger_reason": str,               ("kinematic_collision" if
                                                  any stuck event detected,
                                                  "semantic_ambiguity" if
                                                  the trajectory completed
                                                  but failed the success
                                                  check, otherwise "none")
            "stuck_phase": str | null,           (which trajectory phase
                                                  produced the stuck event)
            "kinematic_state": {                 (snapshot at the stuck
                                                  moment / end of trajectory)
                "eef_pos":          [x, y, z],
                "eef_quat_site":    [x, y, z, w],
                "joint_vel":        [...7],
                "command_pos":      [x, y, z],
                "command_quat":     [x, y, z, w],
                "command_gripper":  float,
                "pos_error":        float (meters),
                "eef_speed":        float (m/s, finite-difference)
            },
            "scene_objects": [str, ...],         (Cereal/Milk/.../ClutterBottle/...
                                                  whose position is in the
                                                  bin1 workspace, for debug)
        }

By default we always save every trial, and we additionally save trials whose
trajectory was deliberately deformed to provoke a stuck moment (when
--inject_obstacles is used). The default behaviour is to run the standard
trajectory and let the natural clutter cause failures.

Example usage:
    export MUJOCO_GL=egl
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

    python generate_stuck_dataset.py \
        --output_dir ./my_stuck_data \
        --num_trials 30 \
        --num_videos 3 \
        --start_index 0
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# Reuse the existing VLA data generator; this gives us env setup, controller
# config, helper movement primitives and the dynamic grasp logic for free.
from generate_vla_dataset import VLADataGenerator
from robosuite.utils import camera_utils


# Distance (m) above which we consider a "move_to_pose" call to have failed
# kinematically (the EEF stalled before reaching its target). 5 cm is a
# reasonable threshold given the workspace scale and the controller's typical
# steady-state error (~1 cm).
STUCK_POS_TOLERANCE_M = 0.05

# Speed (m/s) below which we consider the EEF effectively stationary while a
# non-zero command is being issued. Together with a non-trivial pos_error
# this is the kinematic-collision trigger described in the project overview.
STUCK_SPEED_THRESHOLD_M_PER_S = 0.005


class StuckStateGenerator(VLADataGenerator):
    """Variant of VLADataGenerator that always saves trials and instruments
    the trajectory to capture a "stuck" frame and kinematic state."""

    def __init__(self, output_dir, save_all_phases=False, inject_obstacles=False):
        super().__init__(output_dir=output_dir)
        self.save_all_phases = save_all_phases
        self.inject_obstacles = inject_obstacles
        # Per-trial scratch state, populated inside run_generation_loop.
        self._captured_phases = {}      # phase_name -> rgb frame (HxWx3 uint8)
        self._stuck_event = None        # dict | None
        self._last_command = None       # (target_pos, target_quat, gripper)

        # Camera intrinsics + extrinsics. The agentview camera is fixed across
        # resets in PickPlaceClutter, so we capture these once and re-use them
        # for every trial. The agent-system uses these to project the planned
        # waypoints (A1/A2/A3) into pixel space and overlay the action plan
        # on the agent's view.
        self.cam_K = camera_utils.get_camera_intrinsic_matrix(
            self.env.sim, "agentview", self.cam_height, self.cam_width
        )
        self.cam_T_wc = camera_utils.get_camera_extrinsic_matrix(
            self.env.sim, "agentview"
        )

    # ------------------------------------------------------------------
    # Image capture
    # ------------------------------------------------------------------
    def _render_agentview(self):
        """Render a fresh RGB frame from the agentview camera, regardless of
        whether the agentview observable is currently enabled.

        The parent script disables agentview during stepping for performance,
        so we go straight to sim.render here."""
        rgb = self.env.sim.render(
            camera_name="agentview",
            height=self.cam_height,
            width=self.cam_width,
            depth=False,
        )
        # MuJoCo origin is bottom-left, like the saved initial image.
        return cv2.flip(rgb, 0)

    def _capture_phase(self, phase_name):
        """Render and store the agentview RGB at the given trajectory phase."""
        try:
            self._captured_phases[phase_name] = self._render_agentview()
        except Exception as e:  # pragma: no cover - safety net for render hiccups
            print(f"  [capture] WARN: failed to render phase '{phase_name}': {e}")

    # ------------------------------------------------------------------
    # Stuck detection
    # ------------------------------------------------------------------
    def _kinematic_state(self, target_pos, target_quat, gripper):
        """Snapshot the current kinematic state for logging."""
        eef_pos = self.obs["robot0_eef_pos"].copy()
        joint_vel = self.obs.get("robot0_joint_vel", np.zeros(7))
        # We approximate EEF speed via the recently-computed finite difference
        # if available; fall back to joint-velocity norm as a rough proxy.
        eef_speed = float(getattr(self, "_last_eef_speed", np.linalg.norm(joint_vel)))
        pos_error = float(np.linalg.norm(np.asarray(target_pos) - eef_pos))
        return {
            "eef_pos": eef_pos.tolist(),
            "eef_quat_site": self._get_current_quat().tolist(),
            "joint_vel": np.asarray(joint_vel).tolist(),
            "command_pos": np.asarray(target_pos).tolist(),
            "command_quat": np.asarray(target_quat).tolist(),
            "command_gripper": float(gripper),
            "pos_error": pos_error,
            "eef_speed": eef_speed,
        }

    def _maybe_record_stuck(self, phase_name, target_pos, target_quat, gripper, pos_error, eef_speed):
        """Record the first stuck event we see (one per trial)."""
        if self._stuck_event is not None:
            return
        if pos_error <= STUCK_POS_TOLERANCE_M and eef_speed >= STUCK_SPEED_THRESHOLD_M_PER_S:
            return
        # Two ways to be "stuck":
        #   (a) EEF reached steady state but is still meaningfully off-target
        #       (pos_error > tol AND eef_speed < threshold) -> kinematic stall.
        #   (b) Command non-zero but motion is glacial AND off-target.
        if pos_error > STUCK_POS_TOLERANCE_M and eef_speed < STUCK_SPEED_THRESHOLD_M_PER_S:
            print(
                f"  [stuck] kinematic stall at phase '{phase_name}' "
                f"(pos_error={pos_error:.3f} m, eef_speed={eef_speed:.4f} m/s)"
            )
            self._stuck_event = {
                "phase": phase_name,
                "kinematic_state": self._kinematic_state(target_pos, target_quat, gripper),
            }
            # Capture an image at the stuck moment.
            self._capture_phase(f"stuck_{phase_name}")

    # ------------------------------------------------------------------
    # Movement override: same as parent's move_to_pose, but with extra
    # bookkeeping so we can compute EEF speed and detect stalls without
    # touching the parent code.
    # ------------------------------------------------------------------
    def move_to_pose(self, target_pos, target_quat, gripper, count, time_for_residual_movement=10):
        rotations = R.from_quat([self.robot_quat, target_quat])
        slerp = Slerp([0, 1], rotations)

        prev_eef_pos = self.obs["robot0_eef_pos"].copy()
        # control_freq (Hz) -> dt per env.step
        dt = 1.0 / float(getattr(self.env, "control_freq", 20))

        for i in range(1, count + 1):
            next_target_pos = (target_pos - self.robot_pos) * i / count + self.robot_pos
            next_target_quat = slerp(float(i) / count).as_quat()
            action = np.concatenate(
                [
                    next_target_pos,
                    R.from_quat(next_target_quat).as_rotvec(degrees=False),
                    [gripper],
                ]
            )
            self.obs, _, _, _ = self.env.step(action)
            self._video_frame("Moving to pose (smooth)")

            cur_eef_pos = self.obs["robot0_eef_pos"]
            self._last_eef_speed = float(np.linalg.norm(cur_eef_pos - prev_eef_pos) / dt)
            prev_eef_pos = cur_eef_pos.copy()

        for _ in range(time_for_residual_movement):
            action = np.concatenate(
                [
                    target_pos,
                    R.from_quat(target_quat).as_rotvec(degrees=False),
                    [gripper],
                ]
            )
            self.obs, _, _, _ = self.env.step(action)

            cur_eef_pos = self.obs["robot0_eef_pos"]
            self._last_eef_speed = float(np.linalg.norm(cur_eef_pos - prev_eef_pos) / dt)
            prev_eef_pos = cur_eef_pos.copy()

        self.robot_pos = self.obs["robot0_eef_pos"].copy()
        self.robot_quat = self._get_current_quat()
        self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)
        self._last_command = (np.asarray(target_pos), np.asarray(target_quat), float(gripper))

    # ------------------------------------------------------------------
    # Trajectory: same phases as the parent, but with capture + stuck
    # checks after each one.
    # ------------------------------------------------------------------
    def execute_trajectory_with_capture(self, trajectory_goals_3d):
        """Run the trajectory, capturing per-phase frames and stuck events.

        Returns (is_success, completed_without_exception)."""
        completed = True
        try:
            hover_pos_cereal = np.array(trajectory_goals_3d["A1_pregrasp"][0])
            grasp_pos_cereal = np.array(trajectory_goals_3d["A2_grasp"][0])
            place_pos_bin = np.array(trajectory_goals_3d["A3_release"][0])
            grasp_quat = np.array(trajectory_goals_3d["A1_pregrasp"][1])
            place_quat = np.array(trajectory_goals_3d["A3_release"][1])

            target_bin_pos = self.env.target_bin_placements[2]
            hover_height = hover_pos_cereal[2] - grasp_pos_cereal[2]
            hover_pos_bin = target_bin_pos + np.array([0, 0, hover_height])

            neutral_pos = self.robot_pos.copy()
            neutral_quat = self.robot_quat.copy()

            self.move_gripper(-1, count=20)

            self._capture_phase("home")

            # Phase 1: pre-grasp position with home orientation
            print("  [phase] pre_grasp_position")
            self.move_to_pose(hover_pos_cereal, neutral_quat, gripper=-1.0, count=60)
            self._capture_phase("pre_grasp_position")
            self._maybe_record_stuck(
                "pre_grasp_position",
                hover_pos_cereal, neutral_quat, -1.0,
                pos_error=float(np.linalg.norm(hover_pos_cereal - self.obs["robot0_eef_pos"])),
                eef_speed=float(getattr(self, "_last_eef_speed", 0.0)),
            )

            # Phase 2: rotate to grasp orientation (in air)
            print("  [phase] pre_grasp_orient")
            self.move_to_pose(hover_pos_cereal, grasp_quat, gripper=-1.0, count=40)
            self._capture_phase("pre_grasp_orient")
            self._maybe_record_stuck(
                "pre_grasp_orient",
                hover_pos_cereal, grasp_quat, -1.0,
                pos_error=float(np.linalg.norm(hover_pos_cereal - self.obs["robot0_eef_pos"])),
                eef_speed=float(getattr(self, "_last_eef_speed", 0.0)),
            )

            # Phase 3: descend to grasp -- this is where collisions with
            # clutter most often manifest, so we always capture this frame
            # and treat it as the default "stuck" image if nothing earlier
            # produced a stuck event.
            print("  [phase] grasp_descend")
            self.move_to_pose(grasp_pos_cereal, grasp_quat, gripper=-1.0, count=50)
            self._capture_phase("grasp_descend")
            self._maybe_record_stuck(
                "grasp_descend",
                grasp_pos_cereal, grasp_quat, -1.0,
                pos_error=float(np.linalg.norm(grasp_pos_cereal - self.obs["robot0_eef_pos"])),
                eef_speed=float(getattr(self, "_last_eef_speed", 0.0)),
            )

            self.move_gripper(1, count=20)
            self._capture_phase("grasp_close")

            # Phase 4: lift
            print("  [phase] lift")
            self.move_to_pose(hover_pos_cereal, grasp_quat, gripper=1.0, count=50)
            self._capture_phase("lift")
            self._maybe_record_stuck(
                "lift",
                hover_pos_cereal, grasp_quat, 1.0,
                pos_error=float(np.linalg.norm(hover_pos_cereal - self.obs["robot0_eef_pos"])),
                eef_speed=float(getattr(self, "_last_eef_speed", 0.0)),
            )

            # Phase 5: rotate to place orientation
            self.move_to_pose(hover_pos_cereal, place_quat, gripper=1.0, count=40)

            # Phase 6: move to bin
            print("  [phase] move_to_bin")
            self.move_to_pose(hover_pos_bin, place_quat, gripper=1.0, count=70)
            self._capture_phase("move_to_bin")

            # Phase 7: release
            self.move_to_pose(place_pos_bin, place_quat, gripper=1.0, count=50)
            self.move_gripper(-1, count=20)
            self._capture_phase("release")

            # Retract + home
            self.move_to_pose(hover_pos_bin, place_quat, gripper=-1.0, count=50)
            self.move_to_pose(neutral_pos, neutral_quat, gripper=-1.0, count=60)

            print("Trajectory execution completed.")
            return self.check_success(target_obj_name="Cereal"), completed
        except Exception as e:
            print(f"An error occurred during trajectory execution: {e}")
            self._video_frame(f"CRITICAL ERROR: {e}")
            self._video_stop()
            # On exception, capture whatever the current view is.
            self._capture_phase("exception")
            return False, False

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def _scene_objects_summary(self):
        """List which task + clutter objects landed inside the source bin
        (useful for the dialogue agents to know what's actually in scene)."""
        names = []
        try:
            bin1_x, bin1_y, _ = self.env.bin1_pos
            half_x = self.env.bin_size[0] / 2.0
            half_y = self.env.bin_size[1] / 2.0
            for obj_name, body_id in self.env.obj_body_id.items():
                if "visual" in obj_name.lower():
                    continue
                pos = self.env.sim.data.body_xpos[body_id]
                if (
                    bin1_x - half_x < pos[0] < bin1_x + half_x
                    and bin1_y - half_y < pos[1] < bin1_y + half_y
                ):
                    names.append(obj_name)
        except Exception as e:  # pragma: no cover
            print(f"  [scene] WARN: failed to enumerate scene objects: {e}")
        return names

    def save_stuck_sample(
        self,
        trial_idx,
        initial_rgb,
        instruction,
        waypoint_labels,
        is_success,
        completed_without_exception,
    ):
        episode_dir = os.path.join(self.output_dir, f"episode_{trial_idx:05d}")
        os.makedirs(episode_dir, exist_ok=True)

        # Initial frame (mirrors the parent generator).
        cv2.imwrite(
            os.path.join(episode_dir, "image_initial.png"),
            cv2.cvtColor(initial_rgb, cv2.COLOR_RGB2BGR),
        )
        # Backwards compatible alias - the agent system can read either.
        cv2.imwrite(
            os.path.join(episode_dir, "image_rgb.png"),
            cv2.cvtColor(initial_rgb, cv2.COLOR_RGB2BGR),
        )

        # Pick the "stuck" frame:
        #   1. if a kinematic stall was recorded, use the frame captured at
        #      that exact moment;
        #   2. else if the trajectory completed but the task failed, use the
        #      grasp_descend frame (where occlusion / wrong-object grasping
        #      typically happens);
        #   3. else use the lift frame (final state), which is fine for
        #      successful trials too.
        def _pick_frame(*keys):
            for k in keys:
                frame = self._captured_phases.get(k)
                if frame is not None:
                    return frame
            return None

        stuck_phase = None
        if self._stuck_event is not None:
            stuck_phase = self._stuck_event["phase"]
            stuck_frame = _pick_frame(f"stuck_{stuck_phase}", stuck_phase)
        elif not is_success:
            stuck_phase = "grasp_descend"
            stuck_frame = _pick_frame("grasp_descend")
        else:
            stuck_phase = None
            stuck_frame = _pick_frame("grasp_descend", "lift")

        if stuck_frame is not None:
            cv2.imwrite(
                os.path.join(episode_dir, "image_stuck.png"),
                cv2.cvtColor(stuck_frame, cv2.COLOR_RGB2BGR),
            )

        # Optionally save every captured phase for debugging.
        if self.save_all_phases:
            for phase, frame in self._captured_phases.items():
                cv2.imwrite(
                    os.path.join(episode_dir, f"image_phase_{phase}.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                )

        if is_success:
            trigger_reason = "none"
        elif self._stuck_event is not None:
            trigger_reason = "kinematic_collision"
        else:
            # Trajectory finished but the success check failed — most often
            # this is the cereal slipping out, ending up outside the bin,
            # or the wrong object being grabbed. Treat as semantic ambiguity
            # so the dialogue agent looks at the scene rather than at a
            # collision pose.
            trigger_reason = "semantic_ambiguity"

        kinematic_state = (
            self._stuck_event["kinematic_state"]
            if self._stuck_event is not None
            else None
        )

        metadata = {
            "instruction": instruction,
            "waypoints": waypoint_labels,
            "success": bool(is_success),
            "trajectory_completed": bool(completed_without_exception),
            "trigger_reason": trigger_reason,
            "stuck_phase": stuck_phase,
            "kinematic_state": kinematic_state,
            "scene_objects": self._scene_objects_summary(),
            # Camera matrices for waypoint projection (matches the keys
            # consumed by visualize_dataset_and_affordances.py and the
            # agent-system overlay renderer).
            "camera_intrinsics": self.cam_K.tolist(),
            "camera_extrinsics_wc": self.cam_T_wc.tolist(),
            "camera_image_size": [self.cam_height, self.cam_width],
        }
        with open(os.path.join(episode_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        return episode_dir

    # ------------------------------------------------------------------
    # Generation loop (mirrors parent, but always saves).
    # ------------------------------------------------------------------
    def run_generation_loop(self, num_trials, num_videos, start_index=0):
        print(
            f"Starting STUCK-state data generation for {num_trials} trials, "
            f"beginning at index {start_index}..."
        )
        success_count = 0
        stuck_count = 0
        saved_count = 0

        for i in range(start_index, start_index + num_trials):
            trial_label = f"Trial {i + 1 - start_index} / {num_trials} (episode_{i:05d})"
            print(f"\n--- {trial_label} ---")

            # Reset per-trial scratch state.
            self._captured_phases = {}
            self._stuck_event = None
            self._last_eef_speed = 0.0
            self._last_command = None

            save_video_this_trial = (i - start_index) < num_videos
            if save_video_this_trial:
                video_path = os.path.join(
                    self.output_dir, f"debug_video_trial_{i:05d}.mp4"
                )
                self._video_start(path=video_path)

            # Re-enable observables before reset so we get a fresh image+depth.
            for _obs_name in ("agentview_image", "agentview_depth"):
                if _obs_name in self.env._observables:
                    self.env.modify_observable(
                        observable_name=_obs_name,
                        attribute="enabled",
                        modifier=True,
                    )
            self.obs = self.env.reset()
            self.robot_pos = self.obs["robot0_eef_pos"].copy()
            self.robot_quat = self._get_current_quat()
            self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)

            neutral_pos = self.robot_pos.copy()
            neutral_quat = self.robot_quat.copy()

            rgb_image_raw = self.obs["agentview_image"]
            depth_image_raw = self.obs.get("agentview_depth")

            # Disable agentview observables during stepping (parent's
            # GPU-memory workaround); we re-render via sim.render() ourselves
            # whenever we need a per-phase frame.
            for _obs_name in ("agentview_image", "agentview_depth"):
                if _obs_name in self.env._observables:
                    self.env.modify_observable(
                        observable_name=_obs_name,
                        attribute="enabled",
                        modifier=False,
                    )

            rgb_image = cv2.flip(rgb_image_raw, 0)
            if depth_image_raw is not None:
                depth_image_raw = np.nan_to_num(
                    depth_image_raw, nan=1.0, posinf=1.0, neginf=0.0
                )
                depth_image_raw = np.clip(depth_image_raw, 0.0, 1.0)
                _ = camera_utils.get_real_depth_map(self.env.sim, depth_image_raw)

            # Compute waypoints (copy of parent logic).
            try:
                cereal_body_id = self.env.obj_body_id["Cereal"]
                cereal_pos = self.env.sim.data.body_xpos[cereal_body_id]
                cereal_mj_quat = self.env.sim.data.body_xquat[cereal_body_id]
                target_bin_pos = self.env.target_bin_placements[2]
                target_obj_name = "cereal box"
                receptacle_name = "target bin"
            except Exception as e:
                print(f"ERROR: Could not find objects in scene. Skipping trial. {e}")
                if save_video_this_trial:
                    self._video_stop()
                continue

            instruction = self.generate_instruction(target_obj_name, receptacle_name)
            print(f"Generated Instruction: {instruction}")

            # Dynamic grasp orientation (same math as parent).
            cereal_scipy_quat = np.array(
                [
                    cereal_mj_quat[1],
                    cereal_mj_quat[2],
                    cereal_mj_quat[3],
                    cereal_mj_quat[0],
                ]
            )
            cereal_rotation = R.from_quat(cereal_scipy_quat)
            box_long_side_vector = cereal_rotation.as_matrix()[:, 0]
            x_axis_gripper = np.array(
                [-box_long_side_vector[1], box_long_side_vector[0], 0]
            )
            norm = np.linalg.norm(x_axis_gripper)
            if norm < 1e-5:
                x_axis_gripper = np.array([1.0, 0.0, 0.0])
            else:
                x_axis_gripper /= norm
            z_axis_gripper = np.array([0.0, 0.0, -1.0])
            y_axis_gripper = np.cross(z_axis_gripper, x_axis_gripper)
            grasp_rotation_matrix = np.array(
                [x_axis_gripper, y_axis_gripper, z_axis_gripper]
            ).T
            grasp_quat = R.from_matrix(grasp_rotation_matrix).as_quat()
            standard_place_quat = R.from_euler("xyz", [180, 0, 90], degrees=True).as_quat()

            a1_pos = cereal_pos + np.array([0, 0, 0.30])
            a2_pos = cereal_pos + np.array([0, 0, 0.03])
            a3_pos = target_bin_pos + np.array([0, 0, 0.10])
            a4_pos = neutral_pos

            waypoint_labels = {
                "A1_pregrasp": self._get_7dof_pose(a1_pos, grasp_quat, -1.0),
                "A2_grasp": self._get_7dof_pose(a2_pos, grasp_quat, 1.0),
                "A3_release": self._get_7dof_pose(a3_pos, standard_place_quat, -1.0),
                "A4_home": self._get_7dof_pose(a4_pos, neutral_quat, -1.0),
            }
            trajectory_goals_3d = {
                "A1_pregrasp": (a1_pos.tolist(), grasp_quat.tolist()),
                "A2_grasp": (a2_pos.tolist(), grasp_quat.tolist()),
                "A3_release": (a3_pos.tolist(), standard_place_quat.tolist()),
            }

            is_success, completed = self.execute_trajectory_with_capture(
                trajectory_goals_3d
            )

            episode_dir = self.save_stuck_sample(
                trial_idx=i,
                initial_rgb=rgb_image,
                instruction=instruction,
                waypoint_labels=waypoint_labels,
                is_success=is_success,
                completed_without_exception=completed,
            )
            saved_count += 1
            if is_success:
                success_count += 1
                print(f"Trial {i:05d} SUCCESSFUL. Saved -> {episode_dir}")
            else:
                stuck_count += 1
                tag = "kinematic stall" if self._stuck_event is not None else "task failed"
                print(f"Trial {i:05d} STUCK ({tag}). Saved -> {episode_dir}")

            if save_video_this_trial:
                self._video_frame(
                    "TRIAL SUCCESSFUL" if is_success else "TRIAL STUCK"
                )
                self._video_stop()

        print(
            f"\nGeneration complete. Saved {saved_count}/{num_trials} samples "
            f"({success_count} success, {stuck_count} stuck)."
        )
        self.env.close()


def main():
    if os.environ.get("MUJOCO_GL") is None:
        print("Setting MUJOCO_GL=egl for headless rendering.")
        os.environ["MUJOCO_GL"] = "egl"

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./my_stuck_data")
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--num_videos", type=int, default=2)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--save_all_phases",
        action="store_true",
        help="Also save image_phase_<phase>.png for each captured trajectory phase.",
    )
    parser.add_argument(
        "--inject_obstacles",
        action="store_true",
        help="(Reserved) Force-inject extra clutter to provoke stuck events.",
    )
    args = parser.parse_args()

    print("\nInitializing Stuck-State Data Generator...")
    gen = StuckStateGenerator(
        output_dir=args.output_dir,
        save_all_phases=args.save_all_phases,
        inject_obstacles=args.inject_obstacles,
    )
    gen.run_generation_loop(
        num_trials=args.num_trials,
        num_videos=args.num_videos,
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
