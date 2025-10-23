# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from isaacsim.core.utils.types import ArticulationAction


class ScenarioTemplate:
    def __init__(self):
        pass

    def setup_scenario(self):
        pass

    def teardown_scenario(self):
        pass

    def update_scenario(self):
        pass


class ExampleScenario(ScenarioTemplate):
    def __init__(self):
        self._object = None
        self._articulation = None
        self._running_scenario = False
        self._time = 0.0
        
        # Trajectory parameters
        self._phase = 0
        self._phase_time = 0.0
        self._phase_duration = 3.0
        
        # Joint configurations
        self._home_position = None
        self._pre_grasp_position = None
        self._grasp_position = None
        self._lift_position = None
        self._place_position = None
        self._pre_place_position = None
        
        # Cube tracking
        self._cube_grasped = False
        self._initial_cube_position = None
        self._gripper_to_cube_offset = None
        
        # Debug flag
        self._debug_printed = False

    def setup_scenario(self, articulation, object_prim):
        self._articulation = articulation
        self._object = object_prim
        
        # Store initial cube position
        positions, orientations = self._object.get_world_poses()
        self._initial_cube_position = positions[0] if len(positions) > 0 else positions
        
        self._running_scenario = True
        self._phase = 0
        self._phase_time = 0.0
        self._cube_grasped = False
        self._debug_printed = False
        
        num_dof = articulation.num_dof
        print(f"\n{'='*60}")
        print(f"SETUP DIAGNOSTICS")
        print(f"{'='*60}")
        print(f"Robot DOF: {num_dof}")
        print(f"Initial Cube Position: {self._initial_cube_position}")
        
        # Get robot base position for reference
        base_pos, _ = self._articulation.get_world_pose()
        print(f"Robot Base Position: {base_pos}")
        print(f"Distance from base to cube: {np.linalg.norm(self._initial_cube_position - base_pos):.3f}m")
        
        # TUNED joint positions based on KUKA KR210 reaching a cube
        # Adjust these based on your actual cube position
        
        self._home_position = np.array([0.0, -0.5, 0.8, 0.0, 0.5, 0.0][:num_dof])
        
        # Calculate base rotation needed to face the cube
        cube_x = self._initial_cube_position[0] - base_pos[0]
        cube_y = self._initial_cube_position[1] - base_pos[1]
        base_rotation = np.arctan2(cube_y, cube_x)
        
        print(f"Calculated base rotation to face cube: {np.degrees(base_rotation):.1f}°")
        
        # Pre-grasp: Position above cube (adjusted to actually reach it)
        self._pre_grasp_position = np.array([
            base_rotation,  # Rotate base to face cube
            -0.7,           # Shoulder angle
            1.0,            # Elbow angle  
            0.0,            # Wrist 1
            1.3,            # Wrist 2 (pointing down)
            0.0             # Wrist 3
        ][:num_dof])
        
        # Grasp: Lower down (closer to cube)
        self._grasp_position = np.array([
            base_rotation,
            -0.5,           # Lower shoulder more
            0.8,            # Adjust elbow
            0.0,
            1.5,            # More downward
            0.0
        ][:num_dof])
        
        # Lift: Same as pre-grasp
        self._lift_position = self._pre_grasp_position.copy()
        
        # Pre-place: Move to placement area
        self._pre_place_position = np.array([
            base_rotation + 0.8,  # Rotate 45° from pickup
            -0.7,
            1.0,
            0.0,
            1.3,
            0.0
        ][:num_dof])
        
        # Place: Lower to place
        self._place_position = np.array([
            base_rotation + 0.8,
            -0.5,
            0.8,
            0.0,
            1.5,
            0.0
        ][:num_dof])
        
        # Start at home
        articulation.set_joint_positions(self._home_position)
        
        print(f"{'='*60}")
        print("Scenario Ready - Watch for gripper position updates")
        print(f"{'='*60}\n")

    def teardown_scenario(self):
        self._time = 0.0
        self._phase_time = 0.0
        self._phase = 0
        self._object = None
        self._articulation = None
        self._running_scenario = False
        self._cube_grasped = False
        self._debug_printed = False

    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        
        self._time += step
        self._phase_time += step
        
        # Debug output at start of each phase
        if self._phase_time < step * 2 and not self._debug_printed:
            self._print_debug_info()
            self._debug_printed = True
        
        # Phase transitions
        if self._phase_time >= self._phase_duration:
            self._phase += 1
            self._phase_time = 0.0
            self._debug_printed = False
            
            if self._phase == 1:
                print("\n>>> Phase 1: Moving to grasp position")
            elif self._phase == 2:
                print("\n>>> Phase 2: Grasping cube")
                # Calculate offset when grasping
                self._calculate_gripper_offset()
                self._cube_grasped = True
            elif self._phase == 3:
                print("\n>>> Phase 3: Lifting cube")
            elif self._phase == 4:
                print("\n>>> Phase 4: Moving to pre-place position")
            elif self._phase == 5:
                print("\n>>> Phase 5: Lowering to place")
            elif self._phase == 6:
                print("\n>>> Phase 6: Releasing cube")
                self._cube_grasped = False
            elif self._phase == 7:
                print("\n>>> Phase 7: Returning home")
            elif self._phase >= 8:
                print("\n>>> Pick and Place Complete!")
                self._running_scenario = False
                return
        
        # Execute current phase
        self._execute_phase(step)
        
        # Update cube to follow gripper
        if self._cube_grasped:
            self._attach_cube_to_gripper()

    def _print_debug_info(self):
        """Print debug information about gripper and cube positions"""
        gripper_pos = self._get_end_effector_position()
        cube_pos, _ = self._object.get_world_poses()
        cube_pos = cube_pos[0] if len(cube_pos) > 0 else cube_pos
        
        distance = np.linalg.norm(gripper_pos - cube_pos)
        joint_pos = self._articulation.get_joint_positions()
        
        print(f"  Gripper Position: [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")
        print(f"  Cube Position:    [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
        print(f"  Distance to cube: {distance:.3f}m")
        print(f"  Joint angles (deg): {np.degrees(joint_pos)}")
        
        if self._phase == 1 and distance > 0.15:
            print(f"  ⚠️  WARNING: Gripper may not reach cube! Distance: {distance:.3f}m")
            print(f"  💡 TIP: Adjust joint angles in _grasp_position")

    def _calculate_gripper_offset(self):
        """Calculate the offset from gripper to cube when grasping"""
        gripper_pos = self._get_end_effector_position()
        cube_pos, _ = self._object.get_world_poses()
        cube_pos = cube_pos[0] if len(cube_pos) > 0 else cube_pos
        
        self._gripper_to_cube_offset = cube_pos - gripper_pos
        
        distance = np.linalg.norm(self._gripper_to_cube_offset)
        print(f"  Gripper-to-cube offset: {self._gripper_to_cube_offset}")
        print(f"  Offset magnitude: {distance:.3f}m")
        
        if distance > 0.15:
            print(f"  ⚠️  WARNING: Large offset detected! Cube may not be properly grasped.")
            print(f"  The gripper is {distance:.3f}m away from the cube.")

    def _attach_cube_to_gripper(self):
        """Properly attach cube to gripper using offset"""
        gripper_pos = self._get_end_effector_position()
        gripper_rot = self._get_end_effector_rotation()
        
        # Apply offset to maintain relative position
        new_cube_pos = gripper_pos + self._gripper_to_cube_offset
        
        # Set cube position with proper array format
        self._object.set_world_poses(
            positions=np.array([new_cube_pos]),
            orientations=np.array([gripper_rot])
        )

    def _get_end_effector_position(self):
        """Get end-effector position using forward kinematics"""
        # FIX: Use get_world_pose() instead of get_world_poses()
        base_pos, _ = self._articulation.get_world_pose()
        
        joint_positions = self._articulation.get_joint_positions()
        
        # KUKA KR210 L150 approximate link lengths (in meters)
        # ADJUST THESE to match your actual robot!
        L1 = 0.675  # Base to shoulder height
        L2 = 1.15   # Shoulder to elbow (upper arm)
        L3 = 1.15   # Elbow to wrist (forearm)
        L4 = 0.30   # Wrist to gripper tip
        
        # Extract joint angles
        q0 = joint_positions[0] if len(joint_positions) > 0 else 0.0  # Base rotation
        q1 = joint_positions[1] if len(joint_positions) > 1 else 0.0  # Shoulder
        q2 = joint_positions[2] if len(joint_positions) > 2 else 0.0  # Elbow
        q3 = joint_positions[3] if len(joint_positions) > 3 else 0.0  # Wrist 1
        q4 = joint_positions[4] if len(joint_positions) > 4 else 0.0  # Wrist 2
        
        # Forward kinematics calculation
        # Horizontal reach
        horizontal_reach = (L2 * np.cos(q1) + 
                           L3 * np.cos(q1 + q2) + 
                           L4 * np.cos(q1 + q2 + q4))
        
        # Position in robot's coordinate frame
        x = horizontal_reach * np.cos(q0)
        y = horizontal_reach * np.sin(q0)
        z = L1 + L2 * np.sin(q1) + L3 * np.sin(q1 + q2) + L4 * np.sin(q1 + q2 + q4)
        
        # Add base position offset
        ee_pos = base_pos + np.array([x, y, z])
        
        return ee_pos

    def _get_end_effector_rotation(self):
        """Get approximate end-effector rotation"""
        # Return identity quaternion [w, x, y, z]
        return np.array([1.0, 0.0, 0.0, 0.0])

    def _execute_phase(self, step):
        """Execute the current phase with smooth interpolation"""
        
        # Smooth interpolation factor
        t = min(self._phase_time / self._phase_duration, 1.0)
        
        current_positions = self._articulation.get_joint_positions()
        
        # Determine target based on phase
        if self._phase == 0:
            target = self._interpolate_positions(self._home_position, self._pre_grasp_position, t)
        elif self._phase == 1:
            target = self._interpolate_positions(self._pre_grasp_position, self._grasp_position, t)
        elif self._phase == 2:
            target = self._grasp_position  # Hold for grasping
        elif self._phase == 3:
            target = self._interpolate_positions(self._grasp_position, self._lift_position, t)
        elif self._phase == 4:
            target = self._interpolate_positions(self._lift_position, self._pre_place_position, t)
        elif self._phase == 5:
            target = self._interpolate_positions(self._pre_place_position, self._place_position, t)
        elif self._phase == 6:
            target = self._place_position  # Hold for releasing
        elif self._phase == 7:
            target = self._interpolate_positions(self._place_position, self._home_position, t)
        else:
            target = self._home_position
        
        # Smooth velocity control
        velocity = (target - current_positions) / max(step, 0.001)
        velocity = np.clip(velocity, -1.5, 1.5)  # Limit velocity
        
        # Apply action
        action = ArticulationAction(
            joint_positions=target,
            joint_velocities=velocity
        )
        self._articulation.apply_action(action)

    def _interpolate_positions(self, start, end, t):
        """Smooth cosine interpolation"""
        smooth_t = (1 - np.cos(t * np.pi)) / 2
        return start + (end - start) * smooth_t