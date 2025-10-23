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
        self._articulation = None
        self._cubes = []  # Now handles multiple cubes
        self._current_cube_index = 0
        self._current_cube = None
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
        self._initial_cube_positions = []
        self._gripper_to_cube_offset = None
        
        # Placement positions for each cube
        self._placement_offsets = [
            np.array([0.0, 0.0, 0.0]),    # Cube 0: original offset
            np.array([0.15, 0.0, 0.0]),   # Cube 1: 15cm to the right
            np.array([0.30, 0.0, 0.0]),   # Cube 2: 30cm to the right
            np.array([0.45, 0.0, 0.0]),   # Cube 3: 45cm to the right
        ]
        
        # Debug flag
        self._debug_printed = False

    def setup_scenario(self, articulation, cube_prims):
        """
        Setup scenario with multiple cubes
        Args:
            articulation: The robot articulation
            cube_prims: List of cube XFormPrims or single cube
        """
        self._articulation = articulation
        
        # Handle both single cube and list of cubes
        if isinstance(cube_prims, list):
            self._cubes = cube_prims
        else:
            self._cubes = [cube_prims]
        
        # Store initial positions for all cubes
        self._initial_cube_positions = []
        for cube in self._cubes:
            positions, orientations = cube.get_world_poses()
            pos = positions[0] if len(positions) > 0 else positions
            self._initial_cube_positions.append(pos)
        
        # Start with first cube
        self._current_cube_index = 0
        self._current_cube = self._cubes[0]
        
        self._running_scenario = True
        self._phase = 0
        self._phase_time = 0.0
        self._cube_grasped = False
        self._debug_printed = False
        
        num_dof = articulation.num_dof
        print(f"\n{'='*60}")
        print(f"SETUP DIAGNOSTICS - MULTI-CUBE PICK AND PLACE")
        print(f"{'='*60}")
        print(f"Robot DOF: {num_dof}")
        print(f"Number of cubes: {len(self._cubes)}")
        
        # Get robot base position
        base_pos, _ = self._articulation.get_world_pose()
        print(f"Robot Base Position: {base_pos}")
        
        # Print all cube positions
        for i, cube_pos in enumerate(self._initial_cube_positions):
            distance = np.linalg.norm(cube_pos - base_pos)
            print(f"  Cube {i} Position: {cube_pos} (distance: {distance:.3f}m)")
        
        # Home position
        self._home_position = np.array([0.0, -0.5, 0.8, 0.0, 0.5, 0.0][:num_dof])
        
        # Start at home
        articulation.set_joint_positions(self._home_position)
        
        print(f"{'='*60}")
        print(f"Starting with Cube 0")
        print("Scenario Ready - Watch for gripper position updates")
        print(f"{'='*60}\n")

    def teardown_scenario(self):
        self._time = 0.0
        self._phase_time = 0.0
        self._phase = 0
        self._cubes = []
        self._current_cube = None
        self._current_cube_index = 0
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
                print(f"\n>>> Cube {self._current_cube_index}: Phase 1 - Moving to grasp position")
            elif self._phase == 2:
                print(f"\n>>> Cube {self._current_cube_index}: Phase 2 - Grasping cube")
                self._calculate_gripper_offset()
                self._cube_grasped = True
            elif self._phase == 3:
                print(f"\n>>> Cube {self._current_cube_index}: Phase 3 - Lifting cube")
            elif self._phase == 4:
                print(f"\n>>> Cube {self._current_cube_index}: Phase 4 - Moving to pre-place position")
            elif self._phase == 5:
                print(f"\n>>> Cube {self._current_cube_index}: Phase 5 - Lowering to place")
            elif self._phase == 6:
                print(f"\n>>> Cube {self._current_cube_index}: Phase 6 - Releasing cube")
                self._cube_grasped = False
            elif self._phase == 7:
                print(f"\n>>> Cube {self._current_cube_index}: Phase 7 - Returning home")
            elif self._phase >= 8:
                # Completed current cube
                print(f"\n✓ Cube {self._current_cube_index} Pick and Place Complete!")
                
                # Move to next cube
                self._current_cube_index += 1
                
                if self._current_cube_index < len(self._cubes):
                    # Start next cube
                    print(f"\n{'='*60}")
                    print(f"Starting Cube {self._current_cube_index}")
                    print(f"{'='*60}\n")
                    
                    self._current_cube = self._cubes[self._current_cube_index]
                    self._phase = 0
                    self._phase_time = 0.0
                    self._cube_grasped = False
                    self._debug_printed = False
                else:
                    # All cubes completed
                    print(f"\n{'='*60}")
                    print(f"🎉 ALL {len(self._cubes)} CUBES COMPLETED! 🎉")
                    print(f"{'='*60}\n")
                    self._running_scenario = False
                    return
        
        # Execute current phase
        self._execute_phase(step)
        
        # Update cube to follow gripper
        if self._cube_grasped:
            self._attach_cube_to_gripper()

    def _calculate_joint_positions_for_cube(self, cube_index):
        """Calculate joint positions to reach a specific cube"""
        base_pos, _ = self._articulation.get_world_pose()
        cube_pos = self._initial_cube_positions[cube_index]
        
        # Calculate base rotation needed to face the cube
        cube_x = cube_pos[0] - base_pos[0]
        cube_y = cube_pos[1] - base_pos[1]
        base_rotation = np.arctan2(cube_y, cube_x)
        
        num_dof = self._articulation.num_dof
        
        # Pre-grasp: Position above cube
        self._pre_grasp_position = np.array([
            base_rotation,
            -0.7,
            1.0,
            0.0,
            1.3,
            0.0
        ][:num_dof])
        
        # Grasp: Lower to cube
        self._grasp_position = np.array([
            base_rotation,
            -0.5,
            0.8,
            0.0,
            1.5,
            0.0
        ][:num_dof])
        
        # Lift: Same as pre-grasp
        self._lift_position = self._pre_grasp_position.copy()
        
        # Calculate placement position with offset
        placement_offset = self._placement_offsets[cube_index]
        
        # Pre-place: Above placement location
        self._pre_place_position = np.array([
            base_rotation + 0.8,
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

    def _print_debug_info(self):
        """Print debug information about gripper and cube positions"""
        gripper_pos = self._get_end_effector_position()
        cube_pos, _ = self._current_cube.get_world_poses()
        cube_pos = cube_pos[0] if len(cube_pos) > 0 else cube_pos
        
        distance = np.linalg.norm(gripper_pos - cube_pos)
        joint_pos = self._articulation.get_joint_positions()
        
        print(f"  [Cube {self._current_cube_index}] Phase {self._phase}")
        print(f"  Gripper Position: [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")
        print(f"  Cube Position:    [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
        print(f"  Distance to cube: {distance:.3f}m")
        print(f"  Joint angles (deg): {np.degrees(joint_pos)}")
        
        if self._phase == 1 and distance > 0.15:
            print(f"  ⚠️  WARNING: Gripper may not reach cube! Distance: {distance:.3f}m")

    def _calculate_gripper_offset(self):
        """Calculate the offset from gripper to cube when grasping"""
        gripper_pos = self._get_end_effector_position()
        cube_pos, _ = self._current_cube.get_world_poses()
        cube_pos = cube_pos[0] if len(cube_pos) > 0 else cube_pos
        
        self._gripper_to_cube_offset = cube_pos - gripper_pos
        
        distance = np.linalg.norm(self._gripper_to_cube_offset)
        print(f"  Gripper-to-cube offset: {self._gripper_to_cube_offset}")
        print(f"  Offset magnitude: {distance:.3f}m")

    def _attach_cube_to_gripper(self):
        """Properly attach cube to gripper using offset"""
        gripper_pos = self._get_end_effector_position()
        gripper_rot = self._get_end_effector_rotation()
        
        # Apply offset to maintain relative position
        new_cube_pos = gripper_pos + self._gripper_to_cube_offset
        
        # Set cube position
        self._current_cube.set_world_poses(
            positions=np.array([new_cube_pos]),
            orientations=np.array([gripper_rot])
        )

    def _get_end_effector_position(self):
        """Get end-effector position using forward kinematics"""
        base_pos, _ = self._articulation.get_world_pose()
        joint_positions = self._articulation.get_joint_positions()
        
        # Link lengths (adjust for your robot)
        L1 = 0.675
        L2 = 1.15
        L3 = 1.15
        L4 = 0.30
        
        # Extract joint angles
        q0 = joint_positions[0] if len(joint_positions) > 0 else 0.0
        q1 = joint_positions[1] if len(joint_positions) > 1 else 0.0
        q2 = joint_positions[2] if len(joint_positions) > 2 else 0.0
        q3 = joint_positions[3] if len(joint_positions) > 3 else 0.0
        q4 = joint_positions[4] if len(joint_positions) > 4 else 0.0
        
        # Forward kinematics
        horizontal_reach = (L2 * np.cos(q1) + 
                           L3 * np.cos(q1 + q2) + 
                           L4 * np.cos(q1 + q2 + q4))
        
        x = horizontal_reach * np.cos(q0)
        y = horizontal_reach * np.sin(q0)
        z = L1 + L2 * np.sin(q1) + L3 * np.sin(q1 + q2) + L4 * np.sin(q1 + q2 + q4)
        
        ee_pos = base_pos + np.array([x, y, z])
        return ee_pos

    def _get_end_effector_rotation(self):
        """Get approximate end-effector rotation"""
        return np.array([1.0, 0.0, 0.0, 0.0])

    def _execute_phase(self, step):
        """Execute the current phase with smooth interpolation"""
        
        # Calculate joint positions for current cube at start of cycle
        if self._phase == 0 and self._phase_time < step * 2:
            self._calculate_joint_positions_for_cube(self._current_cube_index)
        
        # Smooth interpolation factor
        t = min(self._phase_time / self._phase_duration, 1.0)
        
        current_positions = self._articulation.get_joint_positions()
        
        # Determine target based on phase
        if self._phase == 0:
            target = self._interpolate_positions(self._home_position, self._pre_grasp_position, t)
        elif self._phase == 1:
            target = self._interpolate_positions(self._pre_grasp_position, self._grasp_position, t)
        elif self._phase == 2:
            target = self._grasp_position
        elif self._phase == 3:
            target = self._interpolate_positions(self._grasp_position, self._lift_position, t)
        elif self._phase == 4:
            target = self._interpolate_positions(self._lift_position, self._pre_place_position, t)
        elif self._phase == 5:
            target = self._interpolate_positions(self._pre_place_position, self._place_position, t)
        elif self._phase == 6:
            target = self._place_position
        elif self._phase == 7:
            target = self._interpolate_positions(self._place_position, self._home_position, t)
        else:
            target = self._home_position
        
        # Smooth velocity control
        velocity = (target - current_positions) / max(step, 0.001)
        velocity = np.clip(velocity, -1.5, 1.5)
        
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