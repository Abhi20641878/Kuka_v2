# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import omni.timeline
import omni.ui as ui
from isaacsim.core.api.objects.cuboid import FixedCuboid
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation, XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage, get_current_stage
from isaacsim.gui.components.element_wrappers import CollapsableFrame, StateButton
from isaacsim.gui.components.ui_utils import get_style
from omni.usd import StageEventType
from pxr import Sdf, UsdLux

from .scenario import ExampleScenario


class UIBuilder:
    def __init__(self):
        self.frames = []
        self.wrapped_ui_elements = []
        self._timeline = omni.timeline.get_timeline_interface()
        self._on_init()

    def on_menu_callback(self):
        pass

    def on_timeline_event(self, event):
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False

    def on_physics_step(self, step: float):
        pass

    def on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED):
            self._reset_extension()

    def cleanup(self):
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

    def build_ui(self):
        world_controls_frame = CollapsableFrame("World Controls", collapsed=False)

        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # Create custom load button
                with ui.HStack():
                    ui.Button(
                        "LOAD",
                        clicked_fn=self._on_load_world,
                        height=40,
                        style=get_style()
                    )

                with ui.HStack():
                    self._reset_button = ui.Button(
                        "RESET",
                        clicked_fn=self._on_reset_world,
                        height=40,
                        style=get_style(),
                        enabled=False
                    )

        run_scenario_frame = CollapsableFrame("Run Scenario")

        with run_scenario_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._scenario_state_btn = StateButton(
                    "Run Scenario",
                    "RUN",
                    "STOP",
                    on_a_click_fn=self._on_run_scenario_a_text,
                    on_b_click_fn=self._on_run_scenario_b_text,
                    physics_callback_fn=self._update_scenario,
                )
                self._scenario_state_btn.enabled = False
                self.wrapped_ui_elements.append(self._scenario_state_btn)

    def _on_init(self):
        self._articulation = None
        self._cuboid = None
        self._cubes = []
        self._table = None
        self._scenario = ExampleScenario()
        self._world = None

    def _add_light_to_stage(self):
        sphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_poses(np.array([[6.5, 0, 12]]))

    def _on_load_world(self):
        """Custom load function that properly manages World creation"""
        print("\n" + "="*50)
        print("Loading KUKA Pick and Place Scenario...")
        print("="*50)
        
        # CRITICAL: Stop timeline first
        if self._timeline.is_playing():
            print("Stopping timeline...")
            self._timeline.stop()
        
        # CRITICAL: Clear World singleton - this is the key fix
        print("Clearing World singleton...")
        try:
            World.clear_instance()
        except Exception as e:
            print(f"Note: {e}")
        
        # Reset internal state
        self._on_init()
        
        # Step 1: Create new stage and load assets
        print("Creating new stage...")
        create_new_stage()
        self._add_light_to_stage()
        
        # Load the KUKA robot
        robot_prim_path = "/kr210_l150"
        path_to_robot_usd = "D:/Ext/kuka/data/Collected_kr210_l150/kr210_l150.usd"
        
        print(f"Loading robot from: {path_to_robot_usd}")
        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        
        # ============================================================
        # LOAD YOUR TABLE USD FILE HERE
        # ============================================================
        print("Loading table with cubes...")
        
        # CHANGE THIS PATH TO YOUR TABLE USD FILE:
        table_usd_path = "D:/Ext/table_1.usd"
        table_prim_path = "/Scenario/TableWithCubes"
        
        # Load the table USD file
        add_reference_to_stage(table_usd_path, table_prim_path)
        print(f"✓ Successfully loaded table from: {table_usd_path}")
            
        # Create XFormPrim wrapper for the table
        self._table = XFormPrim(
            f"{table_prim_path}/Default",
            "table_main"  # Give it a unique name
        )
        print(f"✓ Created table prim wrapper")
            
        # Get the cube prims from your USD file
        self._cubes = []
        cube_names = ["Pickup_A", "Pickup_B", "Pickup_C", "Pickup_D"]
            
        for i, cube_name in enumerate(cube_names):
            try:
                # Adjust this path based on your USD hierarchy
                cube_prim_path = f"{table_prim_path}/Default/Cubes/{cube_name}"
                cube_prim = XFormPrim(
                    cube_prim_path,
                    f"cube_{i}"  # Give each cube a unique name
                )
                self._cubes.append(cube_prim)
                print(f"  ✓ Found cube: {cube_name} at {cube_prim_path}")
            except Exception as e:
                print(f"  ⚠ Could not find cube {cube_name}: {e}")
            
        # Use the first cube for the scenario
        if len(self._cubes) > 0:
            self._cuboid = self._cubes[0]
            print(f"✓ Using {len(self._cubes)} cubes from USD file")
        else:
            raise Exception("No cubes found in USD file")

        # Create articulation wrapper with UNIQUE NAME
        print("Creating articulation wrapper...")
        self._articulation = SingleArticulation(robot_prim_path, name="kuka_robot")

        # Step 2: Create World with physics settings
        print("Creating World...")
        self._world = World(physics_dt=1/60.0, rendering_dt=1/60.0)
        
        # Step 3: Add objects to world
        print("Adding objects to World scene...")
        
        try:
            self._world.scene.add(self._articulation)
            
            # Only add table if it was successfully created
            if self._table is not None:
                self._world.scene.add(self._table)
            
            for i, cube in enumerate(self._cubes):
                self._world.scene.add(cube)
                
        except Exception as e:
            print(f"Error adding objects to scene: {e}")
            raise
        
        # Step 4: Initialize world (this calls reset internally)
        print("Initializing World...")
        self._world.reset()
        
        # Step 5: Setup scenario
        print("Setting up scenario...")
        self._reset_scenario()
        
        # Enable UI
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._enable_reset_button(True)
        
        print("\n" + "="*50)
        print("✓ KUKA Pick and Place Ready!")
        print(f"✓ Loaded 1 table and {len(self._cubes)} cubes")
        print("✓ Click RUN to start the scenario")
        print("="*50 + "\n")

    def _on_reset_world(self):
        """Reset the world and scenario"""
        if self._world is None:
            print("World not loaded yet!")
            return
            
        print("\nResetting world...")
        
        # Stop timeline if playing
        if self._timeline.is_playing():
            self._timeline.stop()
        
        self._world.reset()
        self._reset_scenario()
        
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        print("World reset complete!\n")

    def _enable_reset_button(self, enabled):
        """Helper to enable/disable reset button"""
        if hasattr(self, '_reset_button'):
            self._reset_button.enabled = enabled

    def _reset_scenario(self):
        if self._articulation is None or self._cuboid is None:
            print("Warning: Cannot reset scenario - objects not loaded")
            return
            
        self._scenario.teardown_scenario()
        self._scenario.setup_scenario(self._articulation, self._cuboid)

    def _update_scenario(self, step: float):
        self._scenario.update_scenario(step)

    def _on_run_scenario_a_text(self):
        print("\n>>> Starting Pick and Place <<<\n")
        self._timeline.play()

    def _on_run_scenario_b_text(self):
        print("\n>>> Stopping Scenario <<<\n")
        self._timeline.pause()

    def _reset_extension(self):
        # Stop timeline first
        if self._timeline.is_playing():
            self._timeline.stop()
        
        # Clear World singleton
        try:
            World.clear_instance()
        except Exception as e:
            print(f"Note during extension reset: {e}")
            
        self._on_init()
        
        if hasattr(self, '_scenario_state_btn'):
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False
        if hasattr(self, '_reset_button'):
            self._reset_button.enabled = False