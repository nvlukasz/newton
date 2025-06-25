# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.examples import compute_env_offsets

COLLAPSE_FIXED_JOINTS = False
VERBOSE = True
USE_FEATHERSTONE = True


class Example:
    def __init__(self, stage_path=None, num_envs=8):
        self.num_envs = num_envs

        up_axis = newton.Axis.Z

        articulation_builder = newton.ModelBuilder(up_axis=up_axis)
        newton.utils.parse_mjcf(
            newton.examples.get_asset("nv_ant.xml"),
            articulation_builder,
            ignore_names=["floor", "ground"],
            up_axis=up_axis,
            xform=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()),
            collapse_fixed_joints=COLLAPSE_FIXED_JOINTS,
        )

        self.env_offsets = compute_env_offsets(num_envs, env_offset=(4.0, 4.0, 0.0), up_axis=up_axis)

        builder = newton.ModelBuilder()
        for i in range(self.num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(self.env_offsets[i], wp.quat_identity()))

        builder.add_ground_plane()

        # finalize model
        self.model = builder.finalize()

        if USE_FEATHERSTONE:
            self.solver = newton.solvers.FeatherstoneSolver(self.model)
        else:
            self.solver = newton.solvers.MuJoCoSolver(self.model)

        self.renderer = None
        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(
                path=stage_path,
                model=self.model,
                scaling=2.0,
                up_axis=str(up_axis),
                screen_width=1280,
                screen_height=720,
                camera_pos=(0, 4, 30),
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.default_joint_q = wp.to_torch(self.state_0.joint_q).clone()
        self.default_joint_qd = wp.to_torch(self.state_0.joint_qd).clone()

        self.sim_time = 0.0
        fps = 240
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.next_reset = 0.0

        # reset all
        self.reset()
        self.next_reset = self.sim_time + 2.0

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # explicit collisions needed without MuJoCo solver
            if not isinstance(self.solver, newton.solvers.MuJoCoSolver):
                collisions = self.model.collide(self.state_0, rigid_contact_margin=0.1)
            else:
                collisions = None

            self.solver.step(self.model, self.state_0, self.state_1, self.control, collisions, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.sim_time >= self.next_reset:
            self.reset()
            self.next_reset = self.sim_time + 2.0

        with wp.ScopedTimer("step", active=False):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def reset(self):
        self.state_0.joint_q = wp.from_torch(self.default_joint_q.clone())
        self.state_0.joint_qd = wp.from_torch(self.default_joint_qd.clone())

        if not isinstance(self.solver, newton.solvers.MuJoCoSolver):
            newton.sim.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0, mask=None)

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_selection_ant.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=1200, help="Total number of frames.")
    parser.add_argument("--num_envs", type=int, default=16, help="Total number of simulated environments.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device), torch.device(wp.device_to_torch(args.device)):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)

        for _ in range(args.num_frames):
            example.step()
            example.render()

            # import time
            # time.sleep(0.2)

        if example.renderer:
            example.renderer.save()
