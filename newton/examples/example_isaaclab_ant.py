import numpy as np
import torch
import warp as wp

import newton
import newton.core.articulation
import newton.examples
import newton.utils
import newton.utils.selection


class Example:
    def __init__(self, stage_path="example_ant.usd", num_envs=8):
        self.num_envs = num_envs

        up_axis = "Z"
        up_vector = (0.0, 0.0, 1.0)
        gravity = -9.81

        articulation_builder = newton.ModelBuilder(up_vector=up_vector, gravity=gravity)
        newton.utils.parse_usd(
            newton.examples.get_asset("flattened_ant.usd"),
            articulation_builder,
            collapse_fixed_joints=True,
        )

        builder = newton.ModelBuilder(up_vector=up_vector, gravity=gravity)

        positions = newton.examples.compute_env_offsets(num_envs, env_offset=(2.0, 2.0, 0.0), up_axis=up_axis)

        for i in range(self.num_envs):
            builder.add_builder(
                articulation_builder, xform=wp.transform(positions[i] + np.array((0, 0, 1)), wp.quat_identity())
            )

        # finalize model
        self.model = builder.finalize()
        self.model.ground = True

        # from pprint import pprint
        # pprint(self.model.articulation_key)
        # pprint(self.model.body_key)
        # pprint(self.model.joint_key)

        self.solver = newton.solvers.MuJoCoSolver(self.model)

        self.renderer = None
        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(
                path=stage_path,
                model=self.model,
                scaling=2.0,
                up_axis=up_axis,
                screen_width=1280,
                screen_height=720,
                camera_pos=(0, 4, 20),
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.sim_time = 0.0
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        newton.core.articulation.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        # ===========================================================
        # create articulation view, note the include_root_joint flag
        # ===========================================================
        self.ants = newton.utils.selection.ArticulationView(self.model, "/ant/torso", include_root_joint=False)

        print(f"articulation count: {self.ants.count}")
        print(f"link_count:         {self.ants.link_count}")
        print(f"joint_count:        {self.ants.joint_count}")
        print(f"joint_axis_count:   {self.ants.joint_axis_count}")

        print(f"joint_q shape:      {self.ants.get_attribute_shape('joint_q')}")
        print(f"joint_qd shape:     {self.ants.get_attribute_shape('joint_qd')}")
        print(f"joint_act shape:    {self.ants.get_attribute_shape('joint_act')}")
        print(f"body_q shape:       {self.ants.get_attribute_shape('body_q')}")
        print(f"body_qd shape:      {self.ants.get_attribute_shape('body_qd')}")

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.model, self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # =========================
        # apply random controls
        # =========================
        act_shape = self.ants.get_attribute_shape("joint_act")
        joint_forces = 10.0 - 20.0 * torch.rand(act_shape)
        self.ants.set_attribute("joint_act", self.control, joint_forces)

        with wp.ScopedTimer("step", active=False):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

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
        default="example_ant.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=1200, help="Total number of frames.")
    parser.add_argument("--num_envs", type=int, default=16, help="Total number of simulated environments.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)

        for _ in range(args.num_frames):
            # time.sleep(1)
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
