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

import unittest

import warp as wp

import newton
import newton.examples
from newton.selection import ArticulationView


class TestSelection(unittest.TestCase):
    def test_no_match(self):
        builder = newton.ModelBuilder()
        builder.add_body()
        model = builder.finalize()
        self.assertRaises(KeyError, ArticulationView, model, pattern="no_match")

    def test_empty_selection(self):
        builder = newton.ModelBuilder()
        body = builder.add_link()
        joint = builder.add_joint_free(child=body)
        builder.add_articulation([joint], key="my_articulation")
        model = builder.finalize()
        control = model.control()
        selection = ArticulationView(model, pattern="my_articulation", exclude_joint_types=[newton.JointType.FREE])
        self.assertEqual(selection.count, 1)
        self.assertEqual(selection.get_root_transforms(model).shape, (1,))
        self.assertEqual(selection.get_dof_positions(model).shape, (1, 0))
        self.assertEqual(selection.get_dof_velocities(model).shape, (1, 0))
        self.assertEqual(selection.get_dof_forces(control).shape, (1, 0))

    def test_selection_squeeze_ants(self):
        # load articulation
        ant = newton.ModelBuilder()
        ant.add_mjcf(
            newton.examples.get_asset("nv_ant.xml"),
            ignore_names=["floor", "ground"],
        )

        W = 10  # num worlds
        A = 3  # num ants per world
        L = 9  # num ant links

        #
        # scene with just one ant
        #
        single_ant_model = ant.finalize()

        single_ant_squeeze_default = ArticulationView(
            single_ant_model, "ant", exclude_joint_types=[newton.JointType.FREE]
        )
        self.assertEqual(single_ant_squeeze_default.get_root_transforms(single_ant_model).shape, (1,))
        self.assertEqual(single_ant_squeeze_default.get_link_transforms(single_ant_model).shape, (1, L))
        self.assertEqual(single_ant_squeeze_default.get_attribute("body_mass", single_ant_model).shape, (1, L))

        single_ant_squeeze_false = ArticulationView(
            single_ant_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=False
        )
        self.assertEqual(single_ant_squeeze_false.get_root_transforms(single_ant_model).shape, (1, 1))
        self.assertEqual(single_ant_squeeze_false.get_link_transforms(single_ant_model).shape, (1, 1, L))
        self.assertEqual(single_ant_squeeze_false.get_attribute("body_mass", single_ant_model).shape, (1, 1, L))

        single_ant_squeeze_true = ArticulationView(
            single_ant_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=True
        )
        self.assertEqual(single_ant_squeeze_true.get_root_transforms(single_ant_model).shape, (1,))
        self.assertEqual(single_ant_squeeze_true.get_link_transforms(single_ant_model).shape, (L,))
        self.assertEqual(single_ant_squeeze_true.get_attribute("body_mass", single_ant_model).shape, (L,))

        single_ant_squeeze_0 = ArticulationView(
            single_ant_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(0,)
        )
        self.assertEqual(single_ant_squeeze_0.get_root_transforms(single_ant_model).shape, (1,))
        self.assertEqual(single_ant_squeeze_0.get_link_transforms(single_ant_model).shape, (1, L))
        self.assertEqual(single_ant_squeeze_0.get_attribute("body_mass", single_ant_model).shape, (1, L))

        single_ant_squeeze_1 = ArticulationView(
            single_ant_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(1,)
        )
        self.assertEqual(single_ant_squeeze_1.get_root_transforms(single_ant_model).shape, (1,))
        self.assertEqual(single_ant_squeeze_1.get_link_transforms(single_ant_model).shape, (1, L))
        self.assertEqual(single_ant_squeeze_1.get_attribute("body_mass", single_ant_model).shape, (1, L))

        single_ant_squeeze_2 = ArticulationView(
            single_ant_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(2,)
        )
        self.assertEqual(single_ant_squeeze_2.get_root_transforms(single_ant_model).shape, (1, 1))
        self.assertEqual(single_ant_squeeze_2.get_link_transforms(single_ant_model).shape, (1, 1, L))
        self.assertEqual(single_ant_squeeze_2.get_attribute("body_mass", single_ant_model).shape, (1, 1, L))

        single_ant_squeeze_01 = ArticulationView(
            single_ant_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(0, 1)
        )
        self.assertEqual(single_ant_squeeze_01.get_root_transforms(single_ant_model).shape, (1,))
        self.assertEqual(single_ant_squeeze_01.get_link_transforms(single_ant_model).shape, (L,))
        self.assertEqual(single_ant_squeeze_01.get_attribute("body_mass", single_ant_model).shape, (L,))

        single_ant_squeeze_12 = ArticulationView(
            single_ant_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(1, 2)
        )
        self.assertEqual(single_ant_squeeze_12.get_root_transforms(single_ant_model).shape, (1,))
        self.assertEqual(single_ant_squeeze_12.get_link_transforms(single_ant_model).shape, (1, L))
        self.assertEqual(single_ant_squeeze_12.get_attribute("body_mass", single_ant_model).shape, (1, L))

        single_ant_squeeze_02 = ArticulationView(
            single_ant_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(0, 2)
        )
        self.assertEqual(single_ant_squeeze_02.get_root_transforms(single_ant_model).shape, (1,))
        self.assertEqual(single_ant_squeeze_02.get_link_transforms(single_ant_model).shape, (1, L))
        self.assertEqual(single_ant_squeeze_02.get_attribute("body_mass", single_ant_model).shape, (1, L))

        single_ant_squeeze_012 = ArticulationView(
            single_ant_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(0, 1, 2)
        )
        self.assertEqual(single_ant_squeeze_012.get_root_transforms(single_ant_model).shape, (1,))
        self.assertEqual(single_ant_squeeze_012.get_link_transforms(single_ant_model).shape, (L,))
        self.assertEqual(single_ant_squeeze_012.get_attribute("body_mass", single_ant_model).shape, (L,))

        #
        # scene with one ant per world
        #
        single_ant_per_world_scene = newton.ModelBuilder()
        single_ant_per_world_scene.replicate(ant, num_worlds=W)
        single_ant_per_world_model = single_ant_per_world_scene.finalize()

        single_ant_per_world_squeeze_default = ArticulationView(
            single_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE]
        )
        self.assertEqual(
            single_ant_per_world_squeeze_default.get_root_transforms(single_ant_per_world_model).shape, (W,)
        )
        self.assertEqual(
            single_ant_per_world_squeeze_default.get_link_transforms(single_ant_per_world_model).shape, (W, L)
        )
        self.assertEqual(
            single_ant_per_world_squeeze_default.get_attribute("body_mass", single_ant_per_world_model).shape, (W, L)
        )

        single_ant_per_world_squeeze_false = ArticulationView(
            single_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=False
        )
        self.assertEqual(
            single_ant_per_world_squeeze_false.get_root_transforms(single_ant_per_world_model).shape, (W, 1)
        )
        self.assertEqual(
            single_ant_per_world_squeeze_false.get_link_transforms(single_ant_per_world_model).shape, (W, 1, L)
        )
        self.assertEqual(
            single_ant_per_world_squeeze_false.get_attribute("body_mass", single_ant_per_world_model).shape, (W, 1, L)
        )

        single_ant_per_world_squeeze_true = ArticulationView(
            single_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=True
        )
        self.assertEqual(single_ant_per_world_squeeze_true.get_root_transforms(single_ant_per_world_model).shape, (W,))
        self.assertEqual(
            single_ant_per_world_squeeze_true.get_link_transforms(single_ant_per_world_model).shape, (W, L)
        )
        self.assertEqual(
            single_ant_per_world_squeeze_true.get_attribute("body_mass", single_ant_per_world_model).shape, (W, L)
        )

        single_ant_per_world_squeeze_0 = ArticulationView(
            single_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(0,)
        )
        self.assertEqual(single_ant_per_world_squeeze_0.get_root_transforms(single_ant_per_world_model).shape, (W, 1))
        self.assertEqual(
            single_ant_per_world_squeeze_0.get_link_transforms(single_ant_per_world_model).shape, (W, 1, L)
        )
        self.assertEqual(
            single_ant_per_world_squeeze_0.get_attribute("body_mass", single_ant_per_world_model).shape, (W, 1, L)
        )

        single_ant_per_world_squeeze_1 = ArticulationView(
            single_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(1,)
        )
        self.assertEqual(single_ant_per_world_squeeze_1.get_root_transforms(single_ant_per_world_model).shape, (W,))
        self.assertEqual(single_ant_per_world_squeeze_1.get_link_transforms(single_ant_per_world_model).shape, (W, L))
        self.assertEqual(
            single_ant_per_world_squeeze_1.get_attribute("body_mass", single_ant_per_world_model).shape, (W, L)
        )

        single_ant_per_world_squeeze_2 = ArticulationView(
            single_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(2,)
        )
        self.assertEqual(single_ant_per_world_squeeze_2.get_root_transforms(single_ant_per_world_model).shape, (W, 1))
        self.assertEqual(
            single_ant_per_world_squeeze_2.get_link_transforms(single_ant_per_world_model).shape, (W, 1, L)
        )
        self.assertEqual(
            single_ant_per_world_squeeze_2.get_attribute("body_mass", single_ant_per_world_model).shape, (W, 1, L)
        )

        single_ant_per_world_squeeze_01 = ArticulationView(
            single_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(0, 1)
        )
        self.assertEqual(single_ant_per_world_squeeze_01.get_root_transforms(single_ant_per_world_model).shape, (W,))
        self.assertEqual(single_ant_per_world_squeeze_01.get_link_transforms(single_ant_per_world_model).shape, (W, L))
        self.assertEqual(
            single_ant_per_world_squeeze_01.get_attribute("body_mass", single_ant_per_world_model).shape, (W, L)
        )

        single_ant_per_world_squeeze_12 = ArticulationView(
            single_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(1, 2)
        )
        self.assertEqual(single_ant_per_world_squeeze_12.get_root_transforms(single_ant_per_world_model).shape, (W,))
        self.assertEqual(single_ant_per_world_squeeze_12.get_link_transforms(single_ant_per_world_model).shape, (W, L))
        self.assertEqual(
            single_ant_per_world_squeeze_12.get_attribute("body_mass", single_ant_per_world_model).shape, (W, L)
        )

        single_ant_per_world_squeeze_02 = ArticulationView(
            single_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(0, 2)
        )
        self.assertEqual(single_ant_per_world_squeeze_02.get_root_transforms(single_ant_per_world_model).shape, (W, 1))
        self.assertEqual(
            single_ant_per_world_squeeze_02.get_link_transforms(single_ant_per_world_model).shape, (W, 1, L)
        )
        self.assertEqual(
            single_ant_per_world_squeeze_02.get_attribute("body_mass", single_ant_per_world_model).shape, (W, 1, L)
        )

        single_ant_per_world_squeeze_012 = ArticulationView(
            single_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(0, 1, 2)
        )
        self.assertEqual(single_ant_per_world_squeeze_012.get_root_transforms(single_ant_per_world_model).shape, (W,))
        self.assertEqual(single_ant_per_world_squeeze_012.get_link_transforms(single_ant_per_world_model).shape, (W, L))
        self.assertEqual(
            single_ant_per_world_squeeze_012.get_attribute("body_mass", single_ant_per_world_model).shape, (W, L)
        )

        #
        # scene with multiple ants per world
        #
        multi_ant_world = newton.ModelBuilder()
        for i in range(A):
            multi_ant_world.add_builder(ant, xform=wp.transform((0.0, 0.0, 1.0 + i), wp.quat_identity()))
        multi_ant_per_world_scene = newton.ModelBuilder()
        multi_ant_per_world_scene.replicate(multi_ant_world, num_worlds=W)
        multi_ant_per_world_model = multi_ant_per_world_scene.finalize()

        multi_ant_per_world_squeeze_default = ArticulationView(
            multi_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE]
        )
        self.assertEqual(
            multi_ant_per_world_squeeze_default.get_root_transforms(multi_ant_per_world_model).shape, (W, A)
        )
        self.assertEqual(
            multi_ant_per_world_squeeze_default.get_link_transforms(multi_ant_per_world_model).shape, (W, A, L)
        )
        self.assertEqual(
            multi_ant_per_world_squeeze_default.get_attribute("body_mass", multi_ant_per_world_model).shape, (W, A, L)
        )

        multi_ant_per_world_squeeze_false = ArticulationView(
            multi_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=False
        )
        self.assertEqual(multi_ant_per_world_squeeze_false.get_root_transforms(multi_ant_per_world_model).shape, (W, A))
        self.assertEqual(
            multi_ant_per_world_squeeze_false.get_link_transforms(multi_ant_per_world_model).shape, (W, A, L)
        )
        self.assertEqual(
            multi_ant_per_world_squeeze_false.get_attribute("body_mass", multi_ant_per_world_model).shape, (W, A, L)
        )

        multi_ant_per_world_squeeze_true = ArticulationView(
            multi_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=True
        )
        self.assertEqual(multi_ant_per_world_squeeze_true.get_root_transforms(multi_ant_per_world_model).shape, (W, A))
        self.assertEqual(
            multi_ant_per_world_squeeze_true.get_link_transforms(multi_ant_per_world_model).shape, (W, A, L)
        )
        self.assertEqual(
            multi_ant_per_world_squeeze_true.get_attribute("body_mass", multi_ant_per_world_model).shape, (W, A, L)
        )

        multi_ant_per_world_squeeze_0 = ArticulationView(
            multi_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(0,)
        )
        self.assertEqual(multi_ant_per_world_squeeze_0.get_root_transforms(multi_ant_per_world_model).shape, (W, A))
        self.assertEqual(multi_ant_per_world_squeeze_0.get_link_transforms(multi_ant_per_world_model).shape, (W, A, L))
        self.assertEqual(
            multi_ant_per_world_squeeze_0.get_attribute("body_mass", multi_ant_per_world_model).shape, (W, A, L)
        )

        multi_ant_per_world_squeeze_1 = ArticulationView(
            multi_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(1,)
        )
        self.assertEqual(multi_ant_per_world_squeeze_1.get_root_transforms(multi_ant_per_world_model).shape, (W, A))
        self.assertEqual(multi_ant_per_world_squeeze_1.get_link_transforms(multi_ant_per_world_model).shape, (W, A, L))
        self.assertEqual(
            multi_ant_per_world_squeeze_1.get_attribute("body_mass", multi_ant_per_world_model).shape, (W, A, L)
        )

        multi_ant_per_world_squeeze_2 = ArticulationView(
            multi_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(2,)
        )
        self.assertEqual(multi_ant_per_world_squeeze_2.get_root_transforms(multi_ant_per_world_model).shape, (W, A))
        self.assertEqual(multi_ant_per_world_squeeze_2.get_link_transforms(multi_ant_per_world_model).shape, (W, A, L))
        self.assertEqual(
            multi_ant_per_world_squeeze_2.get_attribute("body_mass", multi_ant_per_world_model).shape, (W, A, L)
        )

        multi_ant_per_world_squeeze_01 = ArticulationView(
            multi_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(0, 1)
        )
        self.assertEqual(multi_ant_per_world_squeeze_01.get_root_transforms(multi_ant_per_world_model).shape, (W, A))
        self.assertEqual(multi_ant_per_world_squeeze_01.get_link_transforms(multi_ant_per_world_model).shape, (W, A, L))
        self.assertEqual(
            multi_ant_per_world_squeeze_01.get_attribute("body_mass", multi_ant_per_world_model).shape, (W, A, L)
        )

        multi_ant_per_world_squeeze_12 = ArticulationView(
            multi_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(1, 2)
        )
        self.assertEqual(multi_ant_per_world_squeeze_12.get_root_transforms(multi_ant_per_world_model).shape, (W, A))
        self.assertEqual(multi_ant_per_world_squeeze_12.get_link_transforms(multi_ant_per_world_model).shape, (W, A, L))
        self.assertEqual(
            multi_ant_per_world_squeeze_12.get_attribute("body_mass", multi_ant_per_world_model).shape, (W, A, L)
        )

        multi_ant_per_world_squeeze_02 = ArticulationView(
            multi_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(0, 2)
        )
        self.assertEqual(multi_ant_per_world_squeeze_02.get_root_transforms(multi_ant_per_world_model).shape, (W, A))
        self.assertEqual(multi_ant_per_world_squeeze_02.get_link_transforms(multi_ant_per_world_model).shape, (W, A, L))
        self.assertEqual(
            multi_ant_per_world_squeeze_02.get_attribute("body_mass", multi_ant_per_world_model).shape, (W, A, L)
        )

        multi_ant_per_world_squeeze_012 = ArticulationView(
            multi_ant_per_world_model, "ant", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=(0, 1, 2)
        )
        self.assertEqual(multi_ant_per_world_squeeze_012.get_root_transforms(multi_ant_per_world_model).shape, (W, A))
        self.assertEqual(
            multi_ant_per_world_squeeze_012.get_link_transforms(multi_ant_per_world_model).shape, (W, A, L)
        )
        self.assertEqual(
            multi_ant_per_world_squeeze_012.get_attribute("body_mass", multi_ant_per_world_model).shape, (W, A, L)
        )

    def test_selection_squeeze_cubes(self):
        cube = newton.ModelBuilder()
        body = cube.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
        cube.add_shape_box(body)
        j = cube.add_joint_free(body)
        cube.add_articulation([j], key="cube")

        W = 10  # num worlds
        A = 3  # num cubes per world

        #
        # scene with just one cube
        #
        single_cube_model = cube.finalize()

        single_cube_squeeze_default = ArticulationView(
            single_cube_model, "cube", exclude_joint_types=[newton.JointType.FREE]
        )
        self.assertEqual(single_cube_squeeze_default.get_root_transforms(single_cube_model).shape, (1,))
        self.assertEqual(single_cube_squeeze_default.get_link_transforms(single_cube_model).shape, (1, 1))
        self.assertEqual(single_cube_squeeze_default.get_attribute("body_mass", single_cube_model).shape, (1, 1))

        single_cube_squeeze_false = ArticulationView(
            single_cube_model, "cube", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=False
        )
        self.assertEqual(single_cube_squeeze_false.get_root_transforms(single_cube_model).shape, (1, 1))
        self.assertEqual(single_cube_squeeze_false.get_link_transforms(single_cube_model).shape, (1, 1, 1))
        self.assertEqual(single_cube_squeeze_false.get_attribute("body_mass", single_cube_model).shape, (1, 1, 1))

        single_cube_squeeze_true = ArticulationView(
            single_cube_model, "cube", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=True
        )
        self.assertEqual(single_cube_squeeze_true.get_root_transforms(single_cube_model).shape, (1,))
        self.assertEqual(single_cube_squeeze_true.get_link_transforms(single_cube_model).shape, (1,))
        self.assertEqual(single_cube_squeeze_true.get_attribute("body_mass", single_cube_model).shape, (1,))

        #
        # scene with one cube per world
        #
        single_cube_per_world_scene = newton.ModelBuilder()
        single_cube_per_world_scene.replicate(cube, num_worlds=W)
        single_cube_per_world_model = single_cube_per_world_scene.finalize()

        single_cube_per_world_squeeze_default = ArticulationView(
            single_cube_per_world_model, "cube", exclude_joint_types=[newton.JointType.FREE]
        )
        self.assertEqual(
            single_cube_per_world_squeeze_default.get_root_transforms(single_cube_per_world_model).shape, (W,)
        )
        self.assertEqual(
            single_cube_per_world_squeeze_default.get_link_transforms(single_cube_per_world_model).shape, (W, 1)
        )
        self.assertEqual(
            single_cube_per_world_squeeze_default.get_attribute("body_mass", single_cube_per_world_model).shape, (W, 1)
        )

        single_cube_per_world_squeeze_false = ArticulationView(
            single_cube_per_world_model, "cube", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=False
        )
        self.assertEqual(
            single_cube_per_world_squeeze_false.get_root_transforms(single_cube_per_world_model).shape, (W, 1)
        )
        self.assertEqual(
            single_cube_per_world_squeeze_false.get_link_transforms(single_cube_per_world_model).shape, (W, 1, 1)
        )
        self.assertEqual(
            single_cube_per_world_squeeze_false.get_attribute("body_mass", single_cube_per_world_model).shape, (W, 1, 1)
        )

        single_cube_per_world_squeeze_true = ArticulationView(
            single_cube_per_world_model, "cube", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=True
        )
        self.assertEqual(
            single_cube_per_world_squeeze_true.get_root_transforms(single_cube_per_world_model).shape, (W,)
        )
        self.assertEqual(
            single_cube_per_world_squeeze_true.get_link_transforms(single_cube_per_world_model).shape, (W,)
        )
        self.assertEqual(
            single_cube_per_world_squeeze_true.get_attribute("body_mass", single_cube_per_world_model).shape, (W,)
        )

        #
        # scene with multiple ants per world
        #
        multi_cube_world = newton.ModelBuilder()
        for i in range(A):
            multi_cube_world.add_builder(cube, xform=wp.transform((0.0, 0.0, 1.0 + i), wp.quat_identity()))
        multi_cube_per_world_scene = newton.ModelBuilder()
        multi_cube_per_world_scene.replicate(multi_cube_world, num_worlds=W)
        multi_cube_per_world_model = multi_cube_per_world_scene.finalize()

        multi_cube_per_world_squeeze_default = ArticulationView(
            multi_cube_per_world_model, "cube", exclude_joint_types=[newton.JointType.FREE]
        )
        self.assertEqual(
            multi_cube_per_world_squeeze_default.get_root_transforms(multi_cube_per_world_model).shape, (W, A)
        )
        self.assertEqual(
            multi_cube_per_world_squeeze_default.get_link_transforms(multi_cube_per_world_model).shape, (W, A, 1)
        )
        self.assertEqual(
            multi_cube_per_world_squeeze_default.get_attribute("body_mass", multi_cube_per_world_model).shape, (W, A, 1)
        )

        multi_cube_per_world_squeeze_false = ArticulationView(
            multi_cube_per_world_model, "cube", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=False
        )
        self.assertEqual(
            multi_cube_per_world_squeeze_false.get_root_transforms(multi_cube_per_world_model).shape, (W, A)
        )
        self.assertEqual(
            multi_cube_per_world_squeeze_false.get_link_transforms(multi_cube_per_world_model).shape, (W, A, 1)
        )
        self.assertEqual(
            multi_cube_per_world_squeeze_false.get_attribute("body_mass", multi_cube_per_world_model).shape, (W, A, 1)
        )

        multi_cube_per_world_squeeze_true = ArticulationView(
            multi_cube_per_world_model, "cube", exclude_joint_types=[newton.JointType.FREE], squeeze_axes=True
        )
        self.assertEqual(
            multi_cube_per_world_squeeze_true.get_root_transforms(multi_cube_per_world_model).shape, (W, A)
        )
        self.assertEqual(
            multi_cube_per_world_squeeze_true.get_link_transforms(multi_cube_per_world_model).shape, (W, A)
        )
        self.assertEqual(
            multi_cube_per_world_squeeze_true.get_attribute("body_mass", multi_cube_per_world_model).shape, (W, A)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
