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

import functools
from fnmatch import fnmatch

import numpy as np
import warp as wp
from warp.types import is_array

import newton.core.articulation
from newton import Control, Model, State


class AttributeRegistry:
    def __init__(self):
        # look up indexing mode for known attributes
        self._indexing_mode: dict[str:str] = {}

        # addressable by joint id
        self.register_attribute("joint_type", "joint")
        self.register_attribute("joint_parent", "joint")
        self.register_attribute("joint_child", "joint")
        self.register_attribute("joint_ancestor", "joint")
        self.register_attribute("joint_X_p", "joint")
        self.register_attribute("joint_X_c", "joint")
        self.register_attribute("joint_axis_start", "joint")
        self.register_attribute("joint_axis_dim", "joint")
        self.register_attribute("joint_enabled", "joint")
        self.register_attribute("joint_twist_lower", "joint")
        self.register_attribute("joint_twist_upper", "joint")

        # addressable by joint coord offset
        self.register_attribute("joint_q", "joint_coord")

        # addressable by joint dof offset
        self.register_attribute("joint_qd", "joint_dof")
        self.register_attribute("joint_f", "joint_dof")
        self.register_attribute("joint_armature", "joint_dof")

        # addressable by joint axis offset
        self.register_attribute("joint_target", "joint_axis")
        self.register_attribute("joint_axis", "joint_axis")
        self.register_attribute("joint_target_ke", "joint_axis")
        self.register_attribute("joint_target_kd", "joint_axis")
        self.register_attribute("joint_axis_mode", "joint_axis")
        self.register_attribute("joint_limit_lower", "joint_axis")
        self.register_attribute("joint_limit_upper", "joint_axis")
        self.register_attribute("joint_limit_ke", "joint_axis")
        self.register_attribute("joint_limit_kd", "joint_axis")

        # addressable by body id
        self.register_attribute("body_q", "body")
        self.register_attribute("body_qd", "body")
        self.register_attribute("body_com", "body")
        self.register_attribute("body_inertia", "body")
        self.register_attribute("body_inv_inertia", "body")
        self.register_attribute("body_mass", "body")
        self.register_attribute("body_inv_mass", "body")
        self.register_attribute("body_f", "body")

    def register_attribute(self, name: str, mode: str):
        self._indexing_mode[name] = mode

    def get_indexing_mode(self, attribute_name: str):
        return self._indexing_mode[attribute_name]


attribute_registry = AttributeRegistry()


@wp.kernel
def set_mask_kernel(indices: wp.array(dtype=int), mask: wp.array(dtype=bool)):
    tid = wp.tid()
    mask[indices[tid]] = True


@wp.kernel
def set_mask_indexed_kernel(
    indices: wp.array(dtype=int), indices_indices: wp.array(dtype=int), mask: wp.array(dtype=bool)
):
    tid = wp.tid()
    mask[indices[indices_indices[tid]]] = True


@wp.kernel
def set_articulation_root_transforms_kernel(
    articulation_indices: wp.array(dtype=int),
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    root_transforms: wp.array(dtype=wp.transform),
    env_indices: wp.array(dtype=int),
    # outputs
    joint_q: wp.array(dtype=float),
    joint_X_p: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    idx = env_indices[tid]
    root_pose = root_transforms[idx]
    articulation = articulation_indices[idx]
    joint_start = articulation_start[articulation]
    q_start = joint_q_start[joint_start]

    if joint_type[joint_start] == newton.JOINT_FREE:
        for i in range(7):
            joint_q[q_start + i] = root_pose[i]
    elif joint_type[joint_start] == newton.JOINT_FIXED:
        joint_X_p[joint_start] = root_pose


@wp.kernel
def get_articulation_root_transforms_kernel(
    articulation_indices: wp.array(dtype=int),
    articulation_start: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    # outputs
    root_xforms: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    articulation = articulation_indices[tid]
    joint_start = articulation_start[articulation]

    if joint_parent[joint_start] != -1:
        root_body = joint_parent[joint_start]
    else:
        root_body = joint_child[joint_start]

    root_pose = body_q[root_body]

    root_xforms[tid] = root_pose


@wp.kernel
def set_articulation_root_velocities_kernel(
    articulation_indices: wp.array(dtype=int),
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    root_vels: wp.array(dtype=wp.spatial_vector),
    env_indices: wp.array(dtype=int),
    # outputs
    joint_qd: wp.array(dtype=float),
):
    tid = wp.tid()
    idx = env_indices[tid]
    articulation = articulation_indices[idx]
    joint_start = articulation_start[articulation]
    qd_start = joint_qd_start[joint_start]
    root_vel = root_vels[idx]

    if joint_type[joint_start] == newton.JOINT_FREE:
        for i in range(6):
            joint_qd[qd_start + i] = root_vel[i]


@wp.kernel
def get_articulation_root_velocities_kernel(
    articulation_indices: wp.array(dtype=int),
    articulation_start: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    body_qd: wp.array(dtype=wp.spatial_vector),
    # outputs
    root_vels: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    articulation = articulation_indices[tid]
    joint_start = articulation_start[articulation]

    if joint_parent[joint_start] != -1:
        root_body = joint_parent[joint_start]
    else:
        root_body = joint_child[joint_start]

    root_vels[tid] = body_qd[root_body]


class ArticulationView:
    def __init__(self, model: Model, pattern: str, include_free_joint: bool = False, verbose: bool | None = None):
        self.model = model
        self.device = model.device
        self.include_free_joint = include_free_joint

        if verbose is None:
            verbose = wp.config.verbose

        articulation_ids = []
        for id, key in enumerate(model.articulation_key):
            if fnmatch(key, pattern):
                articulation_ids.append(id)

        count = len(articulation_ids)
        if count == 0:
            raise KeyError("No matching articulations")

        # FIXME: avoid this readback?
        articulation_start = model.articulation_start.numpy()
        joint_type = model.joint_type.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_axis_start = model.joint_axis_start.numpy()
        joint_axis_dim = model.joint_axis_dim.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()

        # FIXME:
        # - this assumes homogeneous envs with one selected articulation per env
        # - we're going to have problems if there are any bodies or joints in the "global" env

        arti_0 = articulation_ids[0]

        joint_begin = articulation_start[arti_0]
        joint_end = articulation_start[arti_0 + 1]  # FIXME: is this always correct?
        joint_last = joint_end - 1

        links = {}
        for joint_id in range(joint_begin, joint_end):
            if joint_parent[joint_id] != -1:
                links[int(joint_parent[joint_id])] = None
            if joint_child[joint_id] != -1:
                links[int(joint_child[joint_id])] = None

        links = sorted(links.keys())

        # print stuff for debugging
        if verbose:
            print(f"num_joints: {joint_end - joint_begin}")
            for joint_id in range(joint_begin, joint_end):
                joint_name = model.joint_key[joint_id]
                print(f"  joint {joint_name}:")
                print(f"    bodies: {joint_parent[joint_id]} -> {joint_child[joint_id]}")
                print(f"    axis_start: {joint_axis_start[joint_id]}")
                print(f"    axis_dim: {joint_axis_dim[joint_id]}")
            num_links = len(links)
            print(f"num_links: {num_links}, {links}")
            for body_id in links:
                print(f"  {model.body_key[body_id]}")

        # if the root joint is a free joint, skip it
        if joint_type[joint_begin] == newton.JOINT_FREE and not include_free_joint:
            joint_begin += 1

        joint_coord_begin = joint_q_start[joint_begin]
        joint_coord_end = joint_q_start[joint_end]
        joint_dof_begin = joint_qd_start[joint_begin]
        joint_dof_end = joint_qd_start[joint_end]
        joint_axis_begin = joint_axis_start[joint_begin]
        joint_axis_end = joint_axis_start[joint_last] + joint_axis_dim[joint_last][0] + joint_axis_dim[joint_last][1]
        body_begin = links[0]
        body_end = links[-1] + 1

        self.articulation_indices = wp.array(articulation_ids, dtype=int, device=self.device)

        # create articulation mask
        self.articulation_mask = wp.zeros(model.articulation_count, dtype=bool, device=self.device)
        wp.launch(
            set_mask_kernel, dim=count, inputs=[self.articulation_indices, self.articulation_mask], device=self.device
        )

        self.all_indices = wp.array(np.arange(count, dtype=np.int32), device=self.device)

        # set some counting properties
        self.count = count
        self.link_count = len(links)
        self.joint_count = joint_end - joint_begin
        self.joint_coord_count = joint_coord_end - joint_coord_begin
        self.joint_dof_count = joint_dof_end - joint_dof_begin
        self.joint_axis_count = joint_axis_end - joint_axis_begin

        self.joint_names = []
        self.joint_coord_names = []
        self.joint_dof_names = []
        self.joint_axis_names = []
        self.body_names = []

        def get_name_from_key(key):
            return key.split("/")[-1]

        for joint_id in range(joint_begin, joint_end):
            joint_name = get_name_from_key(model.joint_key[joint_id])
            self.joint_names.append(joint_name)
            num_coords = joint_q_start[joint_id + 1] - joint_q_start[joint_id]
            if num_coords == 1:
                self.joint_coord_names.append(joint_name)
            elif num_coords > 1:
                for coord in range(num_coords):
                    self.joint_coord_names.append(f"{joint_name}:{coord}")
            num_dofs = joint_qd_start[joint_id + 1] - joint_qd_start[joint_id]
            if num_dofs == 1:
                self.joint_dof_names.append(joint_name)
            elif num_dofs > 1:
                for dof in range(num_dofs):
                    self.joint_dof_names.append(f"{joint_name}:{dof}")
            num_axes = joint_axis_dim[joint_id][0] + joint_axis_dim[joint_id][1]
            if num_axes == 1:
                self.joint_axis_names.append(joint_name)
            elif num_axes > 1:
                for axis in range(num_axes):
                    self.joint_axis_names.append(f"{joint_name}:{axis}")

        for body_id in range(body_begin, body_end):
            self.body_names.append(get_name_from_key(model.body_key[body_id]))

        print("Link names:")
        print(self.body_names)
        print("Joint names:")
        print(self.joint_names)
        print("Joint axis names:")
        print(self.joint_axis_names)
        # print("Joint coord names:")
        # print(self.joint_coord_names)
        # print("Joint dof names:")
        # print(self.joint_dof_names)

        # slices by indexing mode
        self._slices = {
            "joint": slice(int(joint_begin), int(joint_end)),
            "joint_coord": slice(int(joint_coord_begin), int(joint_coord_end)),
            "joint_dof": slice(int(joint_dof_begin), int(joint_dof_end)),
            "joint_axis": slice(int(joint_axis_begin), int(joint_axis_end)),
            "body": slice(int(body_begin), int(body_end)),
        }

        self._root_transforms = None
        self._root_velocities = None

    @functools.lru_cache(maxsize=None)  # noqa
    def _get_cached_attribute(self, name: str, source: Model | State | Control):
        # get the attribute array
        attrib = getattr(source, name)
        assert isinstance(attrib, wp.array)

        # reshape with batch dim at front
        assert attrib.shape[0] % self.count == 0
        batched_shape = (self.count, attrib.shape[0] // self.count, *attrib.shape[1:])

        # get attribute slice
        indexing_mode = attribute_registry.get_indexing_mode(name)
        attrib_slice = self._slices[indexing_mode]

        # create strided array
        attrib = attrib.reshape(batched_shape)
        attrib = attrib[:, attrib_slice]

        return attrib

    def get_attribute(self, name: str, source: Model | State | Control):
        return self._get_cached_attribute(name, source)

    def set_attribute(self, name: str, target: Model | State | Control, values, indices=None):
        attrib = self._get_cached_attribute(name, target)
        if not is_array(values):
            values = wp.array(values, dtype=attrib.dtype, shape=attrib.shape, device=self.device)
        if indices is not None:
            if not is_array(indices):
                indices = wp.array(indices, dtype=int, device=self.device)
            attrib = wp.indexedarray(attrib, [indices])
            values = wp.indexedarray(values, [indices])
        wp.copy(attrib, values)

    # convenience wrappers to align with legacy tensor API
    # TODO: do we want this?

    # def get_link_transforms(self, source, copy=False):
    #     return self.get_attribute("body_q", source, copy=copy)

    # def get_link_velocities(self, source, copy=False):
    #     return self.get_attribute("body_qd", source, copy=copy)

    # ...

    def get_root_transforms(self, source: Model | State):
        """
        Get the root transforms of the articulations.

        Args:
            source (Model | State): Where to get the root transforms (Model or State).

        Returns:
            array: The root transforms (dtype=wp.transform).
        """
        if self._root_transforms is None:
            self._root_transforms = wp.empty(self.count, dtype=wp.transform, device=self.device)

        wp.launch(
            get_articulation_root_transforms_kernel,
            self.count,
            inputs=[
                self.articulation_indices,
                self.model.articulation_start,
                self.model.joint_parent,
                self.model.joint_child,
                source.body_q,
            ],
            outputs=[
                self._root_transforms,
            ],
            device=self.device,
        )

        return self._root_transforms

    def set_root_transforms(self, target: Model | State, root_transforms: wp.array, indices=None):
        """
        Set the root transforms of the articulations.
        Call `eval_fk()` to apply changes to all articulation links.

        Args:
            target (Model | State): Where to set the root transforms (Model or State).
            root_transforms (array): The root transforms to set (dtype=wp.transform).
        """

        if not is_array(root_transforms):
            root_transforms = wp.array(root_transforms, dtype=wp.transform, device=self.device)

        assert len(root_transforms) == self.count, "Root poses should be provided for each articulation"

        if indices is not None:
            if not is_array(indices):
                indices = wp.array(indices, dtype=int, device=self.device)
        else:
            indices = self.all_indices

        wp.launch(
            set_articulation_root_transforms_kernel,
            indices.size,
            inputs=[
                self.articulation_indices,
                self.model.articulation_start,
                self.model.joint_type,
                self.model.joint_q_start,
                root_transforms,
                indices,
            ],
            outputs=[
                target.joint_q,
                self.model.joint_X_p,  # hmmm
            ],
            device=self.device,
        )

    def get_root_velocities(self, source: Model | State):
        """
        Get the root velocities of the articulations.

        Args:
            source (Model | State): Where to get the root velocities (Model or State).

        Returns:
            array: The root velocities (dtype=wp.spatial_vector).
        """
        if self._root_velocities is None:
            self._root_velocities = wp.empty(self.count, dtype=wp.spatial_vector, device=self.device)

        wp.launch(
            get_articulation_root_velocities_kernel,
            self.count,
            inputs=[
                self.articulation_indices,
                self.model.articulation_start,
                self.model.joint_parent,
                self.model.joint_child,
                source.body_qd,
            ],
            outputs=[
                self._root_velocities,
            ],
            device=self.device,
        )

        return self._root_velocities

    def set_root_velocities(self, target: Model | State, root_vels: wp.array, indices=None):
        """
        Set the root velocities of the articulations.

        Args:
            target (Model | State): Where to set the root velocities (Model or State).
            root_vels (array): The root velocities to set (dtype=wp.spatial_vector).
        """

        if not is_array(root_vels):
            root_vels = wp.array(root_vels, dtype=wp.spatial_vector, device=self.device)

        assert len(root_vels) == self.count, "Root velocities should be provided for each articulation"

        if indices is not None:
            if not is_array(indices):
                indices = wp.array(indices, dtype=int, device=self.device)
        else:
            indices = self.all_indices

        wp.launch(
            set_articulation_root_velocities_kernel,
            indices.size,
            inputs=[
                self.articulation_indices,
                self.model.articulation_start,
                self.model.joint_type,
                self.model.joint_qd_start,
                root_vels,
                indices,
            ],
            outputs=[
                target.joint_qd,
            ],
            device=self.device,
        )

    def eval_fk(self, target: Model | State, indices=None):
        if indices is not None:
            # create a custom mask for builtin eval_fk()
            # TODO: something more efficient?
            if not is_array(indices):
                indices = wp.array(indices, dtype=int, device=self.device)
            mask = wp.zeros(self.model.articulation_count, dtype=bool, device=self.device)
            wp.launch(set_mask_indexed_kernel, dim=indices.size, inputs=[self.articulation_indices, indices, mask])
        else:
            mask = self.articulation_mask

        newton.core.articulation.eval_fk(self.model, target.joint_q, target.joint_qd, target, mask=mask)
