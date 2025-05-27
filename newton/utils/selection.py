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


class ArticulationView:
    def __init__(self, model: Model, pattern: str, verbose: bool | None = None):
        self.model = model
        self.device = model.device

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
        num_links = len(links)

        # print stuff for debugging
        if verbose:
            print(f"num_joints: {joint_end - joint_begin}")
            for joint_id in range(joint_begin, joint_end):
                joint_name = model.joint_key[joint_id]
                print(f"  joint {joint_name}:")
                print(f"    bodies: {joint_parent[joint_id]} -> {joint_child[joint_id]}")
                print(f"    axis_start: {joint_axis_start[joint_id]}")
                print(f"    axis_dim: {joint_axis_dim[joint_id]}")
            print(f"num_links: {num_links}, {links}")
            for body_id in links:
                print(f"  {model.body_key[body_id]}")

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
        self.articulation_mask = wp.zeros(model.articulation_count, dtype=bool)
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

        newton.core.articulation.eval_fk(self.model, target.joint_q, target.joint_qd, mask, target)
