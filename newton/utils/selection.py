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

from fnmatch import fnmatch

import warp as wp
from warp.types import is_array

from newton import Control, Model, State


@wp.kernel
def set_mask_kernel(indices: wp.array(dtype=int), mask: wp.array(dtype=bool)):
    tid = wp.tid()
    mask[indices[tid]] = True


class ArticulationView:
    def __init__(self, model: Model, pattern: str, include_root_joint=True):
        articulation_ids = []
        for id, key in enumerate(model.articulation_key):
            if fnmatch(key, pattern):
                articulation_ids.append(id)

        count = len(articulation_ids)
        if count == 0:
            raise KeyError("No matching articulations")

        articulation_start = model.articulation_start.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_axis_start = model.joint_axis_start.numpy()
        joint_axis_dim = model.joint_axis_dim.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()

        # FIXME: this assumes homogeneous envs with one selected articulation per env

        arti_0 = articulation_ids[0]

        joint_begin = articulation_start[arti_0]
        joint_end = articulation_start[arti_0 + 1]

        # FIXME: is this always correct?
        num_joints = joint_end - joint_begin
        print(f"  num_joints: {num_joints}")

        links = {}
        for joint_id in range(joint_begin, joint_end):
            joint_name = model.joint_key[joint_id]
            print(f"    joint {joint_name}:")
            print(f"      bodies: {joint_parent[joint_id]} -> {joint_child[joint_id]}")
            print(f"      axis_start: {joint_axis_start[joint_id]}")
            print(f"      axis_dim: {joint_axis_dim[joint_id]}")
            if joint_parent[joint_id] != -1:
                links[int(joint_parent[joint_id])] = None
            if joint_child[joint_id] != -1:
                links[int(joint_child[joint_id])] = None

        links = sorted(links.keys())
        num_links = len(links)
        print(f"  num_links: {num_links}, {links}")
        for body_id in links:
            print(f"    {model.body_key[body_id]}")

        self.attrib_shapes = {}
        self.attrib_slices = {}

        if not include_root_joint:
            # check if a root joint is present and skip it
            if joint_parent[joint_begin] == -1:
                joint_begin += 1

        self.attrib_shapes["joint_q"] = (count, model.joint_q.size // count)
        self.attrib_slices["joint_q"] = (
            slice(0, count),
            slice(int(joint_q_start[joint_begin]), int(joint_q_start[joint_end])),
        )
        self.attrib_shapes["joint_qd"] = (count, model.joint_qd.size // count)
        self.attrib_slices["joint_qd"] = (
            slice(0, count),
            slice(int(joint_qd_start[joint_begin]), int(joint_qd_start[joint_end])),
        )

        joint_axis_begin = joint_axis_start[joint_begin]
        joint_axis_end = joint_axis_start[joint_end]
        self.attrib_shapes["joint_act"] = (count, model.joint_act.size // count)
        self.attrib_slices["joint_act"] = (slice(0, count), slice(int(joint_axis_begin), int(joint_axis_end)))

        body_begin = links[0]
        body_end = links[-1] + 1
        self.attrib_shapes["body_q"] = (count, model.body_q.shape[0] // count)
        self.attrib_slices["body_q"] = (slice(0, count), slice(body_begin, body_end))
        self.attrib_shapes["body_qd"] = (count, model.body_qd.shape[0] // count)
        self.attrib_slices["body_qd"] = (slice(0, count), slice(body_begin, body_end))

        # create articulation mask
        self._articulation_mask = wp.zeros(model.articulation_count, dtype=bool)
        indices = wp.array(articulation_ids, dtype=int)
        wp.launch(set_mask_kernel, dim=indices.shape, inputs=[indices, self._articulation_mask])

        # set some counting properties
        self._count = count
        self._link_count = len(links)
        self._joint_count = joint_end - joint_begin
        self._joint_axis_count = joint_axis_end - joint_axis_begin

    @property
    def articulation_mask(self) -> wp.array(dtype=bool):
        return self._articulation_mask

    @property
    def count(self) -> int:
        return self._count

    @property
    def link_count(self) -> int:
        return self._link_count

    @property
    def joint_count(self) -> int:
        return self._joint_count

    @property
    def joint_axis_count(self) -> int:
        return self._joint_axis_count

    def get_attribute_shape(self, name: str):
        shape = []
        for s in self.attrib_slices[name]:
            shape.append(s.stop - s.start)
        return tuple(shape)

    def get_attribute(self, name: str, source: Model | State | Control, copy=False):
        attrib = getattr(source, name)
        attrib = attrib.reshape(self.attrib_shapes[name])
        attrib = attrib[*self.attrib_slices[name]]
        if copy:
            return wp.clone(attrib)
        else:
            return attrib

    def set_attribute(self, name: str, target: Model | State | Control, values):
        attrib = getattr(target, name)
        attrib = attrib.reshape(self.attrib_shapes[name])
        attrib = attrib[*self.attrib_slices[name]]
        if not is_array(values):
            values = wp.array(values, dtype=attrib.dtype, shape=attrib.shape)
        wp.copy(attrib, values)

    # convenience wrappers to align with legacy tensor API
    # TODO: do we want this?

    # def get_link_transforms(self, source, copy=False):
    #     return self.get_attribute("body_q", source, copy=copy)

    # def get_link_velocities(self, source, copy=False):
    #     return self.get_attribute("body_qd", source, copy=copy)

    # ...
