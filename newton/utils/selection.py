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

import newton.core.articulation
from newton import Control, Model, State


@wp.kernel
def set_mask_kernel(indices: wp.array(dtype=int), mask: wp.array(dtype=bool)):
    tid = wp.tid()
    mask[indices[tid]] = True


@wp.kernel
def set_articulation_root_transforms_kernel(
    articulation_indices: wp.array(dtype=int),
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    root_transforms: wp.array(dtype=wp.transform),
    env_offsets: wp.array(dtype=wp.vec3),
    # outputs
    joint_q: wp.array(dtype=float),
    joint_X_p: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    root_pose = root_transforms[tid]
    articulation = articulation_indices[tid]
    joint_start = articulation_start[articulation]
    q_start = joint_q_start[joint_start]
    env_offset = env_offsets[tid]

    # apply env offset
    root_pose = wp.transform(
        wp.vec3(root_pose[0], root_pose[1], root_pose[2]) + env_offset,
        wp.quat(root_pose[3], root_pose[4], root_pose[5], root_pose[6]),
    )

    if joint_type[joint_start] == newton.JOINT_FREE:
        for i in range(7):
            joint_q[q_start + i] = root_pose[i]
    elif joint_type[joint_start] == newton.JOINT_FIXED:
        joint_X_p[joint_start] = root_pose


@wp.kernel
def get_articulation_root_transforms_kernel_v1(
    articulation_indices: wp.array(dtype=int),
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_q_start: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    env_offsets: wp.array(dtype=wp.vec3),
    # outputs
    root_transforms: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    articulation = articulation_indices[tid]
    joint_start = articulation_start[articulation]
    q_start = joint_q_start[joint_start]
    env_offset = env_offsets[tid]

    if joint_type[joint_start] == newton.JOINT_FREE:
        root_pose = wp.transform(
            wp.vec3(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2]),
            wp.quat(joint_q[q_start + 3], joint_q[q_start + 4], joint_q[q_start + 5], joint_q[q_start + 6]),
        )
    elif joint_type[joint_start] == newton.JOINT_FIXED:
        root_pose = joint_X_p[joint_start]

    # apply env offset
    root_pose = wp.transform(
        wp.vec3(root_pose[0], root_pose[1], root_pose[2]) - env_offset,
        wp.quat(root_pose[3], root_pose[4], root_pose[5], root_pose[6]),
    )

    root_transforms[tid] = root_pose


@wp.kernel
def get_articulation_root_transforms_kernel_v2(
    articulation_indices: wp.array(dtype=int),
    articulation_start: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    env_offsets: wp.array(dtype=wp.vec3),
    # outputs
    root_xforms: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    articulation = articulation_indices[tid]
    joint_start = articulation_start[articulation]
    env_offset = env_offsets[tid]

    if joint_parent[joint_start] != -1:
        root_body = joint_parent[joint_start]
    else:
        root_body = joint_child[joint_start]

    root_pose = body_q[root_body]

    # apply env offset
    root_pose = wp.transform(
        wp.vec3(root_pose[0], root_pose[1], root_pose[2]) - env_offset,
        wp.quat(root_pose[3], root_pose[4], root_pose[5], root_pose[6]),
    )

    root_xforms[tid] = root_pose


@wp.kernel
def set_articulation_root_velocities_kernel(
    articulation_indices: wp.array(dtype=int),
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    root_vels: wp.array(dtype=wp.spatial_vector),
    # outputs
    joint_qd: wp.array(dtype=float),
):
    tid = wp.tid()
    articulation = articulation_indices[tid]
    joint_start = articulation_start[articulation]
    qd_start = joint_qd_start[joint_start]
    root_vel = root_vels[tid]

    if joint_type[joint_start] == newton.JOINT_FREE:
        for i in range(6):
            joint_qd[qd_start + i] = root_vel[i]


@wp.kernel
def get_articulation_root_velocities_kernel_v1(
    articulation_indices: wp.array(dtype=int),
    articulation_start: wp.array(dtype=int),
    joint_qd: wp.array(dtype=float),
    joint_type: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    # outputs
    root_vels: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    articulation = articulation_indices[tid]
    joint_start = articulation_start[articulation]
    qd_start = joint_qd_start[joint_start]

    if joint_type[joint_start] == newton.JOINT_FREE:
        root_vel = wp.spatial_vector(
            joint_qd[qd_start + 0],
            joint_qd[qd_start + 1],
            joint_qd[qd_start + 2],
            joint_qd[qd_start + 3],
            joint_qd[qd_start + 4],
            joint_qd[qd_start + 5],
        )
    elif joint_type[joint_start] == newton.JOINT_FIXED:
        root_vel = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    root_vels[tid] = root_vel


@wp.kernel
def get_articulation_root_velocities_kernel_v2(
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
    def __init__(self, model: Model, pattern: str, include_free_joint=False, env_offsets=None):
        self.model = model
        self.device = model.device

        articulation_ids = []
        for id, key in enumerate(model.articulation_key):
            if fnmatch(key, pattern):
                articulation_ids.append(id)

        count = len(articulation_ids)
        if count == 0:
            raise KeyError("No matching articulations")

        articulation_start = model.articulation_start.numpy()
        joint_type = model.joint_type.numpy()
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

        # if the root joint is a free joint, skip it
        if joint_type[joint_begin] == newton.JOINT_FREE and not include_free_joint:
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

        self.articulation_indices = wp.array(articulation_ids, dtype=int, device=self.device)

        # create articulation mask
        self._articulation_mask = wp.zeros(model.articulation_count, dtype=bool)
        indices = wp.array(articulation_ids, dtype=int, device=self.device)
        wp.launch(set_mask_kernel, dim=indices.shape, inputs=[indices, self._articulation_mask], device=self.device)

        # env offsets
        self.env_offsets = wp.array(env_offsets, dtype=wp.vec3, device=self.device)

        # set some counting properties
        self._count = count
        self._link_count = len(links)
        self._joint_count = joint_end - joint_begin
        self._joint_axis_count = joint_axis_end - joint_axis_begin

        self._root_transforms = None
        self._root_velocities = None

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
            values = wp.array(values, dtype=attrib.dtype, shape=attrib.shape, device=self.device)
        wp.copy(attrib, values)

    # convenience wrappers to align with legacy tensor API
    # TODO: do we want this?

    # def get_link_transforms(self, source, copy=False):
    #     return self.get_attribute("body_q", source, copy=copy)

    # def get_link_velocities(self, source, copy=False):
    #     return self.get_attribute("body_qd", source, copy=copy)

    # ...

    def get_root_transforms(self, source: Model | State):
        if self._root_transforms is None:
            self._root_transforms = wp.empty(self.count, dtype=wp.transform, device=self.device)

        if False:
            # get transforms from root joints
            wp.launch(
                get_articulation_root_transforms_kernel_v1,
                self.count,
                inputs=[
                    self.articulation_indices,
                    self.model.articulation_start,
                    self.model.joint_type,
                    source.joint_q,
                    self.model.joint_q_start,
                    self.model.joint_X_p,  # hmmm
                    self.env_offsets,
                ],
                outputs=[
                    self._root_transforms,
                ],
                device=self.device,
            )
        else:
            # get transforms from root bodies
            wp.launch(
                get_articulation_root_transforms_kernel_v2,
                self.count,
                inputs=[
                    self.articulation_indices,
                    self.model.articulation_start,
                    self.model.joint_parent,
                    self.model.joint_child,
                    source.body_q,
                    self.env_offsets,
                ],
                outputs=[
                    self._root_transforms,
                ],
                device=self.device,
            )

        return self._root_transforms

    def set_root_transforms(self, target: Model | State, root_transforms: wp.array):
        """
        Sets the transforms of the root bodies in the articulations.
        Call `eval_fk()` to apply changes to all links.

        Args:
            target (Model | State): Where to set the root transforms (Model or State).
            root_transforms (array): The root transforms to set.
        """

        if not is_array(root_transforms):
            root_transforms = wp.array(root_transforms, dtype=wp.transform, device=self.device)

        assert len(root_transforms) == self.count, "Root poses should be provided for each articulation"

        wp.launch(
            set_articulation_root_transforms_kernel,
            self.count,
            inputs=[
                self.articulation_indices,
                self.model.articulation_start,
                self.model.joint_type,
                self.model.joint_q_start,
                root_transforms,
                self.env_offsets,
            ],
            outputs=[
                target.joint_q,
                self.model.joint_X_p,  # hmmm
            ],
            device=self.device,
        )

    def get_root_velocities(self, source: Model | State):
        if self._root_velocities is None:
            self._root_velocities = wp.empty(self.count, dtype=wp.spatial_vector, device=self.device)

        if False:
            # get velocities from root joints
            wp.launch(
                get_articulation_root_velocities_kernel_v1,
                self.count,
                inputs=[
                    self.articulation_indices,
                    self.model.articulation_start,
                    source.joint_qd,
                    self.model.joint_type,
                    self.model.joint_q_start,
                ],
                outputs=[
                    self._root_velocities,
                ],
                device=self.device,
            )
        else:
            # get velocities from root bodies
            wp.launch(
                get_articulation_root_velocities_kernel_v2,
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

    def set_root_velocities(self, target: Model | State, root_vels: wp.array):
        """
        Sets the velocities of the root bodies in the articulations.
        Call `eval_fk()` to apply changes to all links.

        Args:
            target (Model | State): Where to set the root velocities (Model or State).
            root_vels (array): The root velocities to set.
        """

        if not is_array(root_vels):
            root_vels = wp.array(root_vels, dtype=wp.spatial_vector, device=self.device)

        assert len(root_vels) == self.count, "Root velocities should be provided for each articulation"

        wp.launch(
            set_articulation_root_velocities_kernel,
            self.count,
            inputs=[
                self.articulation_indices,
                self.model.articulation_start,
                self.model.joint_type,
                self.model.joint_qd_start,
                root_vels,
            ],
            outputs=[
                target.joint_qd,
            ],
            device=self.device,
        )

    def eval_fk(self, target: Model | State):
        newton.core.articulation.eval_fk(self.model, target.joint_q, target.joint_qd, self.articulation_mask, target)
