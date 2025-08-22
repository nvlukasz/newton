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


from __future__ import annotations

import warp as wp
from warp.context import Devicelike


class Contacts:
    """
    Stores contact information for rigid and soft body collisions, to be consumed by a solver.

    This class manages buffers for contact data such as positions, normals, thicknesses, and shape indices
    for both rigid-rigid and soft-rigid contacts. The buffers are allocated on the specified device and can
    optionally require gradients for differentiable simulation.

    Attributes:
        rigid_contact_count (wp.array): Number of rigid contacts (shape: [1], dtype: wp.int32).
        rigid_contact_point_id (wp.array): Unique ID for each rigid contact point (shape: [rigid_contact_max], dtype: wp.int32).
        rigid_contact_shape0 (wp.array): Indices of the first shape in each rigid contact (shape: [rigid_contact_max], dtype: wp.int32).
        rigid_contact_shape1 (wp.array): Indices of the second shape in each rigid contact (shape: [rigid_contact_max], dtype: wp.int32).
        rigid_contact_point0 (wp.array): Contact point on shape0 in world coordinates (shape: [rigid_contact_max, 3], dtype: wp.vec3).
        rigid_contact_point1 (wp.array): Contact point on shape1 in world coordinates (shape: [rigid_contact_max, 3], dtype: wp.vec3).
        rigid_contact_offset0 (wp.array): Offset from shape0's origin to contact point (shape: [rigid_contact_max, 3], dtype: wp.vec3).
        rigid_contact_offset1 (wp.array): Offset from shape1's origin to contact point (shape: [rigid_contact_max, 3], dtype: wp.vec3).
        rigid_contact_normal (wp.array): Contact normal at each rigid contact (shape: [rigid_contact_max, 3], dtype: wp.vec3).
        rigid_contact_thickness0 (wp.array): Thickness at contact on shape0 (shape: [rigid_contact_max], dtype: wp.float32).
        rigid_contact_thickness1 (wp.array): Thickness at contact on shape1 (shape: [rigid_contact_max], dtype: wp.float32).
        rigid_contact_tids (wp.array): Thread IDs for each rigid contact (shape: [rigid_contact_max], dtype: wp.int32).
        rigid_contact_force (wp.array): Contact force at each rigid contact (currently unused) (shape: [rigid_contact_max, 3], dtype: wp.vec3).

        soft_contact_count (wp.array): Number of soft contacts (shape: [1], dtype: wp.int32).
        soft_contact_particle (wp.array): Indices of soft particles in contact (shape: [soft_contact_max], dtype: int).
        soft_contact_shape (wp.array): Indices of shapes in contact with soft particles (shape: [soft_contact_max], dtype: int).
        soft_contact_body_pos (wp.array): Contact position on the body (shape: [soft_contact_max, 3], dtype: wp.vec3).
        soft_contact_body_vel (wp.array): Contact velocity on the body (shape: [soft_contact_max, 3], dtype: wp.vec3).
        soft_contact_normal (wp.array): Contact normal for soft contacts (shape: [soft_contact_max, 3], dtype: wp.vec3).
        soft_contact_tids (wp.array): Thread IDs for each soft contact (shape: [soft_contact_max], dtype: int).

        requires_grad (bool): Whether buffers require gradients.
        rigid_contact_max (int): Maximum number of rigid contacts.
        soft_contact_max (int): Maximum number of soft contacts.

    .. note::
        This class is a temporary solution and its interface may change in the future.
    """

    def __init__(
        self,
        rigid_contact_max: int,
        soft_contact_max: int,
        requires_grad: bool = False,
        device: Devicelike = None,
    ):
        with wp.ScopedDevice(device):
            # rigid contacts
            self.rigid_contact_count = wp.zeros(1, dtype=wp.int32)
            self.rigid_contact_point_id = wp.zeros(rigid_contact_max, dtype=wp.int32)
            self.rigid_contact_shape0 = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            self.rigid_contact_shape1 = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            self.rigid_contact_point0 = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_point1 = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_offset0 = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_offset1 = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_normal = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.rigid_contact_thickness0 = wp.zeros(rigid_contact_max, dtype=wp.float32, requires_grad=requires_grad)
            self.rigid_contact_thickness1 = wp.zeros(rigid_contact_max, dtype=wp.float32, requires_grad=requires_grad)
            self.rigid_contact_tids = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            # to be filled by the solver (currently unused)
            self.rigid_contact_force = wp.zeros(rigid_contact_max, dtype=wp.vec3, requires_grad=requires_grad)

            # soft contacts
            self.soft_contact_count = wp.zeros(1, dtype=wp.int32)
            self.soft_contact_particle = wp.full(soft_contact_max, -1, dtype=int)
            self.soft_contact_shape = wp.full(soft_contact_max, -1, dtype=int)
            self.soft_contact_body_pos = wp.zeros(soft_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.soft_contact_body_vel = wp.zeros(soft_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.soft_contact_normal = wp.zeros(soft_contact_max, dtype=wp.vec3, requires_grad=requires_grad)
            self.soft_contact_tids = wp.full(soft_contact_max, -1, dtype=int)

        self.requires_grad = requires_grad

        self.rigid_contact_max = rigid_contact_max
        self.soft_contact_max = soft_contact_max

    def clear(self):
        """
        Clear all contact data, resetting counts and filling indices with -1.
        """
        self.rigid_contact_count.zero_()
        self.rigid_contact_shape0.fill_(-1)
        self.rigid_contact_shape1.fill_(-1)
        self.rigid_contact_tids.fill_(-1)
        self.rigid_contact_force.zero_()

        self.soft_contact_count.zero_()
        self.soft_contact_particle.fill_(-1)
        self.soft_contact_shape.fill_(-1)
        self.soft_contact_tids.fill_(-1)

    @property
    def device(self):
        """
        Returns the device on which the contact buffers are allocated.
        """
        return self.rigid_contact_count.device
