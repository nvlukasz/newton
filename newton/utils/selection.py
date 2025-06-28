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
from typing import Any

import warp as wp
from warp.types import is_array

import newton.sim
from newton import Control, Model, State
from newton.sim import JOINT_DISTANCE, JOINT_FIXED, JOINT_FREE


@wp.kernel
def set_model_articulation_mask_kernel(
    view_mask: wp.array(dtype=bool),  # mask in ArticulationView
    view_to_model_map: wp.array(dtype=int),  # maps index in ArticulationView to articulation index in Model
    articulation_mask: wp.array(dtype=bool),  # output: mask of Model articulation indices
):
    """
    Set Model articulation mask from a view mask in an ArticulationView.
    """
    tid = wp.tid()
    if view_mask[tid]:
        articulation_mask[view_to_model_map[tid]] = True


@wp.kernel
def set_articulation_attribute_1d_kernel(
    view_mask: wp.array(dtype=bool),  # mask in ArticulationView
    values: wp.array1d(dtype=Any),
    attrib: wp.array1d(dtype=Any),
):
    i = wp.tid()
    if view_mask[i]:
        attrib[i] = values[i]


@wp.kernel
def set_articulation_attribute_2d_kernel(
    view_mask: wp.array(dtype=bool),  # mask in ArticulationView
    values: wp.array2d(dtype=Any),
    attrib: wp.array2d(dtype=Any),
):
    i, j = wp.tid()
    if view_mask[i]:
        attrib[i, j] = values[i, j]


@wp.kernel
def set_articulation_attribute_3d_kernel(
    view_mask: wp.array(dtype=bool),  # mask in ArticulationView
    values: wp.array3d(dtype=Any),
    attrib: wp.array3d(dtype=Any),
):
    i, j, k = wp.tid()
    if view_mask[i]:
        attrib[i, j, k] = values[i, j, k]


@wp.kernel
def set_articulation_attribute_4d_kernel(
    view_mask: wp.array(dtype=bool),  # mask in ArticulationView
    values: wp.array4d(dtype=Any),
    attrib: wp.array4d(dtype=Any),
):
    i, j, k, l = wp.tid()
    if view_mask[i]:
        attrib[i, j, k, l] = values[i, j, k, l]


# explicit overloads to avoid module reloading
for dtype in [float, int, wp.transform, wp.spatial_vector]:
    wp.overload(
        set_articulation_attribute_1d_kernel, {"values": wp.array1d(dtype=dtype), "attrib": wp.array1d(dtype=dtype)}
    )
    wp.overload(
        set_articulation_attribute_2d_kernel, {"values": wp.array2d(dtype=dtype), "attrib": wp.array2d(dtype=dtype)}
    )
    wp.overload(
        set_articulation_attribute_3d_kernel, {"values": wp.array3d(dtype=dtype), "attrib": wp.array3d(dtype=dtype)}
    )
    wp.overload(
        set_articulation_attribute_4d_kernel, {"values": wp.array4d(dtype=dtype), "attrib": wp.array4d(dtype=dtype)}
    )


class ArticulationView:
    def __init__(
        self,
        model: Model,
        pattern: str,
        include_joints: list[str | int] | None = None,
        exclude_joints: list[str | int] | None = None,
        include_links: list[str | int] | None = None,
        exclude_links: list[str | int] | None = None,
        include_joint_types: list[int] | None = None,
        exclude_joint_types: list[int] | None = None,
        verbose: bool | None = None,
    ):
        self.model = model
        self.device = model.device

        if verbose is None:
            verbose = wp.config.verbose

        articulation_ids = []
        for id, key in enumerate(model.articulation_key):
            if fnmatch(key, pattern):
                articulation_ids.append(id)

        articulation_count = len(articulation_ids)
        if articulation_count == 0:
            raise KeyError("No matching articulations")

        # FIXME: avoid/reduce this readback?
        model_articulation_start = model.articulation_start.numpy()
        model_joint_type = model.joint_type.numpy()
        model_joint_child = model.joint_child.numpy()
        model_joint_q_start = model.joint_q_start.numpy()
        model_joint_qd_start = model.joint_qd_start.numpy()

        # FIXME:
        # - this assumes homogeneous envs with one selected articulation per env
        # - we're going to have problems if there are any bodies or joints in the "global" env

        arti_0 = articulation_ids[0]

        arti_joint_begin = model_articulation_start[arti_0]
        arti_joint_end = model_articulation_start[arti_0 + 1]  # FIXME: is this always correct?
        arti_joint_count = arti_joint_end - arti_joint_begin
        arti_link_count = arti_joint_count

        arti_joint_ids = []
        arti_joint_names = []
        arti_joint_types = []
        arti_link_ids = []
        arti_link_names = []

        def get_name_from_key(key):
            return key.split("/")[-1]

        for idx in range(arti_joint_count):
            joint_id = arti_joint_begin + idx
            arti_joint_ids.append(int(joint_id))
            arti_joint_names.append(get_name_from_key(model.joint_key[joint_id]))
            arti_joint_types.append(int(model_joint_type[joint_id]))
            link_id = model_joint_child[joint_id]
            arti_link_ids.append(int(link_id))
            arti_link_names.append(get_name_from_key(model.body_key[link_id]))

        # create joint inclusion set
        if include_joints is None and include_joint_types is None:
            joint_include_indices = set(range(arti_joint_count))
        else:
            joint_include_indices = set()
            if include_joints is not None:
                for id in include_joints:
                    if isinstance(id, str):
                        for idx, name in enumerate(arti_joint_names):
                            if fnmatch(name, id):
                                joint_include_indices.add(idx)
                    elif isinstance(id, int):
                        if id >= 0 and id < arti_joint_count:
                            joint_include_indices.add(id)
                    else:
                        raise TypeError(f"Joint ids must be strings or integers, got {id} of type {type(id)}")
            if include_joint_types is not None:
                for idx in range(arti_joint_count):
                    if arti_joint_types[idx] in include_joint_types:
                        joint_include_indices.add(idx)

        # create joint exclusion set
        joint_exclude_indices = set()
        if exclude_joints is not None:
            for id in exclude_joints:
                if isinstance(id, str):
                    for idx, name in enumerate(arti_joint_names):
                        if fnmatch(name, id):
                            joint_exclude_indices.add(idx)
                elif isinstance(id, int):
                    if id >= 0 and id < arti_joint_count:
                        joint_exclude_indices.add(id)
                else:
                    raise TypeError(f"Joint ids must be strings or integers, got {id} of type {type(id)}")
        if exclude_joint_types is not None:
            for idx in range(arti_joint_count):
                if arti_joint_types[idx] in exclude_joint_types:
                    joint_exclude_indices.add(idx)

        # create link inclusion set
        if include_links is None:
            link_include_indices = set(range(arti_link_count))
        else:
            link_include_indices = set()
            if include_links is not None:
                for id in include_links:
                    if isinstance(id, str):
                        for idx, name in enumerate(arti_link_names):
                            if fnmatch(name, id):
                                link_include_indices.add(idx)
                    elif isinstance(id, int):
                        if id >= 0 and id < arti_link_count:
                            link_include_indices.add(id)
                    else:
                        raise TypeError(f"Link ids must be strings or integers, got {id} of type {type(id)}")

        # create link exclusion set
        link_exclude_indices = set()
        if exclude_links is not None:
            for id in exclude_links:
                if isinstance(id, str):
                    for idx, name in enumerate(arti_link_names):
                        if fnmatch(name, id):
                            link_exclude_indices.add(idx)
                elif isinstance(id, int):
                    if id >= 0 and id < arti_link_count:
                        link_exclude_indices.add(id)
                else:
                    raise TypeError(f"Link ids must be strings or integers, got {id} of type {type(id)}")

        # compute selected indices
        selected_joint_indices = sorted(joint_include_indices - joint_exclude_indices)
        selected_link_indices = sorted(link_include_indices - link_exclude_indices)

        selected_joint_ids = []
        selected_joint_dof_ids = []
        selected_joint_coord_ids = []
        selected_link_ids = []

        self.joint_names = []
        self.joint_dof_names = []
        self.joint_dof_counts = []
        self.joint_coord_names = []
        self.joint_coord_counts = []
        self.body_names = []

        # populate info for selected joints and dofs
        for idx in selected_joint_indices:
            # joint
            joint_id = arti_joint_ids[idx]
            selected_joint_ids.append(joint_id)
            joint_name = get_name_from_key(model.joint_key[joint_id])
            self.joint_names.append(joint_name)
            # joint dofs
            dof_begin = model_joint_qd_start[joint_id]
            dof_end = model_joint_qd_start[joint_id + 1]
            dof_count = dof_end - dof_begin
            if dof_count == 1:
                self.joint_dof_names.append(joint_name)
                selected_joint_dof_ids.append(dof_begin)
            elif dof_count > 1:
                for dof in range(dof_count):
                    self.joint_dof_names.append(f"{joint_name}:{dof}")
                    selected_joint_dof_ids.append(dof_begin + dof)
            # joint coords
            coord_begin = model_joint_q_start[joint_id]
            coord_end = model_joint_q_start[joint_id + 1]
            coord_count = coord_end - coord_begin
            if coord_count == 1:
                self.joint_coord_names.append(joint_name)
                selected_joint_coord_ids.append(coord_begin)
            elif coord_count > 1:
                for coord in range(coord_count):
                    self.joint_coord_names.append(f"{joint_name}:{coord}")
                    selected_joint_coord_ids.append(coord_begin + coord)

        # populate info for selected links
        for idx in selected_link_indices:
            body_id = arti_link_ids[idx]
            selected_link_ids.append(body_id)
            self.body_names.append(get_name_from_key(model.body_key[body_id]))

        # selected counts
        self.count = articulation_count
        self.joint_count = len(selected_joint_ids)
        self.joint_dof_count = len(selected_joint_dof_ids)
        self.joint_coord_count = len(selected_joint_coord_ids)
        self.link_count = len(selected_link_ids)

        # support custom slicing and indexing
        self._arti_joint_begin = int(arti_joint_begin)
        self._arti_joint_end = int(arti_joint_end)
        self._arti_joint_dof_begin = int(model_joint_qd_start[arti_joint_begin])
        self._arti_joint_dof_end = int(model_joint_qd_start[arti_joint_end])
        self._arti_joint_coord_begin = int(model_joint_q_start[arti_joint_begin])
        self._arti_joint_coord_end = int(model_joint_q_start[arti_joint_end])

        if verbose:
            print(f"Articulation count: {self.count}")
            print(f"Link count:         {self.link_count}")
            print(f"Joint count:        {self.joint_count}")
            print(f"Joint DOF count:    {self.joint_dof_count}")

            print("Link names:")
            print(f"  {self.body_names}")
            print("Joint names:")
            print(f"  {self.joint_names}")
            print("Joint DOF names:")
            print(f"  {self.joint_dof_names}")
            # print("Joint coord names:")
            # print(f"  {self.joint_coord_names}")

        def is_contiguous_slice(indices):
            n = len(indices)
            if n > 1:
                for i in range(1, n):
                    if indices[i] != indices[i - 1] + 1:
                        return False
            return True

        # contiguous slices by attribute frequency
        #
        # FIXME: guard against empty selections
        #
        # TODO: set up indexed arrays for non-contiguous attributes
        #
        self._contiguous_slices = {}
        if is_contiguous_slice(selected_joint_ids):
            begin = selected_joint_ids[0]
            end = selected_joint_ids[-1] + 1
            self._contiguous_slices["joint"] = slice(int(begin), int(end))
        if is_contiguous_slice(selected_joint_dof_ids):
            begin = selected_joint_dof_ids[0]
            end = selected_joint_dof_ids[-1] + 1
            self._contiguous_slices["joint_dof"] = slice(int(begin), int(end))
        if is_contiguous_slice(selected_joint_coord_ids):
            begin = selected_joint_coord_ids[0]
            end = selected_joint_coord_ids[-1] + 1
            self._contiguous_slices["joint_coord"] = slice(int(begin), int(end))
        if is_contiguous_slice(selected_link_ids):
            begin = selected_link_ids[0]
            end = selected_link_ids[-1] + 1
            self._contiguous_slices["body"] = slice(int(begin), int(end))

        self.articulation_indices = wp.array(articulation_ids, dtype=int, device=self.device)

        # TODO: zero-stride mask would use less memory
        self.full_mask = wp.full(articulation_count, True, dtype=bool, device=self.device)

        # create articulation mask
        self.articulation_mask = wp.zeros(model.articulation_count, dtype=bool, device=self.device)
        wp.launch(
            set_model_articulation_mask_kernel,
            dim=articulation_count,
            inputs=[self.full_mask, self.articulation_indices, self.articulation_mask],
            device=self.device,
        )

        root_joint_type = arti_joint_types[arti_joint_begin]
        # fixed base means that all linear and angular degrees of freedom are locked at the root
        self.is_fixed_base = root_joint_type == JOINT_FIXED
        # floating base means that all linear and angular degrees of freedom are unlocked at the root
        # (though there might be constraints like distance)
        self.is_floating_base = root_joint_type in (JOINT_FREE, JOINT_DISTANCE)

    # ========================================================================================
    # Generic attribute API

    @functools.lru_cache(maxsize=None)  # noqa
    def _get_cached_attribute(self, name: str, source: Model | State | Control, _slice=None):
        # get the attribute array
        attrib = getattr(source, name)
        assert isinstance(attrib, wp.array)

        # reshape with batch dim at front
        assert attrib.shape[0] % self.count == 0
        batched_shape = (self.count, attrib.shape[0] // self.count, *attrib.shape[1:])

        # get attribute slice
        if _slice is not None:
            attrib_slice = _slice
        else:
            frequency = self.model.get_attribute_frequency(name)
            attrib_slice = self._contiguous_slices.get(frequency)
            if attrib_slice is None:
                # TODO
                raise NotImplementedError("Non-contiguous selections not supported yet")

        # create strided array
        attrib = attrib.reshape(batched_shape)
        attrib = attrib[:, attrib_slice]

        return attrib

    def _set_attribute_values(self, attrib, values, mask=None):
        if not is_array(values):
            values = wp.array(values, dtype=attrib.dtype, shape=attrib.shape, device=self.device, copy=False)
        assert values.shape == attrib.shape
        assert values.dtype == attrib.dtype

        # early out for in-place modifications
        if values.ptr == attrib.ptr:
            return

        # get mask
        if mask is None:
            mask = self.full_mask
        else:
            if not isinstance(mask, wp.array):
                mask = wp.array(mask, dtype=bool, shape=(self.count,), device=self.device, copy=False)
            assert mask.shape == (self.count,)

        # launch appropriate kernel based on attribute dimensionality
        # TODO: cache concrete overload per attribute?
        if attrib.ndim == 1:
            wp.launch(set_articulation_attribute_1d_kernel, dim=attrib.shape, inputs=[mask, values, attrib])
        elif attrib.ndim == 2:
            wp.launch(set_articulation_attribute_2d_kernel, dim=attrib.shape, inputs=[mask, values, attrib])
        elif attrib.ndim == 3:
            wp.launch(set_articulation_attribute_3d_kernel, dim=attrib.shape, inputs=[mask, values, attrib])
        elif attrib.ndim == 4:
            wp.launch(set_articulation_attribute_4d_kernel, dim=attrib.shape, inputs=[mask, values, attrib])
        else:
            raise NotImplementedError(f"Unsupported attribute with ndim={attrib.ndim}")

    def get_attribute(self, name: str, source: Model | State | Control):
        return self._get_cached_attribute(name, source)

    def set_attribute(self, name: str, target: Model | State | Control, values, mask=None):
        attrib = self._get_cached_attribute(name, target)
        self._set_attribute_values(attrib, values, mask=mask)

    # ========================================================================================
    # Convenience wrappers to align with legacy tensor API

    def get_root_transforms(self, source: Model | State):
        """
        Get the root transforms of the articulations.

        Args:
            source (Model | State): Where to get the root transforms (Model or State).

        Returns:
            array: The root transforms (dtype=wp.transform).
        """
        if self.is_floating_base:
            attrib_slice = slice(self._arti_joint_coord_begin, self._arti_joint_coord_begin + 7)
            attrib = self._get_cached_attribute("joint_q", source, _slice=attrib_slice)
        else:
            attrib_slice = slice(self._arti_joint_begin, self._arti_joint_begin + 1)
            attrib = self._get_cached_attribute("joint_X_p", self.model, _slice=attrib_slice)

        if attrib.dtype is wp.transform:
            return attrib
        else:
            return wp.array(attrib, dtype=wp.transform, device=self.device, copy=False)

    def set_root_transforms(self, target: Model | State, values: wp.array, mask=None):
        """
        Set the root transforms of the articulations.
        Call `eval_fk()` to apply changes to all articulation links.

        Args:
            target (Model | State): Where to set the root transforms (Model or State).
            values (array): The root transforms to set (dtype=wp.transform).
            mask (array): Mask of articulations in this ArticulationView (all by default).
        """
        if self.is_floating_base:
            attrib_slice = slice(self._arti_joint_coord_begin, self._arti_joint_coord_begin + 7)
            attrib = self._get_cached_attribute("joint_q", target, _slice=attrib_slice)
        else:
            attrib_slice = slice(self._arti_joint_begin, self._arti_joint_begin + 1)
            attrib = self._get_cached_attribute("joint_X_p", self.model, _slice=attrib_slice)

        self._set_attribute_values(attrib, values, mask=mask)

    def get_root_velocities(self, source: Model | State):
        """
        Get the root velocities of the articulations.

        Args:
            source (Model | State): Where to get the root velocities (Model or State).

        Returns:
            array: The root velocities (dtype=wp.spatial_vector).
        """
        if self.is_floating_base:
            attrib_slice = slice(self._arti_joint_dof_begin, self._arti_joint_dof_begin + 6)
            attrib = self._get_cached_attribute("joint_qd", source, _slice=attrib_slice)
        else:
            # FIXME? Non-floating articulations have no root velocities.
            return None

        if attrib.dtype is wp.spatial_vector:
            return attrib
        else:
            return wp.array(attrib, dtype=wp.spatial_vector, device=self.device, copy=False)

    def set_root_velocities(self, target: Model | State, values: wp.array, mask=None):
        """
        Set the root velocities of the articulations.

        Args:
            target (Model | State): Where to set the root velocities (Model or State).
            values (array): The root velocities to set (dtype=wp.spatial_vector).
            mask (array): Mask of articulations in this ArticulationView (all by default).
        """
        if self.is_floating_base:
            attrib_slice = slice(self._arti_joint_dof_begin, self._arti_joint_dof_begin + 6)
            attrib = self._get_cached_attribute("joint_qd", target, _slice=attrib_slice)
        else:
            return  # no-op

        self._set_attribute_values(attrib, values, mask=mask)

    def get_link_transforms(self, source: Model | State):
        return self._get_cached_attribute("body_q", source)

    def get_link_velocities(self, source: Model | State):
        return self._get_cached_attribute("body_qd", source)

    def get_dof_positions(self, source: Model | State):
        return self._get_cached_attribute("joint_q", source)

    def set_dof_positions(self, target: Model | State, values, mask=None):
        attrib = self._get_cached_attribute("joint_q", target)
        self._set_attribute_values(attrib, values, mask=mask)

    def get_dof_velocities(self, source: Model | State):
        return self._get_cached_attribute("joint_qd", source)

    def set_dof_velocities(self, target: Model | State, values, mask=None):
        attrib = self._get_cached_attribute("joint_qd", target)
        self._set_attribute_values(attrib, values, mask=mask)

    def get_dof_forces(self, source: Control):
        return self._get_cached_attribute("joint_f", source)

    def set_dof_forces(self, target: Control, values, mask=None):
        attrib = self._get_cached_attribute("joint_f", target)
        self._set_attribute_values(attrib, values, mask=mask)

    # ========================================================================================
    # Utilities

    def get_model_articulation_mask(self, mask=None):
        """
        Get Model articulation mask from a mask in this ArticulationView.

        Args:
            mask (array): Mask of articulations in this ArticulationView (all by default).
        """
        if mask is None:
            return self.articulation_mask
        else:
            if not isinstance(mask, wp.array):
                mask = wp.array(mask, dtype=bool, device=self.device, copy=False)
            assert mask.shape == (self.count,)
            articulation_mask = wp.zeros(self.model.articulation_count, dtype=bool, device=self.device)
            wp.launch(
                set_model_articulation_mask_kernel,
                dim=mask.size,
                inputs=[mask, self.articulation_indices, articulation_mask],
            )
            return articulation_mask

    def eval_fk(self, target: Model | State, mask=None):
        """
        Evaluates forward kinematics given the joint coordinates and updates the body information.

        Args:
            mask (array): Mask of articulations in this ArticulationView (all by default).
        """
        # translate view mask to Model articulation mask
        articulation_mask = self.get_model_articulation_mask(mask=mask)
        newton.sim.eval_fk(self.model, target.joint_q, target.joint_qd, target, mask=articulation_mask)
