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

# TODO: move these to Warp?
from ._src.core.spatial import (
    quat_between_axes,
    quat_decompose,
    quat_from_euler,
    quat_to_euler,
    quat_to_rpy,
    quat_twist,
    quat_twist_angle,
    transform_twist,
    transform_wrench,
    velocity_at_point,
)

# sim utils
from ._src.sim import (
    color_graph,
    plot_graph,
)

# TODO: move importers to ModelBuilder, e.g., parse_mjcf() -> ModelBuilder.load_mjcf()
# TODO: move math utils to Warp?
from ._src.utils import (
    boltzmann,
    download_asset,
    leaky_max,
    leaky_min,
    parse_mjcf,
    parse_urdf,
    parse_usd,
    smooth_max,
    smooth_min,
    transform_inertia,
    vec_abs,
    vec_leaky_max,
    vec_leaky_min,
    vec_max,
    vec_min,
)

# recorders
from ._src.utils.recorder import BasicRecorder, ModelAndStateRecorder

# fmt: off
__all__ = [  # noqa
    # spatial math utils
    "quat_between_axes",
    "quat_decompose",
    "quat_from_euler",
    "quat_to_euler",
    "quat_to_rpy",
    "quat_twist",
    "quat_twist_angle",
    "transform_twist",
    "transform_wrench",
    "velocity_at_point",

    # misc math utils
    "boltzmann",
    "leaky_max",
    "leaky_min",
    "smooth_max",
    "smooth_min",
    "transform_inertia",
    "vec_abs",
    "vec_leaky_max",
    "vec_leaky_min",
    "vec_max",
    "vec_min",

    # sim utils
    "color_graph",
    "plot_graph",

    # assets
    "download_asset",
    "parse_mjcf",
    "parse_urdf",
    "parse_usd",

    # recorders
    "BasicRecorder",
    "ModelAndStateRecorder",
]
# fmt: on
