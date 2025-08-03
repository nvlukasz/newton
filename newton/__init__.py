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

from . import render
from . import selection
from . import solvers
from ._version import __version__

# Core functionality
from ._src.core import (
    Axis,
    AxisType,
)

# Geometry functionality
from ._src.geometry import (
    GEO_BOX,
    GEO_CAPSULE,
    GEO_CONE,
    GEO_CYLINDER,
    GEO_MESH,
    GEO_NONE,
    GEO_PLANE,
    GEO_SDF,
    GEO_SPHERE,
    SDF,
    Mesh,
    # create_box,
    # create_capsule,
    # create_cone,
    # create_cylinder,
    # create_none,
    # create_plane,
    # create_sphere,
)

# Simulation functionality
from ._src.sim import (
    EQ_CONNECT,
    EQ_JOINT,
    EQ_WELD,
    JOINT_BALL,
    JOINT_D6,
    JOINT_DISTANCE,
    JOINT_FIXED,
    JOINT_FREE,
    JOINT_MODE_NONE,
    JOINT_MODE_TARGET_POSITION,
    JOINT_MODE_TARGET_VELOCITY,
    JOINT_PRISMATIC,
    JOINT_REVOLUTE,
    Contacts,
    Control,
    Model,
    ModelBuilder,
    State,
    eval_fk,
    eval_ik,
)

# TODO: move as methods of ModelBuilder?
from ._src.utils.import_mjcf import parse_mjcf as load_mjcf
from ._src.utils.import_urdf import parse_urdf as load_urdf
from ._src.utils.import_usd import parse_usd as load_usd


# __all__ = [
#     "EQ_CONNECT",
#     "EQ_JOINT",
#     "EQ_WELD",
#     "GEO_BOX",
#     "GEO_CAPSULE",
#     "GEO_CONE",
#     "GEO_CYLINDER",
#     "GEO_MESH",
#     "GEO_NONE",
#     "GEO_PLANE",
#     "GEO_SDF",
#     "GEO_SPHERE",
#     "JOINT_BALL",
#     "JOINT_D6",
#     "JOINT_DISTANCE",
#     "JOINT_FIXED",
#     "JOINT_FREE",
#     "JOINT_MODE_NONE",
#     "JOINT_MODE_TARGET_POSITION",
#     "JOINT_MODE_TARGET_VELOCITY",
#     "JOINT_PRISMATIC",
#     "JOINT_REVOLUTE",
#     "SDF",
#     "Axis",
#     "AxisType",
#     "Contacts",
#     "Control",
#     "Mesh",
#     "Model",
#     "ModelBuilder",
#     "State",
#     "__version__",
#     # "create_box",
#     # "create_capsule",
#     # "create_cone",
#     # "create_cylinder",
#     # "create_none",
#     # "create_plane",
#     # "create_sphere",
#     "eval_fk",
#     "eval_ik",
#     "solvers",
# ]
