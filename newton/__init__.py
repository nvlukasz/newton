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

# submodule APIs
from . import geometry, ik, selection, solvers, utils, viewer

# Core functionality
from ._src.core import (
    Axis,
    AxisType,
)

# Geometry functionality
from ._src.geometry import (
    SDF,
    GeoType,
    Mesh,
    ParticleFlags,
    ShapeFlags,
)

# Simulation functionality
from ._src.sim import (
    Contacts,
    Control,
    EqType,
    JointMode,
    JointType,
    Model,
    ModelBuilder,
    State,
    eval_fk,
    eval_ik,
)

# TODO: eliminate these helpers, roll functionality into Model and ModelBuilder?
from ._src.sim.style3d import Style3DModel, Style3DModelBuilder

# version
from ._version import __version__

# fmt: off
__all__ = [  # noqa
    "__version__",
    # core
    "Axis",
    "AxisType",

    # geometry
    "GeoType",
    "Mesh",
    "ParticleFlags",
    "SDF",
    "ShapeFlags",

    # sim
    "Contacts",
    "Control",
    "EqType",
    "JointMode",
    "JointType",
    "Model",
    "ModelBuilder",
    "State",
    "eval_fk",
    "eval_ik",

    # style3d helpers
    "Style3DModel",
    "Style3DModelBuilder",

    # submodules
    "geometry",
    "ik",
    "selection",
    "solvers",
    "utils",
    "viewer",
]
# fmt: on
