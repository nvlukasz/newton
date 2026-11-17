# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os

if os.environ.get("NEWTON_STEEL_FORCE_SCENE") == "1":
    from ._scene_parity import install

    install()
