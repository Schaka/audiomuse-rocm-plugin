# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Per-arch overrides, keyed by GPU arch.

Arches with no profile of their own run the defaults in ``base.ArchProfile``,
which is the intended state - a profile exists only where an arch was found to
need something different.
"""

import os
from typing import Dict, Optional

from .base import ArchProfile, ProviderSpec
from .gfx803 import Gfx803Profile
from .gfx1201 import Gfx1201Profile

__all__ = ["ArchProfile", "ProviderSpec", "profile_for", "apply_env"]

PROFILES = (Gfx803Profile, Gfx1201Profile)


def profile_for(arch: Optional[str]) -> ArchProfile:
    """Return the profile covering ``arch``, or the defaults if none does."""
    for profile in PROFILES:
        if arch in profile.arches:
            return profile()
    return ArchProfile()


def apply_env(profile: ArchProfile) -> Dict[str, str]:
    """Apply ``profile.env``, returning the keys it actually set.

    An already-set variable is left alone: whoever set it on the container is
    overriding the profile on purpose.
    """
    applied = {}
    for key, value in (profile.env or {}).items():
        if key in os.environ:
            continue
        os.environ[key] = str(value)
        applied[key] = str(value)
    return applied
