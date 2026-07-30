# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Where MIGraphX keeps its compiled programs, and the options that point at it.

Two shapes, because not every EP build has the same options - see
``arch.base.ArchProfile.supports_model_cache_dir``.
"""

import os
from typing import Dict

CACHE_ROOT = "/app/.cache/migraphx"


def cache_dir(fp16: bool) -> str:
    """The cache directory for one precision, created if missing.

    Split by precision because MIGraphX keys its artifacts on version, graph,
    arch and shapes but *not* precision: one shared directory would serve an
    fp32 artifact as a hit after fp16 is switched on. Created eagerly because
    the EP's save path does no error handling and raises out of session
    creation on a missing directory.
    """
    path = os.path.join(CACHE_ROOT, "fp16" if fp16 else "fp32")
    os.makedirs(path, exist_ok=True)
    return path


def cache_dir_options(fp16: bool) -> Dict[str, str]:
    """Options letting the EP key the cache directory by itself."""
    return {"migraphx_model_cache_dir": cache_dir(fp16)}


def per_model_options(model_label: str, fp16: bool) -> Dict[str, str]:
    """Options for EP builds that cache one explicit file instead of a directory.

    The save and load options are mutually exclusive per session and take a
    literal path, so the keying migraphx_model_cache_dir would do internally is
    done here: one file per model and precision, its existence deciding save vs
    load. Nothing validates that the file still matches the graph, so a stale
    one loads wrong - delete it to force a recompile.
    """
    path = os.path.join(cache_dir(fp16), f"{model_label}.mxr")
    if os.path.exists(path):
        return {
            "migraphx_load_compiled_model": "True",
            "migraphx_load_model_path": path,
        }
    return {
        "migraphx_save_compiled_model": "True",
        "migraphx_save_model_path": path,
    }
