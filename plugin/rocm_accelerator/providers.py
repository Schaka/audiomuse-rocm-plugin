# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""ONNX Runtime execution provider names.

Spelled out once, because a typo here is not an error anywhere: onnxruntime
ignores an unknown provider name and silently runs the session on CPU.
"""

MIGRAPHX = "MIGraphXExecutionProvider"
ROCM = "ROCMExecutionProvider"
CPU = "CPUExecutionProvider"
