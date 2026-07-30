# Adding an arch profile

Most AMD GPUs need nothing special: the plugin offers MIGraphX for musicnn and
CLAP, with fp16 on, and that is it. An **arch profile** exists only where one
generation was found to need something different.

Arches without a profile use the defaults — that is the normal case, not a gap
to fill.

## The seams

Subclass `ArchProfile` (`plugin/rocm_accelerator/arch/base.py`) and override
only what differs.

| Member | Default | Use it for |
| --- | --- | --- |
| `arches` | `frozenset()` | the arch strings (as `rocminfo` reports them) this profile covers |
| `env` | `{}` | environment variables the arch needs, applied before onnxruntime or CTranslate2 are imported. Assign a plain dict; the inherited default is read-only so no profile can extend it into every other one. A variable already set on the container is left alone |
| `fp16_supported` | `True` | set `False` where fp16 buys no throughput; the plugin's `fp16_enable` setting is then ignored with a warning |
| `supports_model_cache_dir` | `True` | set `False` on EP builds without the `migraphx_model_cache_dir` option |
| `migraphx_options()` | `{}` | extra MIGraphX EP options, merged over the generic ones |
| `migraphx_models(providers)` | `("musicnn", "clap")` | narrow which session labels MIGraphX is offered for |
| `extra_providers(providers)` | `()` | additional providers (`ProviderSpec`) for labels MIGraphX cannot serve |

`providers` is the list of execution providers this image's onnxruntime actually
has, so a profile can react to the build instead of hardcoding assumptions about
it.

## Example

Suppose fp16 turns out to be a loss on Vega and the arch needs an environment
variable set. `plugin/rocm_accelerator/arch/gfx9.py`:

```python
from .base import ArchProfile


class Gfx9Profile(ArchProfile):
    arches = frozenset({"gfx900", "gfx906"})

    # <the measurement that showed this, in one line>
    fp16_supported = False

    env = {"MIOPEN_DEBUG_CONV_IMPLICIT_GEMM": "0"}
```

Then list it in `arch/__init__.py`:

```python
from .gfx9 import Gfx9Profile

PROFILES = (Gfx803Profile, Gfx9Profile)
```

That is the whole wiring. `register()` needs no change.

## Routing a model to a different provider

When MIGraphX cannot compile one model on an arch, take it out of
`migraphx_models` and name it in a `ProviderSpec` instead. Two providers must
never end up in one session — see
[ARCH_NOTES.md](ARCH_NOTES.md#migraphx-and-the-rocm-ep-must-never-share-a-session)
— which is what removing the label from `migraphx_models` guarantees.

Provider names come from `plugin/rocm_accelerator/providers.py` rather than being
spelled out — onnxruntime ignores a name it does not know and runs on CPU, so a
typo produces no error, only a slow analysis.

```python
from ..providers import ROCM
from .base import ArchProfile, ProviderSpec


def migraphx_models(self, providers):
    if ROCM in providers:
        return ("musicnn",)
    return super().migraphx_models(providers)

def extra_providers(self, providers):
    if ROCM not in providers:
        return ()
    return (ProviderSpec(ROCM, {"device_id": 0}, only_models=("clap",)),)
```

If the provider is only safe with one of ORT's graph optimizers turned off for
that session (gfx803's ROCM EP + CLAP needs `ConvActivationFusion` disabled —
see [ARCH_NOTES.md](ARCH_NOTES.md#clap-on-the-rocm-ep-needs-convactivationfusion-disabled)
for why), name it in `disable_optimizers` instead of adding a seam in core:

```python
return (ProviderSpec(
    ROCM, {"device_id": 0}, only_models=("clap",),
    disable_optimizers=("ConvActivationFusion",),
),)
```

`register()` applies this by wrapping `onnxruntime.InferenceSession` for the
lifetime of the worker process (`ort_fusion_guard.py`) rather than a core seam,
because onnxruntime has no environment variable for
`optimization.disable_specified_optimizers` and core builds its own
`SessionOptions`. The wrap is scoped by provider name, so it only touches
sessions that actually use the named provider.

Valid labels are core's: `musicnn`, `clap`, `clap_text`, `whisper_encoder`,
`whisper_decoder`, `gte`, `silero_vad`. An unknown one matches nothing and is
warned about.

## Conventions

- **Record the evidence, not the theory.** A one-line comment saying what was
  measured, with the detail in [ARCH_NOTES.md](ARCH_NOTES.md).
- **A profile only declares.** It decides *what* to ask for; `register()` owns
  how registration happens. A profile that has to know about registration order,
  the cache layout or core's internals means a seam is missing — add the seam.
- **Do not add an empty profile** for an arch you have not measured. The
  defaults already cover it.

## Testing without the hardware

`register()` takes every decision from two inputs, both trivially faked: the
arch string from `gpu.detect_arch()` and the provider list from
`gpu.available_providers()`. Stub both and pass a recording object as `ctx` to
see exactly what a profile would register.
