# Documentation

[← Back to project README](../README.md)

Reference and how-to docs for the ESMFold2 Pipeline. Start with the
[project README](../README.md) for an overview and quickstart; come here for the
field-level detail.

| Guide | What's inside |
| --- | --- |
| [Installation & requirements](installation.md) | Hardware, disk, the installer, advanced flags, model preload, Protenix runtime. |
| [YAML configuration reference](configuration.md) | Every `target` / `binder` / `campaign` / `loss` field, frameworks, recommended defaults. |
| [CLI reference](cli-reference.md) | All commands and flags, multi-GPU execution, resume recovery, dev checks. |
| [Structure targets & hotspots](structure-targets.md) | Target input modes, hotspots, structure indexing, distogram conditioning. |
| [Optional Protenix validation](validation.md) | Launch-integrated validation plus the lower-level validation lifecycle. |
| [Runtime & scaling](runtime-and-scaling.md) | Preprint campaign scale, compute estimates, recommended campaign progression. |
| [Output layout](outputs.md) | The full campaign directory tree and per-file notes. |

## Framework panels

- [Bundled scFv frameworks](../src/esmfold2_pipeline/frameworks/scfv/README.md)
- [Bundled VHH frameworks](../src/esmfold2_pipeline/frameworks/vhh/README.md)

## Example configs

Ready-to-run YAML lives under [`example_configs/`](../example_configs/)
(miniprotein and scFv variants).
