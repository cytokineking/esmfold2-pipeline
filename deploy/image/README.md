# Machine image build and qualification

`bootstrap-image.sh` creates a reproducible ESMFold2-pipeline installation on a
fresh Ubuntu 22.04 or 24.04 GPU machine. It installs a pinned pipeline revision,
the selected model checkpoints, Protenix v2 and its verified checkpoint, CUDA
build and inspection tools, `rclone`, `pigz`, `sqlite3`, `tmux`, and HMMER.

Set `ESMFOLD2_PIPELINE_REF` to an immutable commit before building a reusable
image. Other optional inputs are:

- `ESMFOLD2_IMAGE_CUDA_VERSION`: `12.8` (default) or `13.0`;
- `ESMFOLD2_PRELOAD_MODELS`: comma- or space-separated model aliases (default:
  `fast`);
- `ESMFOLD2_IMAGE_DISK_GB`: optional declared size of the provisioned image;
- `ESMFOLD2_IMAGE_MIN_ROOT_GB`: optional measured root-filesystem floor (use a
  lower value than the declared cloud size to account for GB/GiB and filesystem
  overhead); and
- `ESMFOLD2_IMAGE_CLEAN_USER_CACHES=1`: remove duplicate package/model caches
  from non-root home directories without deleting user projects.

The bootstrap writes a sanitized evidence set under `/etc/esmfold2-pipeline`:

- `esmfold2-pipeline-image.json`: source revisions, CUDA contract, selected
  models, disk information, utility list, and subordinate-manifest hashes;
- `esmfold2-pipeline-sbom.json`: exact Python package names and versions without
  source URLs or editable paths;
- `esmfold2-os-packages.txt`: exact installed Debian package inventory; and
- `esmfold2-models.sha256`: content hashes for every preloaded Hugging Face
  model blob.

Run `qualify-image.sh --full` before capturing a reusable image. Qualification
re-hashes the evidence, OS inventory, lockfile, bootstrap script, model blobs,
and Protenix checkpoint; checks every model recorded by the image manifest; and
performs a real ESMFold2 GPU smoke. Full qualification additionally launches
one template-enabled Protenix v2 validation for each supported modality:
miniprotein, VHH, and scFv. It verifies the compiled Protenix extension's CUDA
runtime linkage and native architecture targets.

Optional qualification inputs are:

- `ESMFOLD2_QUALIFICATION_OUTPUT_ROOT`: local evidence/output directory;
- `ESMFOLD2_QUALIFICATION_MIN_GPU_MEMORY_MIB`: enforce a site-selected minimum
  memory size for every visible GPU; and
- `ESMFOLD2_QUALIFICATION_REMOTE`: an rclone base path for a temporary,
  automatically removed object-storage round trip.

The result is written to `qualification.json` in the output directory and to
`/etc/esmfold2-pipeline/esmfold2-pipeline-qualification.json`. Promote an image
only after the required qualification mode reports `passed: true`. Storage
credentials are runtime inputs and must not be persisted in captured images.

CUDA 12.8 uses the `cu128` PyTorch backend and cuEquivariance `cu12` operator.
CUDA 13.0 uses `cu130` and `cu13`. Bootstrap removes the opposite development
profile and qualification rejects mixed compiler, PyTorch, and accelerator
backends.
