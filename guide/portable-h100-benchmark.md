# H100 qualification of the portable ESMFold2 design path

The source-only `portable` path was tested on a Hyperstack NVIDIA H100 PCIe
80 GB VM created from the production image
`ariax-esmfold2-pipeline-0861dbb-cu128-20260922a`. The installed pipeline was
at production commit `0861dbbd8fda027c845a9b3f84085ab176941a3c` with
PyTorch 2.11 CUDA 12.8. The candidate was overlaid through `PYTHONPATH`; the
installed image, checkpoint cache, and Protenix environment were not changed.

The benchmark used one VHH against human CD274/PD-L1 chain A from [PDB
4ZQK](https://www.rcsb.org/structure/4ZQK), with hotspots at the deposited
PD-1 contact surface. Its input and configs are
`example_input_structures/4zqk_pd1_pdl1.cif` and
`example_configs/4zqk_cd274_pd1_interface_vhh_{h100,150step_h100}.yaml`.
Each timing is a fresh worker process with the same config and seed. The
20-step config SHA-256 is
`8f425e93bf7d48831291d10074d73ff4248eb7b2f3ad4a42fc850e737ee7676e`;
the 150-step config SHA-256 is
`6076b41e98cc5dab54178dee8cb8b5c8be6003002d28fc3f0271729189d11e1a`.

| Run | Source and mode | Wall time | Peak GPU memory | Gain versus candidate off |
| --- | --- | ---: | ---: | ---: |
| 20 steps | candidate source, off | 45.22 s | 56,399 MiB | reference |
| 20 steps | candidate source, portable | 43.70 s | 56,725 MiB | 3.4% |
| 150 steps | candidate source, off | 163.27 s | 56,399 MiB | reference |
| 150 steps | candidate source, portable | 154.31 s | 56,727 MiB | 5.5% |

For the 150-step runs, the timestamped logs put the design phase at 145.25 s
off and 136.87 s portable, a 5.8% reduction. The peak-memory figures came
from one-second `nvidia-smi` sampling and are approximate. A first, cold
installed-source 20-step run took 61.80 s; it is excluded from the gain
calculation because checkpoint and runtime warm-up distort the comparison.

The portable 150-step log reported 130 unused confidence computations and
one-step structure samples deferred, plus one target pair-bias preparation
reused 149 times and one PLM-constant preparation reused 149 times. The late
confidence folds and critic still ran normally. A separate experiment using
Anthropic's ESMC PLM CUDA graph passed its replay check but took 43.93 s for
20 steps and peaked at 74,053 MiB. That graph is deliberately not included
in the patch because it provided no throughput gain and cost about 17 GiB.
A synchronized 20-step profile measured the early-step mean at 0.614 s fold
forward, 0.402 s structure backward, and 0.148 s combined PLM forward and
backward. Further substantial gains would need to address the folding model's
GPU trunk, which is a larger kernel and compatibility project.

The 150-step design produced a pose overlapping the crystallographic PD-1
interface: after aligning PD-L1 to 4ZQK, the predicted VHH contacted six of
the eight selected hotspots within 5 Å. This is only a geometric screen.
ESMFold2 Fast reported binder–target ipTM 0.856 and binder pLDDT 91.15.
Independent Protenix v2 validation reported scoped ipTM 0.669 and ipSAE
0.288, but the binder Cα RMSD after target alignment was 4.70 Å, above the
pipeline's 2.5 Å consensus cutoff. The candidate was therefore not ranked as
a validated design, and there is no experimental evidence of PD-1 blockade.

Focused H100 tests passed for the portable adapter and CUDA PLM loss,
gradient, masked inputs, and RNG state. The broad H100 rerun passed 404
tests. It excluded two image-environment checks: `test_image_contract`
cannot import because this runtime lacks `pytest`, and an existing CLI test
assumes that Protenix `accelerate` is absent even though this image supplies
it. Stock ESMFold2 does not
repeat bitwise in this image, so exact full-fold equality has not been
claimed. On identical prepared inputs, the maximum distogram difference
between two stock calls was 2.25, versus 2.27 for stock versus portable;
the CUDA RNG state agreed in both comparisons.
`scripts/inspect_portable_fold_variability.py` prints the complete diagnostic.

The result supports opt-in use on this H100 image for longer designs, with a
modest measured gain. A100 and Blackwell have not been qualified. The switch
remains off by default, and no image or worker deployment was changed.
