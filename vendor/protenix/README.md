This Apache-2.0 patch is relative to
[cytokineking/Protenix at b806d254](https://github.com/cytokineking/Protenix/tree/b806d2540ab326d71edcc47128343e627248c099).
It enables native PyTorch MPS inference on the V2 model and includes the fork's
expected-TM confidence export for the binder–target interface. The installer
applies it to a fresh copy of that revision in the install prefix. Model weights
and chemical dictionaries are separate runtime inputs.
