# models/

Model weights are local only and never committed: they are large, and their
licences do not permit redistribution.

| File | Notes | MD5 |
|---|---|---|
| `tabpfn/tabpfn-v3.5-fast-20260909.safetensors` | TabPFN v3.5 "fast" checkpoint, downloaded by the owner from Prior Labs after licence acceptance (2026-10-03). 334,181,708 bytes; 307 float32 tensors, 83.5 M parameters; architecture `tabpfn_v3_5`. Load with `TabPFNClassifier(model_path=...)`; no API key needed once local. | 9340b9f6927ab257228158cbe446f01f |

ESM-2 (`esm2_t12_35M_UR50D`) is not stored here: `fair-esm` downloads it to the
torch hub cache (`~/.cache/torch/hub/checkpoints/`) on first use.
