# Environments

Two environments, both created 2026-10-02.

## Python — benchmark

A venv **outside OneDrive** (so the sync client never touches it), layered on
the Miniconda base interpreter to reuse its CUDA build of torch:

```
python -m venv --system-site-packages C:\Users\X\.venvs\smalln
C:\Users\X\.venvs\smalln\Scripts\python.exe -m pip install tabpfn pyarrow cptac
```

Pinned versions are in `requirements-pinned.txt`. Two cautions:

- Both `fair-esm` 2.0.0 (ESM-2, used here) and `esm` 3.2.1.post1
  (EvolutionaryScale ESM3) are installed in the base interpreter, and both
  provide a module named `esm`. `import esm` currently resolves to fair-esm
  2.0.0. Embedding code must assert this before computing anything.
- TabPFN 9.0.0: the licence was accepted by the owner and the pinned checkpoint
  is local (`models/tabpfn/`, specification §16 A1). Always pass
  `model_path=` — without it the package tries to download the default
  checkpoint and prompts for an API key.

## R — limma reference

```
conda create -n smalln-r -c conda-forge r-base=4.4 r-biocmanager r-jsonlite
conda run -n smalln-r Rscript -e "BiocManager::install('limma')"
```

R 4.4.3 (2025-02-28 ucrt), Bioconductor 3.20, limma 3.62.2, BiocManager
1.30.27. Used only to validate the Python moderated-t implementation on a
fixture; the benchmark itself does not call R.
