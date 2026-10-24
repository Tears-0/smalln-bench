"""smalln-bench: subsampling benchmark for small-N proteomic biomarker discovery.

The design is described in the paper (Sections 2-3) and its supplement. Module map:

    paths        repository locations
    reference    UniProt / Ensembl sequences and Reactome membership
    loaders      Task objects for B1, B2, B5 (paper §3.2)
    limma        moderated t (eBayes), validated against R limma
    subsample    seeded, balanced, unit-disjoint draws (paper §3.3)
    cv           fold plans and the internal / external AUC estimators
    preprocess   in-fold presence filter, imputation, scaling, filters
    models       CCP, EN, RF, TabPFN
    fusion       batched lockstep trainer for the frozen fusion arm (supplement S1)
    metrics      recovery, RBO, directional agreement, Kuncheva
    configs      the 15 configurations (paper §3.4, Table 2)
    run          grid execution, one parquet row per run
"""
