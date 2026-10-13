"""smalln-bench: subsampling benchmark for small-N proteomic biomarker discovery.

The design is fixed in specification.md (§5-§6, amendments in §16). Module map:

    paths        repository locations
    reference    UniProt / Ensembl sequences and Reactome membership
    loaders      Task objects for B1, B2, B5 (spec §5.0, §16 A3)
    limma        moderated t (eBayes), validated against R limma
    subsample    seeded, balanced, unit-disjoint draws (spec §5.2)
    cv           fold plans and the internal / external AUC estimators
    preprocess   in-fold presence filter, imputation, scaling, filters
    models       CCP, EN, RF, TabPFN
    fusion       batched lockstep trainer for the frozen fusion arm (F27)
    metrics      recovery, RBO, directional agreement, Kuncheva
    configs      the 15 configurations (spec §5.1, §16 A3-A4)
    run          grid execution, one parquet row per run
"""
