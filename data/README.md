# Local data policy

Raw datasets, processed media, and feature caches are local artifacts and must not be committed.

```text
data/
├── raw/          ignored source datasets
├── processed/    ignored normalized/converted data
├── cache/        ignored model-specific features or tokens
├── manifests/    tracked resolved example metadata and content hashes
└── splits/       tracked resolved partition and subset-mask manifests
```

Dataset-specific acquisition, parsing, quality control, grouping, and collation are owned by integrations and documented by the experiments that use them. Framework-core data utilities operate only on the canonical records and grouping metadata produced by those integrations.

Every materialized manifest must use stable example IDs and record enough source information and hashes to reproduce the exact selection without committing raw data.
