This branch = competition/amd-cagra@7067ad64 (platform-validated tuning) + one
commit that adds only default-neutral changes to the bridge:

1. cuvsCagraSearchParams_t cached per handle (create once, re-tune fields per
   call with identical values) instead of create/destroy per single-row SQL.
2. Optional dataset H2D staging before cuvsCagraBuild behind
   SEEKDB_CUVS_H2D_STAGE=1 (default 0 = untouched, platform-validated build
   path; per-handle buffer freed with the handle; falls back to kDLCPU on any
   failure).
3. Optional workspace alignment behind SEEKDB_CUVS_MAXQ=1 (default 0 = off).

All build defaults (per-dimension GD/IGD/NITER/ALGO, itopk max(2k,256),
width=4, param file, timing logs) are byte-identical to the parent commit.
