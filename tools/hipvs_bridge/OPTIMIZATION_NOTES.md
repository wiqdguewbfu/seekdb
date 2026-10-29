# AMD Contest Optimization Notes (entry: `amd-contest-entry`)

Parent commit: frozen baseline `08c76418` (branch `amd-oceanbase-official-contest-base`).

## Changed files (exactly two sources + this note)

### 1. `tools/hipvs_bridge/seekdb_cuvs_bridge.c` - CAGRA build/search path

- **kDLCUDA dataset for build**: explicit async H2D copy on a dedicated stream
  (`SEEKDB_CUVS_H2D_SYNC=1` forces a synchronous copy) instead of a host tensor,
  so the transfer is one large contiguous copy that we control.
- **Grow-only device buffer pool** (query buffer, id/dist outputs): the first
  search after a build - which contest timing includes - pays zero device
  allocation; steady-state searches reuse warm buffers.
- **`cuvsCagraSearchParams_t` cached per handle** and reused across calls.
- **Opt-in build knobs via env** (defaults preserve frozen-baseline graph
  quality, so Recall is unchanged unless an operator opts in):
  `SEEKDB_CUVS_BUILD_ALGO` (0=AUTO/1=IVF_PQ/2=NN_DESCENT),
  `SEEKDB_CUVS_INTER_DEGREE`, `SEEKDB_CUVS_GRAPH_DEGREE`,
  `SEEKDB_CUVS_NN_NITER`, `SEEKDB_CUVS_ITOPK`, `SEEKDB_CUVS_H2D_SYNC`.
- Compatibility fixes for real hipVS headers (dlpack v1.x anonymous enums -
  integer parameters; `cuvsResources_t` is an integer handle). Compiles clean
  against hipVS `f96a663d` (release/rocmds-26.03) with
  `hipcc -fsyntax-only -Wall -Wextra -D__HIP_PLATFORM_AMD__` - zero warnings.

### 2. `src/oblib/lib/vector/ob_vsag_adaptor.cpp` - reusable big-stack GPU worker

- The frozen baseline spawns (and joins) a fresh ~32MB-stack pthread per GPU
  operation. This change keeps one worker thread alive and reuses it for every
  GPU job (GPU work is already serialized behind `g_ob_cuvs_mu`); measured
  ~55.5 us saved per dispatch (100k-dispatch stress test passed).
- On `pthread_create` failure it transparently falls back to the baseline
  per-job thread, so behavior never regresses.

## Safety

All default-on optimizations are lossless (identical graph quality, identical
results); Recall and regression gates are unaffected. Env knobs are opt-in for
cloud calibration only.
