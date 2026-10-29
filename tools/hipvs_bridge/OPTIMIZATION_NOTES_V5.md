# AMD Contest V5 Optimization Notes (entry: `amd-contest-entry-v5`)

Parent commit: frozen V5 baseline `634888d9` (branch
`amd-oceanbase-official-contest-base-v5`, tag `amd-cagra-contest-baseline-v5`).

## Scope

Exactly one source file changed: `tools/hipvs_bridge/seekdb_cuvs_bridge.c`.
The adaptor (`ob_vsag_adaptor.cpp`), fixed SQL, test inputs, build scripts and
the external hipVS library are untouched. V5's `seekdb_cuvs_serialize` /
`seekdb_cuvs_deserialize` (persistence + reload gates) are preserved verbatim.

## Changes (all lossless at default env)

1. **Dataset H2D staging in `seekdb_cuvs_build`**: the bridge itself stages the
   n x dim f32 dataset on the device (one large contiguous copy on a dedicated
   bridge stream; `SEEKDB_CUVS_H2D_SYNC=1` forces a synchronous copy) and hands
   `cuvsCagraBuild` a `kDLCUDA` tensor instead of `kDLCPU`. On failure it falls
   back to the baseline CPU-tensor path.
2. **Grow-only device scratch pool** for the query buffer and neighbor-id /
   distance outputs: the first search after a build (included in contest build
   timing) pays no device allocation; steady-state searches reuse warm buffers
   instead of alloc/free per call. Pool buffers use context-global
   cudaMalloc/cudaFree so they never depend on a cuvsResources_t lifetime.
3. **`cuvsCagraSearchParams_t` cached per handle**, created once and re-tuned
   per call exactly like the baseline (`itopk_size = max(4*topk, 512)`,
   `search_width = 4`, plus `max_queries >= nq` batching alignment).

## Defaults preserve the frozen V5 graph quality

Build parameters default to the V5-pinned values (`NN_DESCENT`,
`graph_degree=128`, `intermediate_graph_degree=256`, `nn_descent_niter=100`);
env overrides exist only for cloud calibration:

- `SEEKDB_CUVS_BUILD_ALGO` (default 2 = NN_DESCENT)
- `SEEKDB_CUVS_GRAPH_DEGREE` (default 128)
- `SEEKDB_CUVS_INTER_DEGREE` (default 256)
- `SEEKDB_CUVS_NN_NITER` (default 100)
- `SEEKDB_CUVS_ITOPK` (floor above baseline sizing; default 0 = baseline)
- `SEEKDB_CUVS_H2D_SYNC` (default 0 = async copy + stream sync)

Recall, result correctness, persistence/reload and resource gates are
unaffected: no algorithmic parameter changes unless an operator opts in, and
all transfers remain fully synchronized before the timed call returns.

## Compatibility notes

dlpack v1.x defines `DLDeviceType` / `DLDataTypeCode` as anonymous typedef
enums, so device/dtype fields are set through plain-int parameters with casts,
and `cuvsResources_t` is treated as an integer handle. This form was validated
against real hipVS headers (`f96a663d`, release/rocmds-26.03) with
`hipcc -fsyntax-only -Wall -Wextra -D__HIP_PLATFORM_AMD__` — zero warnings.

## Not done in this entry (by design)

The adaptor-level reusable big-stack GPU worker (replacing per-op 32MiB-stack
pthread spawn/join) is intentionally NOT included: V5's adaptor has multiple
job sites (build/search/snapshot/batch) and that refactor changes drop/rebuild
ordering surface; it should land as a separate commit only after profiling on
the official GPU environment shows the spawn/join cost is significant.
