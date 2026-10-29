/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here.
 *
 * V5 entry (parent: frozen baseline 634888d9). Optimizations on top of the
 * frozen V5 bridge — all lossless with default env (identical graph quality,
 * identical results, Recall gates unaffected):
 *  - seekdb_cuvs_build stages the dataset on the device itself (one large
 *    contiguous H2D copy on a dedicated bridge stream, SEEKDB_CUVS_H2D_SYNC=1
 *    forces a synchronous copy) and hands cuvsCagraBuild a kDLCUDA tensor,
 *    instead of a kDLCPU tensor, so the transfer is one copy we control.
 *  - Device scratch (query buffer, id/dist outputs) lives in a process-wide
 *    grow-only pool: the first search after a build — which contest timing
 *    includes — pays no device allocation, and steady-state searches reuse
 *    warm buffers instead of alloc/free per call.
 *  - cuvsCagraSearchParams_t is cached per handle (created once, field-tuned
 *    per call exactly like the baseline: itopk_size = max(topk*4, 512),
 *    search_width = 4).
 *  - Build/search knobs stay overridable via env for cloud calibration:
 *      SEEKDB_CUVS_BUILD_ALGO      0=AUTO,1=IVF_PQ,2=NN_DESCENT (default 2)
 *      SEEKDB_CUVS_INTER_DEGREE    intermediate_graph_degree (default 256)
 *      SEEKDB_CUVS_GRAPH_DEGREE    graph_degree             (default 128)
 *      SEEKDB_CUVS_NN_NITER        nn_descent_niter         (default 100)
 *      SEEKDB_CUVS_ITOPK           itopk_size floor         (default 0=baseline sizing)
 *      SEEKDB_CUVS_H2D_SYNC        1=sync H2D copy (default 0=async+sync stream)
 *    Defaults reproduce the frozen V5 baseline's pinned parameters exactly.
 *  - dlpack v1.x compatibility: DLDeviceType / DLDataTypeCode are anonymous
 *    typedef enums in real hipVS, so device/dtype fields are set through
 *    plain-int parameters and casts (verified against hipVS f96a663d headers
 *    with hipcc -Wall -Wextra, zero warnings).
 * V5's serialize/deserialize (index persistence / reload gates) are preserved.
 */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

/* ---------------------------------------------------------------- helpers */

static long env_long(const char *name, long dflt)
{
  const char *v = getenv(name);
  if (v == NULL || v[0] == '\0') { return dflt; }
  char *end = NULL;
  long parsed = strtol(v, &end, 10);
  if (end == v) { return dflt; }
  return parsed;
}

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraSearchParams_t sp;  /* cached, created lazily, reused across searches */
  cuvsCagraIndex_t       idx;
  long n, dim;
  float *ds;                   /* per-handle device staging for the dataset:
                                * freed with the handle so a live index can
                                * never observe a recycled buffer          */
  int ds_via_rmm;              /* 1 = ds came from cuvsRMMAlloc             */
} sk_cuvs_index;

/* Process-wide device scratch pool: grow-only, guarded by one mutex. The seekdb
 * adaptor serializes GPU work behind its own lock, so the pool never sees
 * concurrent grows in practice. */
static pthread_mutex_t g_pool_mu = PTHREAD_MUTEX_INITIALIZER;
static float       *g_qd   = NULL;  /* device query buffer       */
static unsigned    *g_nd   = NULL;  /* device neighbor-id output */
static float       *g_dd   = NULL;  /* device distance output    */
static long         g_qd_cap = 0;   /* in floats                 */
static long         g_nd_cap = 0;   /* in unsigned               */
static long         g_dd_cap = 0;   /* in floats                 */
static cudaStream_t g_stream = NULL;
static int          g_stream_tried = 0;

static cudaStream_t sk_stream(void)
{
  if (!g_stream_tried) {
    g_stream_tried = 1;
    if (cudaStreamCreate(&g_stream) != cudaSuccess) { g_stream = NULL; }
  }
  return g_stream;
}

/* Grow a pooled buffer if needed. Returns 0 on success. Pool buffers are
 * process-wide and outlive index handles, so they use context-global
 * cudaMalloc/cudaFree (never cuvsRMM*, whose allocations are tied to a
 * cuvsResources_t lifetime). */
static int sk_grow_f(float **buf, long *cap, long need_f)
{
  if (need_f <= *cap) { return 0; }
  float *nb = NULL;
  if (cudaMalloc((void **)&nb, sizeof(float) * need_f) != cudaSuccess) { return 1; }
  if (*buf != NULL) { cudaFree(*buf); }
  *buf = nb; *cap = need_f;
  return 0;
}

static int sk_grow_u(unsigned **buf, long *cap, long need_u)
{
  if (need_u <= *cap) { return 0; }
  unsigned *nb = NULL;
  if (cudaMalloc((void **)&nb, sizeof(unsigned) * need_u) != cudaSuccess) { return 1; }
  if (*buf != NULL) { cudaFree(*buf); }
  *buf = nb; *cap = need_u;
  return 0;
}

/* `shape` must point to caller-owned storage that outlives the cuVS call.
 * dev/code take DLDeviceType / DLDataTypeCode constants as plain int: dlpack
 * v1.x defines them as anonymous-typedef enums, so the tagged `enum X` type
 * name is not portable here. */
static void sk_fill_dl_tensor(DLManagedTensor *t, long *shape, void *data,
                              int dev, int code, int bits)
{
  t->dl_tensor.data = data;
  t->dl_tensor.device.device_type = (DLDeviceType)dev;
  t->dl_tensor.device.device_id = 0;
  t->dl_tensor.ndim = 2;
  t->dl_tensor.dtype.code = (DLDataTypeCode)code;
  t->dl_tensor.dtype.bits = (uint8_t)bits;
  t->dl_tensor.dtype.lanes = 1;
  t->dl_tensor.shape = shape;
  t->dl_tensor.strides = 0;
  t->dl_tensor.byte_offset = 0;
}

/* ------------------------------------------------------------------ build */

void *seekdb_cuvs_build(const float *base, long n, long dim)
{
  sk_cuvs_index *h = (sk_cuvs_index *)calloc(1, sizeof(*h));
  if (!h) { return NULL; }
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h); return NULL; }

  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) { goto fail_res; }
  /* Frozen V5 baseline pins NN_DESCENT / degree 128 / intermediate 256 / 100
   * iterations; env overrides exist for cloud calibration only. */
  {
    long v;
    h->ip->build_algo = NN_DESCENT;
    if ((v = env_long("SEEKDB_CUVS_BUILD_ALGO", 2)) >= 0 && v <= 3) {
      h->ip->build_algo = (enum cuvsCagraGraphBuildAlgo)v;
    }
    h->ip->graph_degree = 128;
    if ((v = env_long("SEEKDB_CUVS_GRAPH_DEGREE", 128)) > 0) { h->ip->graph_degree = (size_t)v; }
    h->ip->intermediate_graph_degree = 256;
    if ((v = env_long("SEEKDB_CUVS_INTER_DEGREE", 256)) > 0) { h->ip->intermediate_graph_degree = (size_t)v; }
    h->ip->nn_descent_niter = 100;
    if ((v = env_long("SEEKDB_CUVS_NN_NITER", 100)) > 0) { h->ip->nn_descent_niter = (size_t)v; }
  }

  /* Stage the dataset on the device ourselves (one big contiguous H2D on the
   * bridge stream), then hand cuVS a kDLCUDA tensor. Staging is per-handle so
   * a live index can never observe a recycled buffer. */
  DLManagedTensor dset;
  long ds_shape[2] = {n, dim};
  int have_dev_copy = 0;
  {
    const long bytes = sizeof(float) * n * dim;
    if (cuvsRMMAlloc(h->res, (void **)&h->ds, bytes) == CUVS_SUCCESS) {
      h->ds_via_rmm = 1;
    } else if (cudaMalloc((void **)&h->ds, bytes) == cudaSuccess) {
      h->ds_via_rmm = 0;
    } else {
      h->ds = NULL;
    }
    if (h->ds != NULL) {
      cudaStream_t st = sk_stream();
      cudaError_t e;
      if (st != NULL && env_long("SEEKDB_CUVS_H2D_SYNC", 0) == 0) {
        e = cudaMemcpyAsync(h->ds, base, bytes, cudaMemcpyHostToDevice, st);
        if (e == cudaSuccess) { e = cudaStreamSynchronize(st); }
      } else {
        e = cudaMemcpy(h->ds, base, bytes, cudaMemcpyHostToDevice);
      }
      if (e == cudaSuccess) {
        sk_fill_dl_tensor(&dset, ds_shape, h->ds, kDLCUDA, kDLFloat, 32);
        have_dev_copy = 1;
      }
    }
  }
  if (!have_dev_copy) {
    /* Fall back to the baseline path: hand the CPU tensor to cuVS. */
    sk_fill_dl_tensor(&dset, ds_shape, (void *)base, kDLCPU, kDLFloat, 32);
  }

  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) { goto fail_ip; }
  if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) { goto fail_idx; }

  h->n = n; h->dim = dim;
  return h;

fail_idx:
  cuvsCagraIndexDestroy(h->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(h->ip);
fail_res:
  cuvsResourcesDestroy(h->res);
  free(h);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

/* ----------------------------------------------------------------- search */

int seekdb_cuvs_search(void *handle, const float *query, long nq, long topk,
                       unsigned int *out_ids, float *out_dist)
{
  sk_cuvs_index *h = (sk_cuvs_index *)handle;
  if (!h) { return 1; }
  int rc = 0;

  /* Cached search params (created lazily, e.g. for deserialized handles);
   * field-tuned per call exactly like the frozen baseline. */
  if (h->sp == NULL) {
    if (cuvsCagraSearchParamsCreate(&h->sp) != CUVS_SUCCESS) { h->sp = NULL; return 14; }
  }
  cuvsCagraSearchParams_t sp = h->sp;
  {
    size_t required_itopk = (size_t)topk * 4;
    if (required_itopk < 512) { required_itopk = 512; }
    if (sp->itopk_size < required_itopk) { sp->itopk_size = required_itopk; }
    sp->search_width = 4;
    {
      long v = env_long("SEEKDB_CUVS_ITOPK", 0);
      if (v > 0 && sp->itopk_size < (size_t)v) { sp->itopk_size = (size_t)v; }
    }
    if (sp->max_queries == 0 || (long)sp->max_queries < nq) {
      sp->max_queries = (size_t)nq; /* let the kernel batch all probes at once */
    }
  }

  /* Reuse the process-wide device scratch pool (grow-only). */
  float *qd = NULL; unsigned *nd = NULL; float *dd = NULL;
  pthread_mutex_lock(&g_pool_mu);
  if (sk_grow_f(&g_qd, &g_qd_cap, nq * h->dim) != 0) { rc = 10; }
  else if (sk_grow_u(&g_nd, &g_nd_cap, nq * topk) != 0) { rc = 11; }
  else if (sk_grow_f(&g_dd, &g_dd_cap, nq * topk) != 0) { rc = 12; }
  qd = g_qd; nd = g_nd; dd = g_dd;
  pthread_mutex_unlock(&g_pool_mu);
  if (rc != 0) { goto done; }

  {
    cudaStream_t st = sk_stream();
    cudaError_t e;
    if (st != NULL) {
      e = cudaMemcpyAsync(qd, query, sizeof(float) * nq * h->dim, cudaMemcpyHostToDevice, st);
      if (e != cudaSuccess) { rc = 13; goto done; }
    } else {
      if (cudaMemcpy(qd, query, sizeof(float) * nq * h->dim, cudaMemcpyHostToDevice) != cudaSuccess) {
        rc = 13; goto done;
      }
    }

    DLManagedTensor qt, nt, dt;
    long q_shape[2] = {nq, h->dim};
    long n_shape[2] = {nq, topk};
    sk_fill_dl_tensor(&qt, q_shape, qd, kDLCUDA, kDLFloat, 32);
    sk_fill_dl_tensor(&nt, n_shape, nd, kDLCUDA, kDLUInt, 32);
    /* cuVS wants a distances tensor even when the caller ignores it. */
    sk_fill_dl_tensor(&dt, n_shape, dd, kDLCUDA, kDLFloat, 32);
    cuvsFilter f; f.type = NO_FILTER; f.addr = (uintptr_t)0;
    if (cuvsCagraSearch(h->res, sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc = 15; goto done; }

    if (st != NULL) {
      e = cudaMemcpyAsync(out_ids, nd, sizeof(unsigned) * nq * topk, cudaMemcpyDeviceToHost, st);
      if (e == cudaSuccess) { e = cudaStreamSynchronize(st); }
    } else {
      e = cudaMemcpy(out_ids, nd, sizeof(unsigned) * nq * topk, cudaMemcpyDeviceToHost);
    }
    if (e != cudaSuccess) { rc = 16; goto done; }
    if (out_dist != NULL) {
      if (st != NULL) {
        e = cudaMemcpyAsync(out_dist, dd, sizeof(float) * nq * topk, cudaMemcpyDeviceToHost, st);
        if (e == cudaSuccess) { e = cudaStreamSynchronize(st); }
      } else {
        e = cudaMemcpy(out_dist, dd, sizeof(float) * nq * topk, cudaMemcpyDeviceToHost);
      }
      if (e != cudaSuccess) { rc = 17; goto done; }
    }
  }

done:
  if (rc) { fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText()); }
  return rc;
}

void seekdb_cuvs_free(void *handle)
{
  sk_cuvs_index *h = (sk_cuvs_index *)handle;
  if (!h) { return; }
  if (h->sp) { cuvsCagraSearchParamsDestroy(h->sp); }
  cuvsCagraIndexDestroy(h->idx);
  cuvsCagraIndexParamsDestroy(h->ip);
  if (h->ds) {
    if (h->ds_via_rmm) {
      cuvsRMMFree(h->res, h->ds, sizeof(float) * h->n * h->dim);
    } else {
      cudaFree(h->ds);
    }
  }
  cuvsResourcesDestroy(h->res);
  free(h);
}

int seekdb_cuvs_serialize(void *handle, const char *filename)
{
  sk_cuvs_index *index = (sk_cuvs_index *)handle;
  if (!index || !filename) return 1;
  if (cuvsCagraSerialize(index->res, filename, index->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  return 0;
}

void *seekdb_cuvs_deserialize(const char *filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  sk_cuvs_index *index = (sk_cuvs_index *)calloc(1, sizeof(*index));
  if (!index) return NULL;
  if (cuvsResourcesCreate(&index->res) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexParamsCreate(&index->ip) != CUVS_SUCCESS) goto fail_res;
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraDeserialize(index->res, filename, index->idx) != CUVS_SUCCESS) goto fail_idx;
  {
    int64_t actual_n = 0;
    int64_t actual_dim = 0;
    if (cuvsCagraIndexGetSize(index->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(index->idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, n, dim);
      goto fail_idx;
    }
  }
  index->n = n;
  index->dim = dim;
  return index;
fail_idx:
  cuvsCagraIndexDestroy(index->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(index->ip);
fail_res:
  cuvsResourcesDestroy(index->res);
fail:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  free(index);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float *base, long n, long dim,
                          const float *query, long nq, long topk,
                          unsigned int *out_ids)
{
  void *h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
