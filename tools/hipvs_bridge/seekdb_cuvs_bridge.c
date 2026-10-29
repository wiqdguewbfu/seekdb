/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here.
 *
* Optimized on top of the official baseline v5 (634888d9):
 *  - Build params tuned within the Recall-safe margin: NN-Descent iterations
 *    and intermediate graph degree lowered from the baseline's very
 *    conservative (niter=100, inter=256); graph_degree and the search-time
 *    quality knobs are kept identical to the baseline so Recall@k stays put.
 *    Every knob is overridable via env vars for cloud calibration:
 *        SEEKDB_CUVS_NN_NITER     (default 40,   baseline 100)
 *        SEEKDB_CUVS_INTER_DEGREE (default 192,  baseline 256)
 *        SEEKDB_CUVS_GRAPH_DEGREE (default 128,  baseline 128)
 *  - Per-handle search scratch pool: the baseline allocates and frees three
 *    RMM buffers on every search; we grow them once per handle and reuse,
 *    under a mutex because concurrent SQL jobs may share one handle.
 *  - cuvsCagraSearchParams_t cached per handle (created once, reused), with
 *    the baseline's itopk/search_width quality settings applied verbatim.
 *  - serialize/deserialize (DDL persistence) unchanged from the baseline.
 * All cuVS struct fields and symbols used below are exactly the ones the
 * official baseline itself uses, so the compile/link surface is identical. */
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

static long env_long(const char* name, long dflt)
{
  const char* v = getenv(name);
  if (!v || !*v) return dflt;
  char* end = 0;
  long parsed = strtol(v, &end, 10);
  return end == v ? dflt : parsed;
}

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t       idx;
  cuvsCagraSearchParams_t sp;   /* created once, reused across searches */
  long n, dim;
  /* per-handle search scratch pool, guarded by mu (shared handle) */
  pthread_mutex_t mu;
  int mu_init;
  float*   qd;  long qd_cap;   /* device query buffer, cap in floats   */
  unsigned* nd;  long nd_cap;  /* device neighbor ids, cap in unsigned */
  float*   dd;  long dd_cap;   /* device distances, cap in floats      */
} sk_cuvs_index;

static int sk_pool_grow(sk_cuvs_index* h,
                        float** fbuf, unsigned** ubuf, long* cap,
                        int is_float, long need)
{
  /* grow-only: alloc new with the handle's res, free the old one right
   * away (same res, both alive for the whole handle lifetime) */
  if (need <= *cap) return 0;
  void* nb = 0;
  size_t bytes = is_float ? sizeof(float) * (size_t)need
                          : sizeof(unsigned) * (size_t)need;
  if (cuvsRMMAlloc(h->res, &nb, bytes) != CUVS_SUCCESS) return 1;
  void* old = is_float ? (void*)*fbuf : (void*)*ubuf;
  if (old) {
    size_t old_bytes = is_float ? sizeof(float) * (size_t)*cap
                                : sizeof(unsigned) * (size_t)*cap;
    cuvsRMMFree(h->res, old, old_bytes);
  }
  if (is_float) *fbuf = (float*)nb; else *ubuf = (unsigned*)nb;
  *cap = need;
  return 0;
}

static void sk_apply_search_quality(cuvsCagraSearchParams_t sp, long topk)
{
  /* identical to the official baseline v5 */
  size_t required_itopk = (size_t)topk * 4;
  if (required_itopk < 512) required_itopk = 512;
  if (sp->itopk_size < required_itopk) sp->itopk_size = required_itopk;
  sp->search_width = 4;
}

static void* sk_index_init_finish(sk_cuvs_index* h)
{
  pthread_mutex_init(&h->mu, 0);
  h->mu_init = 1;
  if (cuvsCagraSearchParamsCreate(&h->sp) != CUVS_SUCCESS) h->sp = 0;
  return h;
}

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h); return NULL; }

  DLManagedTensor dset; long ds[2] = {n, dim};
  dset.dl_tensor.data = (void*)base;
  dset.dl_tensor.device.device_type = kDLCPU; dset.dl_tensor.device.device_id = 0;
  dset.dl_tensor.ndim = 2;
  dset.dl_tensor.dtype.code = kDLFloat; dset.dl_tensor.dtype.bits = 32; dset.dl_tensor.dtype.lanes = 1;
  dset.dl_tensor.shape = ds; dset.dl_tensor.strides = 0; dset.dl_tensor.byte_offset = 0;

  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail_res;
  {
    /* Baseline v5 quality config with the build-time knobs relaxed inside
     * the Recall-safe margin. graph_degree (the recall-critical knob) and
     * everything search-side stay exactly on the baseline. */
    long graph_degree   = env_long("SEEKDB_CUVS_GRAPH_DEGREE", 128);
    long inter_degree   = env_long("SEEKDB_CUVS_INTER_DEGREE", 192);
    long nn_niter       = env_long("SEEKDB_CUVS_NN_NITER", 40);
    if (graph_degree < 32)  graph_degree = 32;
    if (inter_degree < graph_degree) inter_degree = graph_degree;
    if (nn_niter < 10) nn_niter = 10;
    h->ip->build_algo = NN_DESCENT;
    h->ip->graph_degree = (size_t)graph_degree;
    h->ip->intermediate_graph_degree = (size_t)inter_degree;
    h->ip->nn_descent_niter = (size_t)nn_niter;
  }
  if (cuvsCagraIndexCreate(&h->idx)      != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) goto fail_idx;
  h->n = n; h->dim = dim;
  return sk_index_init_finish(h);

fail_idx: cuvsCagraIndexDestroy(h->idx);
fail_ip:  cuvsCagraIndexParamsDestroy(h->ip);
fail_res: cuvsResourcesDestroy(h->res); free(h);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return 1;
  int rc = 0;
  float* qd = 0; unsigned* nd = 0; float* dd = 0;
  cuvsCagraSearchParams_t sp = 0;
  cuvsCagraSearchParams_t tmp_sp = 0;
  long q_need = nq * h->dim;
  long k_need = nq * topk;

  if (h->mu_init) pthread_mutex_lock(&h->mu);

  if (sk_pool_grow(h, &h->qd, 0, &h->qd_cap, 1, q_need) != 0) { rc = 10; goto done; }
  if (sk_pool_grow(h, 0, &h->nd, &h->nd_cap, 0, k_need) != 0) { rc = 11; goto done; }
  if (sk_pool_grow(h, &h->dd, 0, &h->dd_cap, 1, k_need) != 0) { rc = 12; goto done; }
  qd = h->qd; nd = h->nd; dd = h->dd;

  if (cudaMemcpy(qd, query, sizeof(float)*nq*h->dim, cudaMemcpyDefault) != cudaSuccess) { rc = 13; goto done; }
  {
    DLManagedTensor qt, nt, dt; long qs[2]={nq,h->dim}, nsh[2]={nq,topk}, dsh[2]={nq,topk};
    qt.dl_tensor.data=qd; qt.dl_tensor.device.device_type=kDLCUDA; qt.dl_tensor.device.device_id=0;
    qt.dl_tensor.ndim=2; qt.dl_tensor.dtype.code=kDLFloat; qt.dl_tensor.dtype.bits=32; qt.dl_tensor.dtype.lanes=1;
    qt.dl_tensor.shape=qs; qt.dl_tensor.strides=0; qt.dl_tensor.byte_offset=0;
    nt=qt; nt.dl_tensor.data=nd; nt.dl_tensor.dtype.code=kDLUInt; nt.dl_tensor.shape=nsh;
    dt=qt; dt.dl_tensor.data=dd; dt.dl_tensor.dtype.code=kDLFloat; dt.dl_tensor.shape=dsh;

    sp = h->sp;
    if (sp == NULL) {
      if (cuvsCagraSearchParamsCreate(&tmp_sp) != CUVS_SUCCESS) { rc = 14; goto done; }
      sp = tmp_sp;
    }
    sk_apply_search_quality(sp, topk);
    cuvsFilter f; f.type=NO_FILTER; f.addr=(uintptr_t)0;
    if (cuvsCagraSearch(h->res, sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc = 15; goto done; }
    if (cudaMemcpy(out_ids, nd, sizeof(unsigned)*nq*topk, cudaMemcpyDefault) != cudaSuccess) { rc = 16; goto done; }
    if (out_dist) cudaMemcpy(out_dist, dd, sizeof(float)*nq*topk, cudaMemcpyDefault);
  }
done:
  if (h->mu_init) pthread_mutex_unlock(&h->mu);
  if (tmp_sp) cuvsCagraSearchParamsDestroy(tmp_sp);
  if (rc) fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return;
  if (h->qd) cuvsRMMFree(h->res, h->qd, sizeof(float) * (size_t)h->qd_cap);
  if (h->nd) cuvsRMMFree(h->res, h->nd, sizeof(unsigned) * (size_t)h->nd_cap);
  if (h->dd) cuvsRMMFree(h->res, h->dd, sizeof(float) * (size_t)h->dd_cap);
  if (h->sp) cuvsCagraSearchParamsDestroy(h->sp);
  cuvsCagraIndexDestroy(h->idx);
  cuvsCagraIndexParamsDestroy(h->ip);
  cuvsResourcesDestroy(h->res);
  if (h->mu_init) pthread_mutex_destroy(&h->mu);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  if (!index || !filename) return 1;
  if (cuvsCagraSerialize(index->res, filename, index->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  sk_cuvs_index* index = (sk_cuvs_index*)calloc(1, sizeof(*index));
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
  return sk_index_init_finish(index);
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

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
