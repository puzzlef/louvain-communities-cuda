#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "properties.hxx"
#include "louvain.hxx"
#include "hashtableCuda.hxx"

using std::vector;
using std::count_if;
using std::partition;
using std::max;
using std::min;




#pragma region DEGREE LIMITS
#ifndef LOUVAIN_DEGREE_THREAD
/** Degree limit for using thread-per-vertex kernel. */
#define LOUVAIN_DEGREE_THREAD 32
/** Degree limit for using shared memory with block-per-vertex kernel. */
#define LOUVAIN_DEGREE_BLOCK_SHARED 128
#endif
#pragma endregion




#pragma region METHODS
#pragma region INITIALIZE
/**
 * Find the total edge weight of each vertex, using thread-per-vertex approach [kernel].
 * @tparam SDEG switch-degree between thread- and block-per-vertex kernels
 * @param vtot total edge weight of each vertex (output)
 * @param xoff offsets of input graph
 * @param xdeg degrees of input graph
 * @param xwei edge values of input graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <int SDEG=LOUVAIN_DEGREE_THREAD, class O, class K, class V, class W>
void __global__ louvainVertexWeightsThreadCukW(W *vtot, const O *xoff, const K *xdeg, const V *xwei, K NB, K NE) {
  DEFINE_CUDA(t, b, B, G);
  for (K u=NB+B*b+t; u<NE; u+=G*B) {
    size_t EO = xoff[u];
    size_t EN = xdeg[u];
    if (EN >= SDEG) continue;  // Skip high-degree vertices
    W w = W();
    for (size_t i=0; i<EN; ++i)
      w += xwei[EO+i];
    vtot[u] = w;
  }
}


/**
 * Find the total edge weight of each vertex, using thread-per-vertex approach.
 * @tparam SDEG switch-degree between thread- and block-per-vertex kernels
 * @param vtot total edge weight of each vertex (output)
 * @param xoff offsets of input graph
 * @param xdeg degrees of input graph
 * @param xwei edge values of input graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <int SDEG=LOUVAIN_DEGREE_THREAD, class O, class K, class V, class W>
inline void louvainVertexWeightsThreadCuW(W *vtot, const O *xoff, const K *xdeg, const V *xwei, K NB, K NE) {
  const int B = blockSizeCu(NE-NB, LOUVAIN_DEGREE_THREAD);
  const int G = gridSizeCu (NE-NB, B, GRID_LIMIT_MAP_CUDA);
  louvainVertexWeightsThreadCukW<SDEG><<<G, B>>>(vtot, xoff, xdeg, xwei, NB, NE);
}


/**
 * Find the total edge weight of each vertex, using block-per-vertex approach [kernel].
 * @tparam SDEG switch-degree between thread- and block-per-vertex kernels
 * @param vtot total edge weight of each vertex (output)
 * @param xoff offsets of input graph
 * @param xdeg degrees of input graph
 * @param xwei edge values of input graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <int SDEG=LOUVAIN_DEGREE_THREAD, class O, class K, class V, class W>
void __global__ louvainVertexWeightsBlockCukW(W *vtot, const O *xoff, const K *xdeg, const V *xwei, K NB, K NE) {
  DEFINE_CUDA(t, b, B, G);
  __shared__ W cache[BLOCK_LIMIT_MAP_CUDA];
  for (K u=NB+b; u<NE; u+=G) {
    size_t EO = xoff[u];
    size_t EN = xdeg[u];
    if (EN < SDEG) continue;  // Skip low-degree vertices
    W w = W();
    for (size_t i=t; i<EN; i+=B)
      w += xwei[EO+i];
    cache[t] = w;
    __syncthreads();
    sumValuesBlockReduceCudU(cache, B, t);
    if (t==0) vtot[u] = cache[0];
  }
}


/**
 * Find the total edge weight of each vertex, using block-per-vertex approach.
 * @tparam SDEG switch-degree between thread- and block-per-vertex kernels
 * @param vtot total edge weight of each vertex (output)
 * @param xoff offsets of input graph
 * @param xdeg degrees of input graph
 * @param xwei edge values of input graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <int SDEG=LOUVAIN_DEGREE_THREAD, class O, class K, class V, class W>
inline void louvainVertexWeightsBlockCuW(W *vtot, const O *xoff, const K *xdeg, const V *xwei, K NB, K NE) {
  const int B = blockSizeCu<true>(NE-NB, BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu <true>(NE-NB, B, GRID_LIMIT_MAP_CUDA);
  louvainVertexWeightsBlockCukW<SDEG><<<G, B>>>(vtot, xoff, xdeg, xwei, NB, NE);
}


/**
 * Find the total edge weight of each community [kernel].
 * @param ctot total edge weight of each community (updated, must be initialized)
 * @param vcom community each vertex belongs to
 * @param vtot total edge weight of each vertex
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K, class W>
void __global__ louvainCommunityWeightsCukU(W *ctot, const K *vcom, const W *vtot, K NB, K NE) {
  DEFINE_CUDA(t, b, B, G);
  for (K u=NB+B*b+t; u<NE; u+=G*B) {
    K c = vcom[u];
    atomicAdd(&ctot[c], vtot[u]);  // Too many atomic ops.?
  }
}


/**
 * Find the total edge weight of each community.
 * @param ctot total edge weight of each community (updated, must be initialized)
 * @param vcom community each vertex belongs to
 * @param vtot total edge weight of each vertex
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K, class W>
inline void louvainCommunityWeightsCuU(W *ctot, const K *vcom, const W *vtot, K NB, K NE) {
  const int B = blockSizeCu(NE-NB, BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu (NE-NB, B, GRID_LIMIT_MAP_CUDA);
  louvainCommunityWeightsCukU<<<G, B>>>(ctot, vcom, vtot, NB, NE);
}


/**
 * Initialize communities such that each vertex is its own community [kernel].
 * @param vcom community each vertex belongs to (output)
 * @param ctot total edge weight of each community (output)
 * @param vtot total edge weight of each vertex
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K, class W>
void __global__ louvainInitializeCukW(K *vcom, W *ctot, const W *vtot, K NB, K NE) {
  DEFINE_CUDA(t, b, B, G);
  for (K u=NB+B*b+t; u<NE; u+=G*B) {
    vcom[u] = u;
    ctot[u] = vtot[u];
  }
}


/**
 * Initialize communities such that each vertex is its own community.
 * @param vcom community each vertex belongs to (output)
 * @param ctot total edge weight of each community (output)
 * @param vtot total edge weight of each vertex
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K, class W>
inline void louvainInitializeCuW(K *vcom, W *ctot, const W *vtot, K NB, K NE) {
  const int B = blockSizeCu(NE-NB, BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu (NE-NB, B, GRID_LIMIT_MAP_CUDA);
  louvainInitializeCukW<<<G, B>>>(vcom, ctot, vtot, NB, NE);
}
#pragma endregion




#pragma region CHOOSE COMMUNITY
/**
 * Scan communities connected to a vertex [device function].
 * @tparam SELF include self-loops?
 * @tparam BLOCK called from a thread block?
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
 * @param hk hashtable keys (updated)
 * @param hv hashtable values (updated)
 * @param H capacity of hashtable (prime)
 * @param T secondary prime (>H)
 * @param xoff offsets of input graph
 * @param xdeg degrees of input graph
 * @param xedg edge keys of input graph
 * @param xwei edge values of input graph
 * @param u given vertex
 * @param vcom community each vertex belongs to
 * @param i start index
 * @param DI index stride
 */
template <bool SELF=false, bool BLOCK=false, int HTYPE=3, class O, class K, class V, class W>
inline void __device__ louvainScanCommunitiesCudU(K *hk, W *hv, size_t H, size_t T, const O *xoff, const K *xdeg, const K *xedg, const V *xwei, K u, const K *vcom, size_t i, size_t DI) {
  size_t EO = xoff[u];
  size_t EN = xdeg[u];
  for (; i<EN; i+=DI) {
    K v = xedg[EO+i];
    W w = xwei[EO+i];
    K c = vcom[v];
    if (!SELF && u==v) continue;
    hashtableAccumulateCudU<BLOCK, HTYPE>(hk, hv, H, T, c+1, w);
  }
}


/**
 * Calculate delta modularity of moving a vertex to each community [device function].
 * @param hk hashtable keys
 * @param hv hashtable values (updated)
 * @param H capacity of hashtable (prime)
 * @param d previous community of given vertex
 * @param ctot total edge weight of each community
 * @param vdout edge weight of given vertex to previous community
 * @param vtot total edge weight of given vertex
 * @param dtot total edge weight of previous community
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @param i start index
 * @param DI index stride
 */
template <class K, class W>
inline void __device__ louvainCalculateDeltaModularityCudU(K *hk, W *hv, size_t H, K d, const W *ctot, W vdout, W vtot, W dtot, W M, W R, size_t i, size_t DI) {
  for (; i<H; i+=DI) {
    if (!hk[i]) continue;
    K c = hk[i] - 1;
    hv[i] = deltaModularityCud(hv[i], vdout, vtot, ctot[c], dtot, M, R);
  }
}


/**
 * Mark out-neighbors of a vertex as affected [device function].
 * @param vaff vertex affected flags (updated)
 * @param xoff offsets of input graph
 * @param xdeg degrees of input graph
 * @param xedg edge keys of input graph
 * @param u given vertex
 * @param i start index
 * @param DI index stride
 */
template <class O, class K, class F>
inline void __device__ louvainMarkNeighborsCudU(F *vaff, const O *xoff, const K *xdeg, const K *xedg, K u, size_t i, size_t DI) {
  size_t EO = xoff[u];
  size_t EN = xdeg[u];
  for (; i<EN; i+=DI) {
    K v = xedg[EO+i];
    vaff[v] = F(1);  // Use two (synchronous) buffers?
  }
}
#pragma endregion




#pragma region LOCAL-MOVING PHASE
/**
 * Move each vertex to its best community, using thread-per-vertex approach [kernel].
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
 * @tparam BLIM maximum number of threads per block
 * @param el delta modularity of moving all vertices to their best communities (output)
 * @param vcom community each vertex belongs to (updated)
 * @param ctot total edge weight of each community (updated)
 * @param vaff vertex affected flags (updated)
 * @param bufk buffer for hashtable keys (updated)
 * @param bufw buffer for hashtable values (updated)
 * @param xoff offsets of input graph
 * @param xdeg degrees of input graph
 * @param xedg edge keys of input graph
 * @param xwei edge values of input graph
 * @param vtot total edge weight of each vertex
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <int HTYPE=3, int BLIM=LOUVAIN_DEGREE_THREAD, class O, class K, class V, class W, class F>
void __global__ louvainMoveThreadCukU(double *el, K *vcom, W *ctot, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xdeg, const K *xedg, const V *xwei, const W *vtot, W M, W R, K NB, K NE, bool PICKLESS) {
  DEFINE_CUDA(t, b, B, G);
  __shared__ double elb[BLIM];
  const int DMAX = BLIM;
  K shrk[2 * DMAX];
  W shrw[2 * DMAX];
  elb[t] = 0;
  for (K u=NB+B*b+t; u<NE; u+=G*B) {
    if (!vaff[u]) continue;
    // Scan communities connected to u.
    K d = vcom[u];
    // size_t EO = xoff[u]
    size_t EN = xdeg[u];
    if (EN == 0 || EN >= LOUVAIN_DEGREE_THREAD) continue;  // Skip isolated and high-degree vertices
    size_t H = nextPow2Cud(EN) - 1;
    size_t T = nextPow2Cud(H)  - 1;
    K *hk = shrk; // bufk + 2*EO
    W *hv = shrw; // bufw + 2*EO
    hashtableClearCudW(hk, hv, H, 0, 1);
    louvainScanCommunitiesCudU<false, false, HTYPE>(hk, hv, H, T, xoff, xdeg, xedg, xwei, u, vcom, 0, 1);
    // Calculate delta modularity of moving u to each community.
    W vdout = hashtableGetCud(hk, hv, H, T, d+1);  // Can be optimized?
    louvainCalculateDeltaModularityCudU(hk, hv, H, d, ctot, vdout, vtot[u], ctot[d], M, R, 0, 1);
    // Find best community for u.
    hashtableMaxCudU(hk, hv, H, 0, 1);
    vaff[u] = F();               // Mark u as unaffected (Use two buffers?)
    if (hv[0] <= W()) continue;  // No good community found
    K c = hk[0] - 1;             // Best community
    if (c==d) continue;
    if (PICKLESS && c>d) continue;  // Pick smaller community-id (to avoid community swaps)
    // Change community of u.
    atomicAdd(&ctot[d], -vtot[u]);
    atomicAdd(&ctot[c],  vtot[u]);
    vcom[u] = c;
    elb[t] += hv[0];
    louvainMarkNeighborsCudU(vaff, xoff, xdeg, xedg, u, 0, 1);
  }
  // Update total delta modularity.
  __syncthreads();
  sumValuesBlockReduceCudU(elb, B, t);
  if (t==0) atomicAdd(el, elb[0]);
}


/**
 * Move each vertex to its best community, using thread-per-vertex approach.
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
 * @tparam BLIM maximum number of threads per block
 * @param el delta modularity of moving all vertices to their best communities (output)
 * @param vcom community each vertex belongs to (updated)
 * @param ctot total edge weight of each community (updated)
 * @param vaff vertex affected flags (updated)
 * @param bufk buffer for hashtable keys (updated)
 * @param bufw buffer for hashtable values (updated)
 * @param xoff offsets of input graph
 * @param xdeg degrees of input graph
 * @param xedg edge keys of input graph
 * @param xwei edge values of input graph
 * @param vtot total edge weight of each vertex
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <int HTYPE=3, int BLIM=LOUVAIN_DEGREE_THREAD, class O, class K, class V, class W, class F>
inline void louvainMoveThreadCuU(double *el, K *vcom, W *ctot, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xdeg, const K *xedg, const V *xwei, const W *vtot, W M, W R, K NB, K NE, bool PICKLESS) {
  const int B = blockSizeCu(NE-NB, BLIM);
  const int G = gridSizeCu (NE-NB, B, GRID_LIMIT_MAP_CUDA);
  louvainMoveThreadCukU<HTYPE, BLIM><<<G, B>>>(el, vcom, ctot, vaff, bufk, bufw, xoff, xdeg, xedg, xwei, vtot, M, R, NB, NE, PICKLESS);
}


/**
 * Move each vertex to its best community, using block-per-vertex approach [kernel].
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
 * @tparam BLIM maximum number of threads per block
 * @param el delta modularity of moving all vertices to their best communities (output)
 * @param vcom community each vertex belongs to (updated)
 * @param ctot total edge weight of each community (updated)
 * @param vaff vertex affected flags (updated)
 * @param bufk buffer for hashtable keys (updated)
 * @param bufw buffer for hashtable values (updated)
 * @param xoff offsets of input graph
 * @param xdeg degrees of input graph
 * @param xedg edge keys of input graph
 * @param xwei edge values of input graph
 * @param vtot total edge weight of each vertex
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <int HTYPE=3, int BLIM=LOUVAIN_DEGREE_BLOCK_SHARED, class O, class K, class V, class W, class F>
void __global__ louvainMoveBlockCukU(double *el, K *vcom, W *ctot, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xdeg, const K *xedg, const V *xwei, const W *vtot, W M, W R, K NB, K NE, bool PICKLESS) {
  DEFINE_CUDA(t, b, B, G);
  const int DMAX = BLIM;
  __shared__ K shrk[2*DMAX];
  __shared__ W shrw[2*DMAX];
  __shared__ double elb;
  __shared__ bool vaffu;
  __shared__ K d;
  __shared__ W vdout;
  if (t==0) elb = 0;
  for (K u=NB+b; u<NE; u+=G) {
    __syncthreads();
    if (t==0) vaffu = vaff[u];
    if (t==0) d = vcom[u];
    __syncthreads();
    if (!vaffu) continue;
    // Scan communities connected to u.
    size_t EO = xoff[u];
    size_t EN = xdeg[u];
    if (EN==0 || EN < LOUVAIN_DEGREE_THREAD) continue;  // Skip isolated and low-degree vertices
    size_t H = nextPow2Cud(EN) - 1;
    size_t T = nextPow2Cud(H)  - 1;
    K *hk = EN <= DMAX? shrk : bufk + 2*EO;
    W *hv = EN <= DMAX? shrw : bufw + 2*EO;
    __syncthreads();
    hashtableClearCudW(hk, hv, H, t, B);
    __syncthreads();
    louvainScanCommunitiesCudU<false, true, HTYPE>(hk, hv, H, T, xoff, xdeg, xedg, xwei, u, vcom, t, B);
    __syncthreads();
    // Calculate delta modularity of moving u to each community.
    if (t==0) vdout = hashtableGetCud(hk, hv, H, T, d+1);  // Can be optimized?
    __syncthreads();
    louvainCalculateDeltaModularityCudU(hk, hv, H, d, ctot, vdout, vtot[u], ctot[d], M, R, t, B);
    __syncthreads();
    // Find best community for u.
    hashtableMaxCudU<true>(hk, hv, H, t, B);
    __syncthreads();
    if (t==0) vaff[u] = F();               // Mark u as unaffected (Use two buffers?)
    if (!hk[0] || hv[0] <= W()) continue;  // No good community found (Cache hk[0], hv[0]?)
    K c = hk[0] - 1;                       // Best community
    if (c==d) continue;
    if (PICKLESS && c>d) continue;  // Pick smaller community-id (to avoid community swaps)
    // Change community of u.
    if (t==0) atomicAdd(&ctot[d], -vtot[u]);
    if (t==0) atomicAdd(&ctot[c],  vtot[u]);
    if (t==0) vcom[u] = c;
    if (t==0) elb += hv[0];
    louvainMarkNeighborsCudU(vaff, xoff, xdeg, xedg, u, t, B);
  }
  // Update total delta modularity.
  if (t==0) atomicAdd(el, elb);
}


/**
 * Move each vertex to its best community, using block-per-vertex approach.
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
 * @tparam BLIM maximum number of threads per block
 * @param el delta modularity of moving all vertices to their best communities (output)
 * @param vcom community each vertex belongs to (updated)
 * @param ctot total edge weight of each community (updated)
 * @param vaff vertex affected flags (updated)
 * @param bufk buffer for hashtable keys (updated)
 * @param bufw buffer for hashtable values (updated)
 * @param xoff offsets of input graph
 * @param xdeg degrees of input graph
 * @param xedg edge keys of input graph
 * @param xwei edge values of input graph
 * @param vtot total edge weight of each vertex
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <int HTYPE=3, int BLIM=LOUVAIN_DEGREE_BLOCK_SHARED, class O, class K, class V, class W, class F>
inline void louvainMoveBlockCuU(double *el, K *vcom, W *ctot, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xdeg, const K *xedg, const V *xwei, const W *vtot, W M, W R, K NB, K NE, bool PICKLESS) {
  const int B = blockSizeCu<true>(NE-NB, BLIM);
  const int G = gridSizeCu <true>(NE-NB, B, GRID_LIMIT_MAP_CUDA);
  louvainMoveBlockCukU<HTYPE, BLIM><<<G, B>>>(el, vcom, ctot, vaff, bufk, bufw, xoff, xdeg, xedg, xwei, vtot, M, R, NB, NE, PICKLESS);
}


/**
 * Louvain algorithm's local moving phase
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
 * @param el delta modularity of moving all vertices to their best communities (output)
 * @param vcom community each vertex belongs to (updated)
 * @param ctot total edge weight of each community (updated)
 * @param vaff vertex affected flags (updated)
 * @param bufk buffer for hashtable keys (updated)
 * @param bufw buffer for hashtable values (updated)
 * @param xoff offsets of input graph
 * @param xdeg degrees of input graph
 * @param xedg edge keys of input graph
 * @param xwei edge values of input graph
 * @param vtot total edge weight of each vertex
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @param L max interations
 * @param N Total number of vertices
 * @param NL number of vertices with low degree
 * @param fc has local moving phase converged?
 */
template <int HTYPE=3, class O, class K, class V, class W, class F, class FC>
inline int louvainMoveCuU(double *el, K *vcom, W *ctot, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xdeg, const K *xedg, const V *xwei, const W *vtot, W M, W R, int L, K N, K NL, FC fc) {
  int l = 0;
  double elH = 0;
  const int PICKSTEP = 4;
  while (l < L) {
    bool PICKLESS = (l + PICKSTEP / 2) % PICKSTEP == 0;
    fillValueCuW(el, 1, 0.0);
    louvainMoveThreadCuU<HTYPE>(el, vcom, ctot, vaff, bufk, bufw, xoff, xdeg, xedg, xwei, vtot, M, R, K(), N, PICKLESS);
    louvainMoveBlockCuU <HTYPE>(el, vcom, ctot, vaff, bufk, bufw, xoff, xdeg, xedg, xwei, vtot, M, R, K(), N, PICKLESS);
    TRY_CUDA( cudaMemcpy(&elH, el, sizeof(double), cudaMemcpyDeviceToHost) );
    if (fc(elH, l++)) break;
  }
  return l>1 || elH? l : 0;
}
#pragma endregion




#pragma region COMMUNITY PROPERTIES
/**
 * Examine if each community exists [kernel].
 * @param C number of communities (updated, must be initialized)
 * @param a does each community exist (updated, must be initialized)
 * @param vcom community each vertex belongs to
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K, class A>
void __global__ louvainCommunityExistsCukU(uint64_cu *C, A *a, const K *vcom, K NB, K NE) {
  DEFINE_CUDA(t, b, B, G);
  __shared__ uint64_cu CB[BLOCK_LIMIT_MAP_CUDA];
  CB[t] = uint64_cu();
  for (K u=NB+B*b+t; u<NE; u+=G*B) {
    K c = vcom[u];
    if (a[c]) continue;
    if (atomicCAS(&a[c], uint64_cu(), uint64_cu(1)) == uint64_cu()) ++CB[t];  // Too many atomic ops.?
  }
  __syncthreads();
  sumValuesBlockReduceCudU(CB, B, t);
  if (t==0) atomicAdd(C, CB[0]);
}


/**
 * Examine if each community exists.
 * @param C number of communities (updated, must be initialized)
 * @param a does each community exist (updated, must be initialized)
 * @param vcom community each vertex belongs to
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K, class A>
inline void louvainCommunityExistsCuU(uint64_cu *C, A *a, const K *vcom, K NB, K NE) {
  const int B = blockSizeCu(NE-NB, BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu (NE-NB, B, GRID_LIMIT_MAP_CUDA);
  louvainCommunityExistsCukU<<<G, B>>>(C, a, vcom, NB, NE);
}


/**
 * Find the total degree of each community [kernel].
 * @param a total degree of each community (updated, must be initialized)
 * @param xdeg degrees of input graph
 * @param vcom community each vertex belongs to
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K, class A>
void __global__ louvainCommunityTotalDegreeCukU(A *a, const K *xdeg, const K *vcom, K NB, K NE) {
  DEFINE_CUDA(t, b, B, G);
  for (K u=NB+B*b+t; u<NE; u+=G*B) {
    size_t EN = xdeg[u];
    K c = vcom[u];
    atomicAdd(&a[c], A(EN));  // Too many atomic ops.?
  }
}


/**
 * Find the total degree of each community.
 * @param a total degree of each community (updated, must be initialized)
 * @param xdeg offsets of input graph
 * @param vcom community each vertex belongs to
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K, class A>
inline void louvainCommunityTotalDegreeCuU(A *a, const K *xdeg, const K *vcom, K NB, K NE) {
  const int B = blockSizeCu(NE-NB, BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu (NE-NB, B, GRID_LIMIT_MAP_CUDA);
  louvainCommunityTotalDegreeCukU<<<G, B>>>(a, xdeg, vcom, NB, NE);
}


/**
 * Find the number of vertices in each community [kernel].
 * @param a number of vertices belonging to each community (updated, must be initialized)
 * @param vcom community each vertex belongs to
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K, class A>
void __global__ louvainCountCommunityVerticesCukU(A *a, const K *vcom, K NB, K NE) {
  DEFINE_CUDA(t, b, B, G);
  for (K u=NB+B*b+t; u<NE; u+=G*B) {
    K c = vcom[u];
    atomicAdd(&a[c], A(1));  // Too many atomic ops.?
  }
}


/**
 * Find the number of vertices in each community.
 * @param a number of vertices belonging to each community (updated, must be initialized)
 * @param vcom community each vertex belongs to
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K, class A>
inline void louvainCountCommunityVerticesCuU(A *a, const K *vcom, K NB, K NE) {
  const int B = blockSizeCu(NE-NB, BLOCK_LIMIT_REDUCE_CUDA);
  const int G = gridSizeCu (NE-NB, B, GRID_LIMIT_REDUCE_CUDA);
  louvainCountCommunityVerticesCukU<<<G, B>>>(a, vcom, NB, NE);
}


/**
 * Populate community vertices into a CSR structure.
 * @param cdeg number of vertices in each community (updated)
 * @param cedg vertices in each community (updates)
 * @param coff offsets of vertices in each community
 * @param vcom community each vertex belongs to
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K>
void __global__ louvainPopulateCommunityVerticesCukU(K *cdeg, K *cedg, const K *coff, const K *vcom, K NB, K NE) {
  DEFINE_CUDA(t, b, B, G);
  for (K u=NB+B*b+t; u<NE; u+=G*B) {
    K c = vcom[u];
    K n = atomicAdd(&cdeg[c], K(1));
    K i = coff[c] + n;
    cedg[i] = u;
  }
}


/**
 * Populate community vertices into a CSR structure.
 * @param cdeg number of vertices in each community (updated)
 * @param cedg vertices in each community (updated)
 * @param coff offsets of vertices in each community
 * @param vcom community each vertex belongs to
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K>
inline void louvainPopulateCommunityVerticesCuU(K *cdeg, K *cedg, const K *coff, const K *vcom, K NB, K NE) {
  const int B = blockSizeCu(NE-NB, BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu (NE-NB, B, GRID_LIMIT_MAP_CUDA);
  louvainPopulateCommunityVerticesCukU<<<G, B>>>(cdeg, cedg, coff, vcom, NB, NE);
}


/**
 * Find the vertices in each community.
 * @param coff csr offsets for vertices belonging to each community (output)
 * @param cdeg number of vertices in each community (output)
 * @param cedg vertices belonging to each community (output)
 * @param bufk buffer for exclusive scan (scratch)
 * @param vcom community each vertex belongs to
 * @param N number of vertices
 * @param C number of communities
 * @param SCAN size of buffer for exclusive scan
 */
template <class K>
inline void louvainCommunityVerticesCuW(K *coff, K *cdeg, K *cedg, K *bufk, const K *vcom, K N, K C, size_t SCAN) {
  fillValueCuW(cdeg, C, K());
  fillValueCuW(coff, C+1, K());
  louvainCountCommunityVerticesCuU(coff, vcom, K(), N);
  exclusiveScanCubW(coff, bufk, coff, C+1, SCAN);
  louvainPopulateCommunityVerticesCuU(cdeg, cedg, coff, vcom, K(), N);
}
#pragma endregion




#pragma region LOOKUP COMMUNITIES
/**
 * Update community membership in a tree-like fashion (to handle aggregation) [kernel].
 * @param a output community each vertex belongs to (updated)
 * @param vcom community each vertex belongs to (at this aggregation level)
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K>
void __global__ louvainLookupCommunitiesCukU(K *a, const K *vcom, K NB, K NE) {
  DEFINE_CUDA(t, b, B, G);
  for (K u=NB+B*b+t; u<NE; u+=G*B)
    a[u] = vcom[a[u]];
}


/**
 * Update community membership in a tree-like fashion (to handle aggregation).
 * @param a output community each vertex belongs to (updated)
 * @param vcom community each vertex belongs to (at this aggregation level)
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K>
inline void louvainLookupCommunitiesCuU(K *a, const K *vcom, K NB, K NE) {
  const int B = blockSizeCu(NE-NB, BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu (NE-NB, B, GRID_LIMIT_MAP_CUDA);
  louvainLookupCommunitiesCukU<<<G, B>>>(a, vcom, NB, NE);
}
#pragma endregion




#pragma region AGGREGATION PHASE
/**
 * Re-number communities such that they are numbered 0, 1, 2, ...
 * @param vcom community each vertex belongs to (updated)
 * @param cext does each community exist (updated)
 * @param bufk buffer for exclusive scan (scratch)
 * @param N number of vertices
 * @param SCAN size of buffer for exclusive scan
 */
template <class K>
inline void louvainRenumberCommunitiesCuU(K *vcom, K *cext, K *bufk, K N, size_t SCAN) {
  exclusiveScanCubW(cext, bufk, cext, N, SCAN);
  louvainLookupCommunitiesCuU(vcom, cext, K(), N);
}


/**
 * Aggregate outgoing edges of each community, using thread-per-community approach [kernel].
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
 * @tparam BLIM maximum number of threads per block
 * @param ydeg degrees of aggregated graph (updated)
 * @param yedg edge keys of aggregated graph (updated)
 * @param ywei edge values of aggregated graph (updated)
 * @param bufk buffer to store keys of hashtable (scratch)
 * @param bufw buffer to store values of hashtable (scratch)
 * @param xoff offsets of input graph
 * @param xdeg degrees of input graph
 * @param xedg edge keys of input graph
 * @param xwei edge values of input graph
 * @param vcom community each vertex belongs to
 * @param coff offsets of vertices in each community
 * @param cedg vertices in each community
 * @param yoff offsets of aggregated graph
 * @param CB begin community (inclusive)
 * @param CE end community (exclusive)
 */
template <int HTYPE=3, int BLIM=LOUVAIN_DEGREE_THREAD, class O, class K, class V, class W>
void __global__ louvainAggregateEdgesThreadCukU(K *ydeg, K *yedg, V *ywei, K *bufk, W *bufw, const O *xoff, const K *xdeg, const K *xedg, const V *xwei, const K *vcom, const O *coff, const K *cedg, const O *yoff, K CB, K CE) {
  DEFINE_CUDA(t, b, B, G);
  const int DMAX = BLIM;
  K shrk[2*DMAX];
  W shrw[2*DMAX];
  for (K c=CB+B*b+t; c<CE; c+=G*B) {
    // size_t EO = yoff[c];
    size_t EN = yoff[c+1] - yoff[c];
    size_t CO = coff[c];
    size_t CN = coff[c+1] - coff[c];
    if (CN==0 || EN >= LOUVAIN_DEGREE_THREAD) continue;  // Skip empty communities, or those with high total degree
    size_t H = nextPow2Cud(EN) - 1;
    size_t T = nextPow2Cud(H)  - 1;
    K *hk = shrk; // bufk + 2*EO
    W *hv = shrw; // bufw + 2*EO
    // Get edges from community c into hashtable.
    hashtableClearCudW(hk, hv, H, 0, 1);
    for (size_t i=0; i<CN; ++i) {
      K u = cedg[CO+i];
      louvainScanCommunitiesCudU<true, false, HTYPE>(hk, hv, H, T, xoff, xdeg, xedg, xwei, u, vcom, 0, 1);
    }
    // Store edges from hashtable into aggregated graph.
    for (size_t i = 0; i < H; ++i) {
      if (hk[i] == K()) continue;
      K d = hk[i] - 1;
      K n = atomicAdd(&ydeg[c], K(1));
      O j = yoff[c] + n;
      yedg[j] = d;
      ywei[j] = V(hv[i]);
    }
  }
}


/**
 * Aggregate outgoing edges of each community, using thread-per-community approach.
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
 * @tparam BLIM maximum number of threads per block
 * @param ydeg degrees of aggregated graph (updated)
 * @param yedg edge keys of aggregated graph (updated)
 * @param ywei edge values of aggregated graph (updated)
 * @param bufk buffer to store keys of hashtable (scratch)
 * @param bufw buffer to store values of hashtable (scratch)
 * @param xoff offsets of input graph
 * @param xdeg degrees of input graph
 * @param xedg edge keys of input graph
 * @param xwei edge values of input graph
 * @param vcom community each vertex belongs to
 * @param coff offsets of vertices in each community
 * @param cedg vertices in each community
 * @param yoff offsets of aggregated graph
 * @param CB begin community (inclusive)
 * @param CE end community (exclusive)
 */
template <int HTYPE=3, int BLIM=LOUVAIN_DEGREE_THREAD, class O, class K, class V, class W>
inline void louvainAggregateEdgesThreadCuU(K *ydeg, K *yedg, V *ywei, K *bufk, W *bufw, const O *xoff, const K *xdeg, const K *xedg, const V *xwei, const K *vcom, const O *coff, const K *cedg, const O *yoff, K CB, K CE) {
  const int B = blockSizeCu(CE-CB, BLIM);
  const int G = gridSizeCu (CE-CB, B, GRID_LIMIT_MAP_CUDA);
  louvainAggregateEdgesThreadCukU<HTYPE, BLIM><<<G, B>>>(ydeg, yedg, ywei, bufk, bufw, xoff, xdeg, xedg, xwei, vcom, coff, cedg, yoff, CB, CE);
}


/**
 * Aggregate outgoing edges of each community, using block-per-community approach [kernel].
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
 * @tparam BLIM maximum number of threads per block
 * @param ydeg degrees of aggregated graph (updated)
 * @param yedg edge keys of aggregated graph (updated)
 * @param ywei edge values of aggregated graph (updated)
 * @param bufk buffer to store keys of hashtable (scratch)
 * @param bufw buffer to store values of hashtable (scratch)
 * @param xoff offsets of input graph
 * @param xdeg degrees of input graph
 * @param xedg edge keys of input graph
 * @param xwei edge values of input graph
 * @param vcom community each vertex belongs to
 * @param coff offsets of vertices in each community
 * @param cedg vertices in each community
 * @param yoff offsets of aggregated graph
 * @param CB begin community (inclusive)
 * @param CE end community (exclusive)
 */
template <int HTYPE=3, int BLIM=LOUVAIN_DEGREE_THREAD, class O, class K, class V, class W>
void __global__ louvainAggregateEdgesBlockCukU(K *ydeg, K *yedg, V *ywei, K *bufk, W *bufw, const O *xoff, const K *xdeg, const K *xedg, const V *xwei, const K *vcom, const O *coff, const K *cedg, const O *yoff, K CB, K CE) {
  DEFINE_CUDA(t, b, B, G);
  for (K c=CB+b; c<CE; c+=G) {
    __syncthreads();
    size_t EO = yoff[c];
    size_t EN = yoff[c+1] - yoff[c];
    size_t CO = coff[c];
    size_t CN = coff[c+1] - coff[c];
    if (CN==0 || EN < LOUVAIN_DEGREE_THREAD) continue;  // Skip empty communities, or those with low total degree
    size_t H = nextPow2Cud(EN) - 1;
    size_t T = nextPow2Cud(H)  - 1;
    K *hk = bufk + 2*EO;
    W *hv = bufw + 2*EO;
    // Get edges from community c into hashtable.
    __syncthreads();
    hashtableClearCudW(hk, hv, H, t, B);
    __syncthreads();
    for (size_t i = 0; i < CN; ++i) {
      K u = cedg[CO+i];
      louvainScanCommunitiesCudU<true, true, HTYPE>(hk, hv, H, T, xoff, xdeg, xedg, xwei, u, vcom, t, B);
    }
    // Store edges from hashtable into aggregated graph.
    __syncthreads();
    for (size_t i=t; i<H; i+=B) {
      if (hk[i] == K()) continue;
      K d = hk[i] - 1;
      K n = atomicAdd(&ydeg[c], K(1));
      O j = yoff[c] + n;
      yedg[j] = d;
      ywei[j] = V(hv[i]);
    }
  }
}


/**
 * Aggregate outgoing edges of each community, using block-per-community approach.
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
 * @tparam BLIM maximum number of threads per block
 * @param ydeg degrees of aggregated graph (updated)
 * @param yedg edge keys of aggregated graph (updated)
 * @param ywei edge values of aggregated graph (updated)
 * @param bufk buffer to store keys of hashtable (scratch)
 * @param bufw buffer to store values of hashtable (scratch)
 * @param xoff offsets of input graph
 * @param xdeg degrees of input graph
 * @param xedg edge keys of input graph
 * @param xwei edge values of input graph
 * @param vcom community each vertex belongs to
 * @param coff offsets of vertices in each community
 * @param cedg vertices in each community
 * @param yoff offsets of aggregated graph
 * @param CB begin community (inclusive)
 * @param CE end community (exclusive)
 */
template <int HTYPE=3, int BLIM=LOUVAIN_DEGREE_THREAD, class O, class K, class V, class W>
inline void louvainAggregateEdgesBlockCuU(K *ydeg, K *yedg, V *ywei, K *bufk, W *bufw, const O *xoff, const K *xdeg, const K *xedg, const V *xwei, const K *vcom, const O *coff, const K *cedg, const O *yoff, K CB, K CE) {
  const int B = blockSizeCu<true>(CE-CB, 4 * LOUVAIN_DEGREE_BLOCK_SHARED);
  const int G = gridSizeCu <true>(CE-CB, B, GRID_LIMIT_MAP_CUDA);
  louvainAggregateEdgesBlockCukU<HTYPE, BLIM><<<G, B>>>(ydeg, yedg, ywei, bufk, bufw, xoff, xdeg, xedg, xwei, vcom, coff, cedg, yoff, CB, CE);
}


/**
 * Louvain algorithm's community aggregation phase.
 * @param yoff offsets of aggregated graph (output)
 * @param ydeg degrees of aggregated graph (output)
 * @param yedg edge keys of aggregated graph (output)
 * @param ywei edge values of aggregated graph (output)
 * @param bufo buffer for exclusive scan of size |V| (scratch)
 * @param bufk buffer to store keys of hashtable (scratch)
 * @param bufw buffer to store values of hashtable (scratch)
 * @param xoff offsets of input graph
 * @param xdeg degrees of input graph
 * @param xedg edge keys of input graph
 * @param xwei edge values of input graph
 * @param vcom community each vertex belongs to
 * @param coff offsets of vertices in each community
 * @param cedg vertices in each community
 * @param N number of vertices in input graph
 * @param C number of communities
 * @param SCAN size of buffer for exclusive scan
 */
template <class O, class K, class V, class W>
inline void louvainAggregateCuW(O *yoff, K *ydeg, K *yedg, V *ywei, O *bufo, K *bufk, W *bufw, const O *xoff, const K *xdeg, const K *xedg, const V *xwei, const K *vcom, const O *coff, const K *cedg, K N, K C, size_t SCAN) {
  fillValueCuW(yoff, C+1, O());
  fillValueCuW(ydeg, C, K());
  louvainCommunityTotalDegreeCuU(yoff, xdeg, vcom, K(), N);
  exclusiveScanCubW(yoff, bufo, yoff, C+1, SCAN);
  louvainAggregateEdgesThreadCuU(ydeg, yedg, ywei, bufk, bufw, xoff, xdeg, xedg, xwei, vcom, coff, cedg, yoff, K(), C);
  louvainAggregateEdgesBlockCuU (ydeg, yedg, ywei, bufk, bufw, xoff, xdeg, xedg, xwei, vcom, coff, cedg, yoff, K(), C);
}
#pragma endregion




#pragma region PARTITION
/**
 * Partition vertices into low-degree and high-degree sets.
 * @param ks vertex keys (updated)
 * @param x input graph
 * @returns number of low-degree vertices
 */
template <class G, class K>
inline size_t louvainPartitionVerticesCudaU(vector<K>& ks, const G& x) {
  K SWITCH_DEGREE = 64;  // Switch to block-per-vertex approach if degree >= SWITCH_DEGREE
  K SWITCH_LIMIT  = 64;  // Avoid switching if number of vertices < SWITCH_LIMIT
  size_t N = ks.size();
  auto  kb = ks.begin(), ke = ks.end();
  auto  ft = [&](K v) { return x.degree(v) < SWITCH_DEGREE; };
  partition(kb, ke, ft);
  size_t n = count_if(kb, ke, ft);
  if (n   < SWITCH_LIMIT) n = 0;
  if (N-n < SWITCH_LIMIT) n = N;
  return n;
}
#pragma endregion




#pragma region ENVIRONMENT SETUP
/**
 * Setup and perform the Louvain algorithm.
 * @param x input graph
 * @param o louvain options
 * @param fi initializing community membership and total vertex/community weights (vcom, vtot, ctot)
 * @param fm marking affected vertices (vaff)
 * @returns louvain result
 */
template <bool DYNAMIC=false, class FLAG=char, class G, class FI, class FM>
inline auto louvainInvokeCuda(const G& x, const LouvainOptions& o, FI fi, FM fm) {
  using K = typename G::key_type;
  using V = typename G::edge_value_type;
  using W = LOUVAIN_WEIGHT_TYPE;
  using F = FLAG;
  using O = uint32_t;
  // Get graph properties.
  size_t X = x.size();
  size_t S = x.span();
  size_t N = x.order();
  double M = edgeWeightOmp(x) / 2;
  size_t SCAN = max(N, size_t(1024));
  // Options.
  double R = coalesce(o.resolution, 1.0);
  int L = coalesce(o.maxIterations, 10), l = 0;
  int P = coalesce(o.maxPasses, 5), p = 0;
  double EDROP = coalesce(o.toleranceDrop, 10.0);
  double EAGGR = coalesce(o.aggregationTolerance, min(max(1 - 0.025 * X / N, 0.1), 0.9));
  // Allocate buffers on host, original and compressed.
  int T = omp_get_max_threads();
  vector<F> vaff(S), vaffc(N); // Affected vertex flag (any pass)
  vector<K> ucom, ucomc(N);    // Community membership (first pass)
  vector<W> utot, utotc(N);    // Total vertex weights (first pass)
  vector<W> ctot, ctotc(N);    // Total community weights (first pass)
  vector<O> xoff(N + 1);       // Offsets of input graph
  vector<K> xdeg(N);           // Degrees of input graph
  vector<K> xedg(X);           // Edge keys of input graph
  vector<V> xwei(X);           // Edge values of input graph
  if (!DYNAMIC) ucom.resize(S);
  if (!DYNAMIC) utot.resize(S);
  if (!DYNAMIC) ctot.resize(S);
  // Allocate buffers on device.
  F *vaffD = nullptr;  // Affected vertex flag
  K *ucomD = nullptr, *vcomD = nullptr;  // Community membership (first, subsequent passes)
  W *vtotD = nullptr;  // Total vertex weights
  W *ctotD = nullptr;  // Total community weights
  O *bufoD = nullptr;  // Buffer for exclusive scan
  K *bufkD = nullptr;  // Buffer for keys of hashtable
  W *bufwD = nullptr;  // Buffer for values of hashtable
  O *xoffD = nullptr, *yoffD = nullptr;  // Offsets of input and aggregated graph
  K *xdegD = nullptr, *ydegD = nullptr;  // Degrees of input and aggregated graph
  K *xedgD = nullptr, *yedgD = nullptr;  // Edge keys of input and aggregated graph
  V *xweiD = nullptr, *yweiD = nullptr;  // Edge values of input and aggregated graph
  O *coffD = nullptr;  // Offsets of vertices in each community
  K *cdegD = nullptr;  // Number of vertices in each community
  K *cedgD = nullptr;  // Vertices in each community
  uint64_cu *ncomD = nullptr;  // Number of communities
  double      *elD = nullptr;  // Delta modularity per iteration
  TRY_CUDA( cudaSetDeviceFlags(cudaDeviceMapHost) );
  TRY_CUDA( cudaMalloc(&vaffD,  N    * sizeof(F)) );
  TRY_CUDA( cudaMalloc(&ucomD,  N    * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&vcomD,  N    * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&vtotD,  N    * sizeof(W)) );
  TRY_CUDA( cudaMalloc(&ctotD,  N    * sizeof(W)) );
  TRY_CUDA( cudaMalloc(&bufoD,  N    * sizeof(O)) );
  TRY_CUDA( cudaMalloc(&bufkD, (2*X) * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&bufwD, (2*X) * sizeof(W)) );
  TRY_CUDA( cudaMalloc(&xoffD, (N+1) * sizeof(O)) );
  TRY_CUDA( cudaMalloc(&xdegD,  N    * sizeof(O)) );
  TRY_CUDA( cudaMalloc(&xedgD,  X    * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&xweiD,  X    * sizeof(V)) );
  TRY_CUDA( cudaMalloc(&yoffD, (N+1) * sizeof(O)) );
  TRY_CUDA( cudaMalloc(&ydegD,  N    * sizeof(O)) );
  TRY_CUDA( cudaMalloc(&yedgD,  X    * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&yweiD,  X    * sizeof(V)) );
  TRY_CUDA( cudaMalloc(&coffD, (N+1) * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&cdegD,  N    * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&cedgD,  N    * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&ncomD,  1    * sizeof(uint64_cu)) );
  TRY_CUDA( cudaMalloc(&elD,    1    * sizeof(double)) );
  // Partition vertices into low-degree and high-degree sets.
  vector<K> ks = vertexKeys(x);
  size_t    NL = louvainPartitionVerticesCudaU(ks, x);
  // Obtain data for CSR.
  csrCreateOffsetsW (xoff, x, ks);
  csrCreateDegreesW (xdeg, x, ks);
  csrCreateEdgeKeysW(xedg, x, ks);
  csrCreateEdgeValuesW(xwei, x, ks);
  // Perform Louvain algorithm on device.
  float tm = 0, ti = 0, tp = 0, tl = 0, ta = 0; // Time spent in different phases
  float t  = measureDurationMarked([&](auto mark) {
    size_t GN = N;
    double E  = coalesce(o.tolerance, 1e-2);
    auto   fc = [&](double el, int l) { return el<=E; };
    // Reset buffers, in case of multiple runs.
    fillValueOmpU(vaff, F());
    fillValueOmpU(ucom, K());
    fillValueOmpU(utot, W());
    fillValueOmpU(ctot, W());
    fillValueCuW(vaffD, N, F());
    fillValueCuW(ucomD, N, K());
    fillValueCuW(vcomD, N, K());
    fillValueCuW(vtotD, N, W());
    fillValueCuW(ctotD, N, W());
    // Copy input graph to device
    TRY_CUDA( cudaMemcpy(xoffD, xoff.data(), (N+1) * sizeof(O), cudaMemcpyHostToDevice) );
    TRY_CUDA( cudaMemcpy(xdegD, xdeg.data(),  N    * sizeof(K), cudaMemcpyHostToDevice) );
    TRY_CUDA( cudaMemcpy(xedgD, xedg.data(),  X    * sizeof(K), cudaMemcpyHostToDevice) );
    TRY_CUDA( cudaMemcpy(xweiD, xwei.data(),  X    * sizeof(V), cudaMemcpyHostToDevice) );
    // Initialize community membership and total vertex/community weights.
    ti += mark([&]() {
      louvainVertexWeightsThreadCuW(vtotD, xoffD, xdegD, xweiD, K(), K(N));
      louvainVertexWeightsBlockCuW (vtotD, xoffD, xdegD, xweiD, K(), K(N));
      louvainInitializeCuW(ucomD, ctotD, vtotD, K(), K(N));
    });
    // Mark affected vertices.
    tm += mark([&]() { fm(vaff); });
    gatherValuesOmpW(vaffc, vaff, ks);
    TRY_CUDA( cudaMemcpy(vaffD, vaffc.data(), N * sizeof(F), cudaMemcpyHostToDevice) );
    // Perform Louvain iterations
    mark([&]() {
      // Start timing first pass.
      auto t0 = timeNow(), t1 = t0;
      // Start local-moving, aggregation phases.
      // NOTE: In each pass, both the input graph and output graphs are CSR.
      for (l=0, p=0; M>0 && P>0;) {
        if (p==1) t1 = timeNow();
        bool isFirst = p==0;
        int m = 0;
        tl += measureDuration([&]() {
          if (isFirst) m = louvainMoveCuU(elD, ucomD, ctotD, vaffD, bufkD, bufwD, xoffD, xdegD, xedgD, xweiD, vtotD, M, R, L, K(N),  K(NL), fc);
          else         m = louvainMoveCuU(elD, vcomD, ctotD, vaffD, bufkD, bufwD, xoffD, xdegD, xedgD, xweiD, vtotD, M, R, L, K(GN), K(GN), fc);
        });
        l += max(m, 1); ++p;
        if (m<=1 || p>=P) break;
        uint64_cu CN = 0;
        fillValueCuW(ncomD, 1,  uint64_cu());
        fillValueCuW(cdegD, GN, K());
        if (isFirst) louvainCommunityExistsCuU(ncomD, cdegD, ucomD, K(), K(N));
        else         louvainCommunityExistsCuU(ncomD, cdegD, vcomD, K(), K(GN));
        TRY_CUDA( cudaMemcpy(&CN, ncomD, sizeof(uint64_cu), cudaMemcpyDeviceToHost) );
        if (double(CN)/GN >= EAGGR) break;
        if (isFirst) louvainRenumberCommunitiesCuU(ucomD, cdegD, bufkD, K(N),  SCAN);
        else         louvainRenumberCommunitiesCuU(vcomD, cdegD, bufkD, K(GN), SCAN);
        if (isFirst) {}
        else         louvainLookupCommunitiesCuU(ucomD, vcomD, K(), K(N));
        ta += measureDuration([&]() {
          if (isFirst) louvainCommunityVerticesCuW(coffD, cdegD, cedgD, bufkD, ucomD, K(N),  K(CN), SCAN);
          else         louvainCommunityVerticesCuW(coffD, cdegD, cedgD, bufkD, vcomD, K(GN), K(CN), SCAN);
          if (isFirst) louvainAggregateCuW(yoffD, ydegD, yedgD, yweiD, bufoD, bufkD, bufwD, xoffD, xdegD, xedgD, xweiD, ucomD, coffD, cedgD, K(N),  K(CN), SCAN);
          else         louvainAggregateCuW(yoffD, ydegD, yedgD, yweiD, bufoD, bufkD, bufwD, xoffD, xdegD, xedgD, xweiD, vcomD, coffD, cedgD, K(GN), K(CN), SCAN);
        });
        fillValueCuW(vtotD, size_t(CN), W());
        fillValueCuW(vaffD, size_t(CN), F(1));
        louvainVertexWeightsThreadCuW(vtotD, yoffD, ydegD, yweiD, K(), K(CN));
        louvainVertexWeightsBlockCuW (vtotD, yoffD, ydegD, yweiD, K(), K(CN));
        louvainInitializeCuW(vcomD, ctotD, vtotD, K(), K(CN));
        E /= EDROP;
        swap(xoffD, yoffD);
        swap(xdegD, ydegD);
        swap(xedgD, yedgD);
        swap(xweiD, yweiD);
        GN = CN;
      }
      if (p<=1) {}
      else louvainLookupCommunitiesCuU(ucomD, vcomD, K(), K(N));
      if (p<=1) t1 = timeNow();
      tp += duration(t0, t1);
    }); }, o.repeat);
  TRY_CUDA( cudaMemcpy(ucomc.data(), ucomD, N * sizeof(K), cudaMemcpyDeviceToHost) );
  scatterValuesOmpW(ucom, ucomc, ks);
  // Free device memory.
  TRY_CUDA( cudaFree(vaffD) );
  TRY_CUDA( cudaFree(ucomD) );
  TRY_CUDA( cudaFree(vcomD) );
  TRY_CUDA( cudaFree(vtotD) );
  TRY_CUDA( cudaFree(ctotD) );
  TRY_CUDA( cudaFree(bufoD) );
  TRY_CUDA( cudaFree(bufkD) );
  TRY_CUDA( cudaFree(bufwD) );
  TRY_CUDA( cudaFree(xoffD) );
  TRY_CUDA( cudaFree(xdegD) );
  TRY_CUDA( cudaFree(xedgD) );
  TRY_CUDA( cudaFree(xweiD) );
  TRY_CUDA( cudaFree(yoffD) );
  TRY_CUDA( cudaFree(ydegD) );
  TRY_CUDA( cudaFree(yedgD) );
  TRY_CUDA( cudaFree(yweiD) );
  TRY_CUDA( cudaFree(coffD) );
  TRY_CUDA( cudaFree(cdegD) );
  TRY_CUDA( cudaFree(cedgD) );
  TRY_CUDA( cudaFree(ncomD) );
  TRY_CUDA( cudaFree(elD) );
  return LouvainResult<K, W>(ucom, utot, ctot, l, p, t, tm/o.repeat, ti/o.repeat, tp/o.repeat, tl/o.repeat, ta/o.repeat, countValueOmp(vaff, F(1)));
}
#pragma endregion




#pragma region STATIC
/**
 * Obtain the community membership of each vertex with Static Louvain.
 * @tparam FLAG flag type for tracking affected vertices
 * @param x input graph
 * @param o louvain options
 * @returns louvain result
 */
template <class FLAG=char, class G>
inline auto louvainStaticCuda(const G& x, const LouvainOptions& o={}) {
  using K = typename G::key_type;
  auto fi = [&](auto& vcom, auto& vtot, auto& ctot) {
    louvainVertexWeightsW(vtot, x);
    louvainInitializeW(vcom, ctot, x, vtot);
  };
  auto fm = [](auto& vaff) {
    fillValueOmpU(vaff, FLAG(1));
  };
  return louvainInvokeCuda<false, FLAG>(x, o, fi, fm);
}
#pragma endregion
#pragma endregion
