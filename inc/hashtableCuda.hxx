#pragma once
#include <cstdint>
#include "_main.hxx"




/**
 * A hashtable based on double hashing for collision resolution.
 * @details
 * - Supports accumulation of values.
 * - Only non-zero keys are supported.
 * - Performs collision resolution for keys (must have enough capacity).
 * - Capacity of the hashtable is preferably a prime number.
 * - Uses a secondary prime for double hashing.
 */
#pragma region METHODS
/**
 * Accumulate value to an entry in hashtable [device function].
 * @tparam BLOCK called from a thread block?
 * @param hk hashtable keys (updated)
 * @param hv hashtable values (updated)
 * @param i index to accumulate at
 * @param k key to accumulate
 * @param v value to accumulate
 * @returns whether value was accumulated
 */
template <bool BLOCK=false, class K, class V>
inline bool __device__ hashtableAccumulateAtCudU(K *hk, V *hv, size_t i, K k, V v) {
  if (!BLOCK) {
    if (hk[i]!=k && hk[i]!=K()) return false;
    if (hk[i]==K()) hk[i] = k;
    hv[i] += v;
  }
  else {
    // if (hk[i]!=k && (hk[i]!=K() || atomicCAS(&hk[i], K(), k)!=K())) return false;
    if (hk[i]!=k && hk[i]!=K()) return false;
    K old = atomicCAS(&hk[i], K(), k);
    if (old!=K() && old!=k) return false;
    atomicAdd(&hv[i], v);
  }
  return true;
}


/**
 * Accumulate value to an entry in hashtable [device function].
 * @tparam BLOCK called from a thread block?
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
 * @param hk hashtable keys (updated)
 * @param hv hashtable values (updated)
 * @param H capacity of hashtable (prime)
 * @param T secondary prime (>H)
 * @param k key to accumulate
 * @param v value to accumulate
 * @returns whether value was accumulated
 */
template <bool BLOCK=false, int HTYPE=3, class K, class V>
inline bool __device__ hashtableAccumulateCudU(K *hk, V * hv, size_t H, size_t T, K k, V v) {
  size_t i = k, di = 1;
  for (size_t t=0; t<H; ++t) {
    if (hashtableAccumulateAtCudU<BLOCK>(hk, hv, i % H, k, v)) return true;
    switch (HTYPE) {
      case 0: ++i; break;
      case 1: i += di; di *= 2; break;
      case 2: i += k % T; break;
      case 3: i += di; di = di*2 + (k % T); break;
    }
  }
  return false;
}


/**
 * Obtain value of an entry in hashtable [device function].
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
 * @param hk hashtable keys
 * @param hv hashtable values
 * @param H capacity of hashtable (prime)
 * @param T secondary prime (>H)
 * @param k entry key
 * @returns entry value
 */
template <int HTYPE=3, class K, class V>
inline V __device__ hashtableGetCud(const K *hk, const V *hv, size_t H, size_t T, K k) {
  size_t i = k, di = 1;
  for (size_t t=0; t<H; ++t) {
    if (hk[i % H]==k) return hv[i % H];
    switch (HTYPE) {
      case 0: ++i; break;
      case 1: i += di; di *= 2; break;
      case 2: i += k % T; break;
      case 3: i += di; di = di*2 + (k % T); break;
    }
  }
  return V();
}


/**
 * Clear entries in hashtable [device function].
 * @param hk hashtable keys (output)
 * @param hv hashtable values (output)
 * @param H capacity of hashtable (prime)
 * @param i start index
 * @param DI index stride
 */
template <class K, class V>
inline void __device__ hashtableClearCudW(K *hk, V *hv, size_t H, size_t i, size_t DI) {
  for (; i<H; i+=DI) {
    hk[i] = K();
    hv[i] = V();
  }
}


/**
 * Find entry in hashtable with maximum value, from a thread [device function].
 * @param hk hashtable keys (updated, entry i is max)
 * @param hv hashtable values (updated, entry i is max)
 * @param H capacity of hashtable (prime)
 * @param i start index
 * @param DI index stride
 */
template <class K, class V>
inline void __device__ hashtableMaxThreadCudU(K *hk, V *hv, size_t H, size_t i, size_t DI) {
  for (size_t j=i+DI; j<H; j+=DI) {
    if (hv[j] > hv[i]) {
      hk[i] = hk[j];
      hv[i] = hv[j];
    }
  }
}


/**
 * Find entry in block-sized hashtable with maximum value [device function].
 * @param hk hashtable keys (updated, entry 0 is max)
 * @param hv hashtable values (updated, entry 0 is max)
 * @param H capacity of hashtable (prime)
 * @param i start index
 * @param DI index stride
 */
template <class K, class V>
inline void __device__ hashtableMaxBlockReduceCudU(K *hk, V *hv, size_t H, size_t i) {
  for (; H>1;) {
    size_t DH = (H+1)/2;
    if (i<H/2 && hv[i+DH] > hv[i]) {
      hk[i] = hk[i+DH];
      hv[i] = hv[i+DH];
    }
    __syncthreads();
    H = DH;
  }
}


/**
 * Find entry in hashtable with maximum value [device function].
 * @tparam BLOCK called from a thread block?
 * @param hk hashtable keys (updated, entry 0 is max)
 * @param hv hashtable values (updated, entry 0 is max)
 * @param H capacity of hashtable (prime)
 * @param i start index
 * @param DI index stride
 */
template <bool BLOCK=false, class K, class V>
inline void __device__ hashtableMaxCudU(K *hk, V *hv, size_t H, size_t i, size_t DI) {
  hashtableMaxThreadCudU(hk, hv, H, i, DI);
  if (BLOCK) {
    // Wait for all threads within the block to finish.
    __syncthreads();
    // Reduce keys and values in hashtable to obtain key with maximum value at 0th entry.
    hashtableMaxBlockReduceCudU(hk, hv, H, i);
  }
}
#pragma endregion
