// core.cuh -- what every task type of the static megakernel shares, and the
// kernel's start and end. The generated layer (static_schedule.py) defines
// first the kernel-wide sizes (config.cuh: STATIC_TOKENS, STATIC_GPUS,
// STATIC_SMS) and
//   STATIC_EXCHANGE_SET       bytes of one set of the cross-GPU exchange region
//   (two sets: launch n uses set n % 2) STATIC_EXCHANGE_HELLO     the start
//   barrier's slots in a set (one 16-B slot per GPU) STATIC_EXCHANGE_OFFSETS
//   per buffer slot: its offset in a set (an exchange buffer; else 0)
//   STATIC_EXCHANGE_BYTES     per buffer slot: its bytes in a set (an exchange
//   buffer; else 0)
// then includes this file and the task types' files (tasks/<name>.cuh), each
// with its body and its task function run_<name>. A task function's template
// arguments: its node (StaticNode: grid, first counter, output buffer slot),
// its params (StaticParams), its own buffer slots (StaticBufSlots: indices
// into G::buf of the scratch buffers, graph tensors and other buffers its layer
// declared) and map slots (StaticMapSlots: indices into Maps::m), each in the
// layer's order, then per input the node that writes it (StaticNode with that
// node's params as v[]; StaticNode<0, 0, 0>: a graph input).
#pragma once
#include "config.cuh"
#include "runtime.cuh"
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>

#ifndef STATIC_EXCHANGE_SET
#error                                                                         \
    "core.cuh: the generated layer defines STATIC_EXCHANGE_SET, STATIC_EXCHANGE_HELLO, STATIC_EXCHANGE_OFFSETS, STATIC_EXCHANGE_BYTES"
#endif

namespace static_mk {

// ---- the exchange region's layout (from the layers) ----
constexpr size_t EXCHANGE_SET = STATIC_EXCHANGE_SET,
                 EXCHANGE_HELLO = STATIC_EXCHANGE_HELLO;
constexpr size_t exchange_offset[MAX_BUFS] = STATIC_EXCHANGE_OFFSETS;
constexpr size_t exchange_bytes[MAX_BUFS] = STATIC_EXCHANGE_BYTES;

// ---- the tensor maps: TMA descriptors by slot, built by the host code from
// the layers' declarations ----
struct Maps {
  CUtensorMap m[MAX_MAPS];
};

#ifdef STATIC_RESET_IN_KERNEL
// buffers reset at the start of each launch (STATIC_RESET_IN_KERNEL): byte
// value v[i] over n[i] bytes at p[i] (n a multiple of 16).
// STATIC_NO_START_BARRIER: each has two sets stride[i] bytes apart, and a
// launch resets the set it does not use, at its end (kernel_end)
constexpr int MAX_RESETS = 24;
struct ResetList {
  unsigned char *p[MAX_RESETS];
  unsigned long long n[MAX_RESETS];
  unsigned long long stride[MAX_RESETS];
  unsigned v[MAX_RESETS];
  int cnt;
};
#ifdef STATIC_L2_PREFETCH
// the layer's weights kernel_begin prefetches into L2 (StaticMegakernel
// l2_prefetch_weights, the host's "pf." list): n[i] bytes at p[i] (multiples of
// 16), in the order the layer reads them, as nb batches of PF_BATCH bytes of
// the ranges laid end to end; ctr[2]: the batches claimed so far by the
// launches of each parity
constexpr int MAX_PF = 8;
constexpr unsigned long long PF_BATCH = 1ull << 20, PF_CHUNK = 64ull << 10;
struct PrefetchList {
  unsigned char const *p[MAX_PF];
  unsigned long long n[MAX_PF];
  int cnt;
  unsigned nb;
  unsigned long long *ctr;
};
#endif
#endif
// STATIC_AR_TWOSHOT: the all-reduce of allreduce_send -> hc_post /
// hc_post_mix (GLM-5.3-Flash's layer tails; and sum_gpus) as a two-shot by
// token. The generated layer.cu defines it for a graph of T 64 on 4 GPUs
// with an allreduce_send and without sum_rmsnorm (static_schedule.ar_twoshot,
// header_code); every other
// layer has the one-shot (every GPU's partial multicast to every GPU, each GPU
// adds all slots of every vector). Example (4 GPUs, T 64, H 4096; GPU 2 owns
// tokens 32..47, AR_TOWN = T / GPUS = 16):
//   allreduce_send on GPU 1, token 37: its bf16 partial row -> ONE store per
//     16 B into GPU 2's copy (G::peer[2]), slot 1, row 5 of the owner landing
//     [GPUS][AR_TOWN][H] at the send buffer's offset 0 (ar_twoshot_send)
//   hc_post_mix task s on GPU 2, the lane of token 37, vector c: 1. (GPU 2 owns
//     token 37) the 4 slots added in GPU order (fp32, allreduce_land_bf16, the
//     same code as the one-shot), bf16(sum) multicast into the gathered rows
//     [T][H] at offset GPUS AR_TOWN H 2 of the same buffer on every GPU, the 4
//     slots re-armed; 2. (every GPU, every token) the vector polled from the
//     gathered rows (one slot), re-armed (ar_twoshot_get). hc_post task (q, t),
//     sum_gpus task (q, t): the same, step 1 on GPU t / AR_TOWN
// The readers round the sum to bf16 before using it (the one-shot too): R',
// the mixes and the sums are bit-identical. Task x of a reader waits in step 2
// only for step 1 of task x on the owner GPUs (compile_plan checks the GPUs'
// lists together: static_schedule.check_twoshot_deadlock_free). sum_rmsnorm
// reads only the one-shot layout (no STATIC_AR_TWOSHOT in its graphs).
#ifdef STATIC_AR_TWOSHOT
constexpr bool AR_TWOSHOT = true;
#else
constexpr bool AR_TWOSHOT = false;
#endif
static_assert(!AR_TWOSHOT || (T == 64 && GPUS == 4),
              "STATIC_AR_TWOSHOT: the layers of T 64 on 4 GPUs "
              "(static_schedule.ar_twoshot)");

#ifdef STATIC_NO_START_BARRIER
#ifndef STATIC_RESET_IN_KERNEL
#error                                                                         \
    "STATIC_NO_START_BARRIER is a serving mode: it needs STATIC_RESET_IN_KERNEL"
#endif
#endif
#ifdef STATIC_X_FLAG
#ifndef STATIC_NO_START_BARRIER
#error "STATIC_X_FLAG (H1) is a serving mode: it needs STATIC_NO_START_BARRIER"
#endif
constexpr int XFLAG_N = 32; // the post's tasks (hc_post_combine: 4 parts x T)
// static_host_rebind (host.cuh): "@xflag_out" this kernel's post words
// ([XFLAG_N] uint32, zeroed, owned by the caller), "@xflag_in" the kernel
// before's (0: the classic wait at the start). G starts zeroed (GpuState())
#define STATIC_X_FLAG_REBIND(gg, name, p)                                      \
  if ((name) == "@xflag_out") {                                                \
    (gg).xflag_out = reinterpret_cast<uint32_t *>(p);                          \
    return true;                                                               \
  }                                                                            \
  if ((name) == "@xflag_in") {                                                 \
    (gg).xflag_in = reinterpret_cast<uint32_t const *>(p);                     \
    return true;                                                               \
  }
#else
#define STATIC_X_FLAG_REBIND(gg, name, p)
#endif

// ---- the layer state (the kernel's parameter g) ----
struct G {
  void *buf[MAX_BUFS]; // by slot: the nodes' outputs (an exchange buffer: this
                       // GPU's copy, set 0), scratch buffers, graph tensors
                       // read by pointer
  uint32_t *cnt;       // counters (NCNT): the nodes' lines
  int rank;            // this GPU (of GPUS)
  unsigned gen; // launch number + 1; the start barrier waits until every GPU's
                // hello slot holds it
#ifdef STATIC_RESET_IN_KERNEL
  unsigned *gen_ptr; // the launch number in device memory (gen unused): every
                     // SM reads it + 1 at the start, SM 0 stores it back once
                     // all SMs have (a CUDA graph replay gets a new one)
  unsigned *arrive;  // SMs done with the resets, over all launches (only grows)
  ResetList resets;  // the buffers kernel_begin resets (the host's list)
#ifdef STATIC_L2_PREFETCH
  PrefetchList pf; // the weights kernel_begin prefetches into L2
#endif
#endif
#ifdef STATIC_NO_START_BARRIER
  unsigned *gen_sm; // [NSM] the launch number per SM: each SM reads its own + 1
                    // and stores it back (no SM waits for another)
  unsigned long long par_bytes[MAX_BUFS]; // a buffer slot with two sets: the
                                          // second's offset (else 0); the
                                          // counters have two sets of NCNT
#endif
  unsigned char *mc, *rv; // the exchange region: multicast address (GPUS > 1),
                          // this GPU's copy (2 sets)
#ifdef STATIC_X_FLAG
  // H1 (docs/h1_design.md): x with its own flag instead of the wait for the
  // kernel before's grid. xflag_out: [XFLAG_N] this kernel's post tasks each
  // store this launch's number there after x and the streams (release); never
  // reset. xflag_in: the kernel before's xflag_out (its post writes this
  // kernel's x), or nullptr: the classic path (the wait at the start). Both
  // bound by the host (static_host_rebind "@xflag_out" / "@xflag_in")
  uint32_t *xflag_out;
  uint32_t const *xflag_in;
#endif
  long long *stamps; // [NSM][NSTAMP] time stamps (globaltimer ns) read by the
                     // host (config.cuh STAMP_*)
  long long *start_barrier; // [1] written by every SM once the start barrier is
                            // passed: the last SM's time remains
#ifdef STATIC_AR_TWOSHOT
  unsigned char *peer[GPUS]; // (STATIC_AR_TWOSHOT) every GPU's copy of the
                             // exchange region (both sets), mapped in this
                             // process (peer[rank] = rv)
#endif
};

// STATIC_NO_START_BARRIER: the kernel's second G (the host's copy of g with
// every two-set buffer and the counters at their second set); the generated
// kernel picks one, gp, once, so the tasks' addresses stay kernel parameters
#ifdef STATIC_NO_START_BARRIER
#define STATIC_G1_PARAM static_mk::G g_par1,
#define STATIC_G1_ARG(x) , static_host::g1_of(x)
#else
#define STATIC_G1_PARAM
#define STATIC_G1_ARG(x)
#endif

// the exchange region's set of this launch: its launch number's parity, set at
// the kernel's start
__shared__ unsigned dbuf_par;
#ifdef STATIC_X_FLAG
// H1, in shared memory (no register live across the task loop): this launch's
// number (the value the post stores), and the CTA's state: bit 0
// griddepcontrol.wait done (deferred to the first task that is not pre-wait
// safe, at the latest kernel_end), bit 1 x's flag seen (later polls return at
// once). Thread 0 changes the state between two barriers: every thread reads
// the same value at every check
__shared__ unsigned x_gen_s, x_state_s;
#endif
__device__ __forceinline__ size_t rg_set() {
  return (size_t)dbuf_par * EXCHANGE_SET;
}
// buffer `slot` (G::buf) as a P pointer
template <class P>
__device__ __forceinline__ P *buf_at(G const &g, int slot) {
  return reinterpret_cast<P *>(g.buf[slot]);
}
// the nodes' counters
__device__ __forceinline__ uint32_t *cnt_now(G const &g) {
  return g.cnt;
}
// a multicast store into this launch's set of every GPU's exchange region (off:
// an offset in the set)
__device__ __forceinline__ void push16(G const &g, size_t off, uint4 v) {
  push16_mc(g.mc, g.rv, off + rg_set(), v);
}

#ifdef STATIC_AR_TWOSHOT
// (STATIC_AR_TWOSHOT) an allreduce_send buffer [GPUS][T][W] bf16, used as
//   [0, ar_g_off)          the owner landing: slot r (the sending GPU), row j
//                          of the owner's tokens (token owner * AR_TOWN + j) at
//                          r * ar_rank + j * W * 2 (each GPU's copy: its own
//                          tokens only), written by unicast stores
//   [ar_g_off, + T W 2)    the gathered rows [T][W] bf16 (bf16 of the sums over
//                          the GPUs), multicast by the owners (every GPU's
//                          copy: all tokens)
// both 0xFF-filled (the buffer's fill) and re-armed by their readers
constexpr int AR_TOWN = T / GPUS; // tokens per owner GPU (16 at T 64, 4 GPUs)
template <int W>
__host__ __device__ constexpr size_t ar_rank() {
  return (size_t)AR_TOWN * W * 2;
} // one sending GPU's slot in the owner landing
template <int W>
__host__ __device__ constexpr size_t ar_g_off() {
  return (size_t)GPUS * ar_rank<W>();
} // the gathered rows' offset in the buffer
// 16 B at offset `off` of this launch's set of GPU h's exchange region (one
// NVLink store; h = rank: the local copy). G::peer by constant indices (a
// dynamic index into the kernel parameter would copy G to local memory)
__device__ __forceinline__ void ar_twoshot_put16(G const &g,
                                                 int h,
                                                 size_t off,
                                                 uint4 v) {
  unsigned char *dst = g.peer[0];
#pragma unroll
  for (int q = 1; q < GPUS; q++) {
    dst = h == q ? g.peer[q] : dst;
  }
  asm volatile(
      "st.relaxed.sys.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(
          dst + off + rg_set()),
      "r"(v.x),
      "r"(v.y),
      "r"(v.z),
      "r"(v.w)
      : "memory");
}
// allreduce_send's store of this GPU's bf16 partial of token t, vector c (of
// W / 8) into the buffer at OFF: slot g.rank of the owner landing of token t's
// owner (t / AR_TOWN)
template <int W, size_t OFF>
__device__ __forceinline__ void
    ar_twoshot_send(G const &g, int t, int c, uint4 v) {
  int const h = t / AR_TOWN, j = t - h * AR_TOWN;
  ar_twoshot_put16(g,
                   h,
                   OFF + (size_t)g.rank * ar_rank<W>() +
                       ((size_t)j * (W / 8) + c) * 16,
                   v);
}
#else
// declared only: the calls are in discarded `if constexpr (AR_TWOSHOT)`
// branches
template <int W, size_t OFF>
__device__ void ar_twoshot_send(G const &g, int t, int c, uint4 v);
#endif

// params[I] of a node (StaticParams<P...>::v holds P..., 0)
template <class PARAMS, int I>
__host__ __device__ constexpr int param_at() {
  static_assert(sizeof(PARAMS::v) / sizeof(int) > I + 1,
                "the node has fewer params");
  return PARAMS::v[I];
}
// params[I] of a node, or DEF when the node has fewer params
template <class PARAMS, int I, int DEF>
__host__ __device__ constexpr int param_or() {
  if constexpr (sizeof(PARAMS::v) / sizeof(int) > I + 1) {
    return PARAMS::v[I];
  } else {
    return DEF;
  }
}

// params[I] of a node, 0 when it has fewer
template <class PARAMS, int I>
__host__ __device__ constexpr int param_or0() {
  if constexpr (sizeof(PARAMS::v) / sizeof(int) > I + 1) {
    return PARAMS::v[I];
  } else {
    return 0;
  }
}

// entry I of a node's buffer slots or map slots (StaticBufSlots<S...>::v,
// StaticMapSlots<S...>::v hold S..., -1)
template <class SLOTS, int I>
__host__ __device__ constexpr int slot_at() {
  static_assert(sizeof(SLOTS::v) / sizeof(int) > I + 1,
                "the node has fewer slots");
  return SLOTS::v[I];
}
// entry I of a node's buffer slots or map slots, or DEF when it has fewer
template <class SLOTS, int I, int DEF>
__host__ __device__ constexpr int slot_or() {
  if constexpr (sizeof(SLOTS::v) / sizeof(int) > I + 1) {
    return SLOTS::v[I];
  } else {
    return DEF;
  }
}

// how a gemm_tile node's K parts are combined in its output buffer out (N = the
// weight's rows; task K part `split`, thread row `row`, v[t] = token t). The
// node's consumer reads the buffer in that layout (its producer's params[0]).
enum Combine {
  COMBINE_SLOTS = 0, // fp32 out[split][t][N]: out[split][t][row0 + row] = v[t];
                     // one 0xFF-prefilled slot per K part, the reader polls
                     // them and adds the parts in a fixed order
  COMBINE_ADD =
      1, // int64 out[t][N]: += v[t] in fixed point (runtime.cuh red_add_fx: the
         // same sum in any order), then the node's counter [row0 / 128] += 1
         // per task; the reader waits for all K parts of its 128-row blocks
  COMBINE_STORE =
      2, // fp32 out[t][N] = v[t], plain stores (one K part), then the node's
         // counter [0] += 1 per task; the reader waits for all tasks
  COMBINE_STORE_BF16 = 3, // bf16 out[t][N] = bf16(v[t]), else as COMBINE_STORE
                          // (a layer's final bf16 output, read after the
                          // kernel)
};

// 8 fp32 values -> one 16-B bf16 vector
__device__ __forceinline__ uint4 pack_bf16x8(float4 const &p0,
                                             float4 const &p1) {
  __nv_bfloat162 const h0 = __floats2bfloat162_rn(p0.x, p0.y),
                       h1 = __floats2bfloat162_rn(p0.z, p0.w);
  __nv_bfloat162 const h2 = __floats2bfloat162_rn(p1.x, p1.y),
                       h3 = __floats2bfloat162_rn(p1.z, p1.w);
  return make_uint4(*reinterpret_cast<uint32_t const *>(&h0),
                    *reinterpret_cast<uint32_t const *>(&h1),
                    *reinterpret_cast<uint32_t const *>(&h2),
                    *reinterpret_cast<uint32_t const *>(&h3));
}

// the adding side of an all-reduce (allreduce_send): thread v polls 16-B bf16
// vector v of `row` from every GPU (slot r at row + r * RANK, 0xFF-prefilled)
// and adds them in fp32 in GPU order -> the 8 sums. REARM: re-armed by
// allreduce_rearm after the caller used them.
// W16W (STATIC_AR_TWOSHOT's owner landing): the slots come by unicast
// NVLink stores, every 4-B word must be written (valid16w)
template <size_t RANK, bool W16W = false>
__device__ __forceinline__ void
    allreduce_land_bf16(G const &g, unsigned char const *row, int v, float *s) {
  uint4 u[GPUS];
  bool ok[GPUS] = {};
  bool all;
  do {
    all = true;
#pragma unroll
    for (int r = 0; r < GPUS; r++) {
      if (!ok[r]) {
        u[r] = ld16_poll(row + (size_t)r * RANK + (size_t)v * 16);
      }
    }
#pragma unroll
    for (int r = 0; r < GPUS; r++) {
      if (!ok[r]) {
        ok[r] = W16W ? valid16w(u[r]) : valid16(u[r]);
      }
      all = all && ok[r];
    }
  } while (!all);
#pragma unroll
  for (int k = 0; k < 8; k++) {
    s[k] = 0.f;
  }
#pragma unroll
  for (int r = 0; r < GPUS;
       r++) { // a fixed bound (unrolled), so u stays in registers
    uint32_t const wd[4] = {u[r].x, u[r].y, u[r].z, u[r].w};
#pragma unroll
    for (int k = 0; k < 4; k++) {
      float2 const f =
          __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const *>(&wd[k]));
      s[2 * k] += f.x;
      s[2 * k + 1] += f.y;
    }
  }
}
template <size_t RANK>
__device__ __forceinline__ void
    allreduce_rearm(G const &g, unsigned char const *row, int v) {
  for (int r = 0; r < GPUS; r++) {
    rearm16(row + (size_t)r * RANK + (size_t)v * 16);
  }
}
#ifdef STATIC_AR_TWOSHOT
// (STATIC_AR_TWOSHOT) a reader's all-reduced vector c (of W / 8) of token
// t from the allreduce_send buffer at SOFF (the same task on every GPU reads
// it): 1. on token t's owner (t / AR_TOWN), the GPUS landing slots added in
// fp32 in GPU order (allreduce_land_bf16, the one-shot's code), bf16(sum)
// multicast into the gathered rows of every GPU, the slots re-armed; 2. on
// every GPU, the gathered vector polled (one slot), re-armed. Returns the 8
// bf16 sums: the one-shot's bf16(sum). Step 2 waits for step 1 of the same
// task on the owner GPU
template <size_t SOFF, int W>
__device__ __forceinline__ uint4 ar_twoshot_get(G const &g, int t, int c) {
  constexpr size_t AR = ar_rank<W>(), GOFF = ar_g_off<W>();
  static_assert(GOFF + (size_t)T * W * 2 <= (size_t)GPUS * T * W * 2,
                "STATIC_AR_TWOSHOT: the landing and the gathered rows fit "
                "in the allreduce_send buffer");
  unsigned char const *const sl = g.rv + SOFF + rg_set();
  size_t const gv = GOFF + ((size_t)t * (W / 8) + c) * 16;
  if (t / AR_TOWN == g.rank) {
    int const lv = (t % AR_TOWN) * (W / 8) + c; // the vector in each slot
    float s[8];
    allreduce_land_bf16<AR, true>(g, sl, lv, s);
    push16(g,
           SOFF + gv,
           pack_bf16x8(make_float4(s[0], s[1], s[2], s[3]),
                       make_float4(s[4], s[5], s[6], s[7])));
    if constexpr (REARM) { // the one reader of these slots
      allreduce_rearm<AR>(g, sl, lv);
    }
  }
  uint4 u = ld16_poll(sl + gv);
  while (!valid16(u)) {
    u = ld16_poll(sl + gv);
  }
  if constexpr (REARM) { // this GPU's one reader of this gathered vector
    rearm16(sl + gv);
  }
  return u;
}
// the same as 8 fp32 values (each the one-shot's bf16(sum), as hc_post and
// hc_post_mix use it)
template <size_t SOFF, int W>
__device__ __forceinline__ void
    ar_twoshot_get_f32(G const &g, int t, int c, float *s) {
  uint4 const u = ar_twoshot_get<SOFF, W>(g, t, c);
  __nv_bfloat16 const *const b = reinterpret_cast<__nv_bfloat16 const *>(&u);
#pragma unroll
  for (int k = 0; k < 8; k++) {
    s[k] = __bfloat162float(b[k]);
  }
}
#endif

// the kernel's local variables (every function is inlined, so they stay in
// registers)
struct KernelLocals {
  uint32_t
      base; // 1024-aligned dynamic shared memory start (shared-window address)
  char *sm; // the same, as a pointer
  int sm_id;
  uint32_t tb;      // TMEM base
  int gl, gi0, gi1; // ring stage counters: loader, issuer 0, issuer 1 (they
                    // advance by the same stages)
  int wt;           // accumulator tiles so far (all roles)
  bool ring_relaid; // a task re-laid the ring's shared memory (moe_experts'
                    // phase switch): a GEMM after it starts with fresh ring
                    // barriers

#ifdef STATIC_TIMING_BUILD
  unsigned long long *stage_stamps =
      nullptr; // this list entry's 2 GEMM stage stamps (set by the generated
               // loop)
#endif
};

#if defined(STATIC_RESET_IN_KERNEL) && defined(STATIC_L2_PREFETCH)
// batch b of the prefetch list into L2 (cp.async.bulk.prefetch.L2: fire and
// forget, the lines land after the CTA moved on), PF_CHUNK bytes an
// instruction; false when b is past the list
__device__ __forceinline__ bool pf_batch(PrefetchList const &pl,
                                         unsigned long long b) {
  if (b >= pl.nb) {
    return false;
  }
  unsigned long long o = b * PF_BATCH, left = PF_BATCH;
  for (int i = 0; i < pl.cnt && left > 0; i++) {
    if (o >= pl.n[i]) {
      o -= pl.n[i];
      continue;
    }
    for (; o < pl.n[i] && left > 0;) {
      unsigned long long const c = min(min(PF_CHUNK, pl.n[i] - o), left);
      asm volatile(
          "cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(pl.p[i] + o),
          "r"((uint32_t)c)
          : "memory");
      o += c;
      left -= c;
    }
    o = 0;
  }
  return true;
}
#endif

#ifdef STATIC_NO_START_BARRIER
// share idx of n of the resets of the set the next launch uses: no task of this
// launch touches it, the previous launch (which used it) is over because the
// layer runs once per step (not by griddepcontrol.wait: this runs before it, so
// a PDL chain must not launch one layer twice in a row), and the next launch's
// tasks start after this grid's end. 16-B stores, not waited for
__device__ __forceinline__ void reset_next_set(G const &g, int idx, int n) {
  unsigned const other = dbuf_par ^ 1u;
  for (int i = 0; i < g.resets.cnt; i++) {
    unsigned const w = g.resets.v[i] * 0x01010101u;
    uint4 const val = make_uint4(w, w, w, w);
    uint4 *const q =
        reinterpret_cast<uint4 *>(g.resets.p[i] + other * g.resets.stride[i]);
    size_t const n16 = g.resets.n[i] / 16;
    for (size_t k = (size_t)idx * blockDim.x + threadIdx.x; k < n16;
         k += (size_t)n * blockDim.x) {
      q[k] = val;
    }
  }
}
#endif

// before the task loop: shared memory, barriers, the resets
// (STATIC_RESET_IN_KERNEL) or the start barrier over the GPUs, TMEM
__device__ __forceinline__ void kernel_begin(G const &g, KernelLocals &L) {
  extern __shared__ __align__(1024) char smem_raw[];
  L.base = (su32(smem_raw) + 1023u) & ~1023u;
  L.sm = smem_raw + (L.base - su32(smem_raw));
  L.sm_id = blockIdx.x;
  int const sm_id = L.sm_id;
  cta_prologue(L.sm);
#ifdef STATIC_NO_START_BARRIER
  // serving with no start barrier: every buffer the start would reset has two
  // sets (and so do the counters); launch n uses set n % 2, the other was reset
  // at the start of launch n - 1, which this launch's tasks follow
  // (griddepcontrol.wait below), and no task of launch n touches it. The launch
  // number: this SM's own word. PDL as below
  __shared__ unsigned gen_s;
  long long t_start = 0;
  if (threadIdx.x == 0) {
    if constexpr (PDL_TRIGGER == 1) {
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    t_start = gtime();
    unsigned volatile *const gw = g.gen_sm + sm_id;
    unsigned const gen = *gw + 1;
    *gw = gen;
    gen_s = gen;
    dbuf_par = gen & 1u;
#ifdef STATIC_X_FLAG
    x_gen_s = gen;
#endif
  }
  if (threadIdx.x < NSTAMP) {
    g.stamps[sm_id * NSTAMP + threadIdx.x] = 0;
  }
  __syncthreads();

  // the next launch's buffer set, spread over the CTAs, before the wait for the
  // previous kernel (#811 STATIC_RESET_PARITY): this layer's launch before used
  // that set and has ended (one launch per step); the tasks read this launch's
  // set through the generated kernel's gp (the second G, a kernel parameter)
  reset_next_set(g, sm_id, gridDim.x);
  if (threadIdx.x == 0) {
    g.stamps[sm_id * NSTAMP + STAMP_START] = t_start;
    *g.start_barrier = gtime();
  }
#elif defined(STATIC_RESET_IN_KERNEL)
  // the resets left to the start (the host's list: buffers only this GPU
  // writes; the others are re-armed by their readers, REARM), 16-B stores
  // spread over all SMs, each SM its own stamp row. Every SM waits until every
  // SM of this GPU has done its resets (release / acquire on the arrival count,
  // gpu scope: no other GPU writes into these buffers). No barrier over the
  // GPUs: the exchange region has two sets (EXCHANGE_SET), and another GPU's
  // launch n writes only into set n % 2, which this GPU last read in launch n -
  // 2
  // PDL (programmatic dependent launch): the next kernel may launch now (it
  // waits for this grid's end before reading y), and everything up to the TMEM
  // allocation below runs before the wait for the previous kernel: it touches
  // only this kernel's own buffers, never the previous kernel's output (x,
  // prefix). Both instructions are no-ops without a programmatic launch.
  // Only thread 0 uses the launch number (the arrival count, dbuf_par): its
  // load is in flight while the threads do the resets, which do not need it
  long long t_start = 0;
  unsigned gen = 0;
  if (threadIdx.x == 0) {
    if constexpr (PDL_TRIGGER == 1) {
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    t_start = gtime();
    gen = *(unsigned const volatile *)g.gen_ptr + 1;
  }
  for (int i = 0; i < g.resets.cnt; i++) {
    unsigned const w = g.resets.v[i] * 0x01010101u;
    uint4 const val = make_uint4(w, w, w, w);
    uint4 *const q = reinterpret_cast<uint4 *>(g.resets.p[i]);
    size_t const n16 = g.resets.n[i] / 16;
    for (size_t k = (size_t)sm_id * blockDim.x + threadIdx.x; k < n16;
         k += (size_t)gridDim.x * blockDim.x) {
      q[k] = val;
    }
  }
  if (threadIdx.x < NSTAMP) {
    g.stamps[sm_id * NSTAMP + threadIdx.x] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    dbuf_par = gen & 1u; // the other threads read it after tmem_alloc_512's
                         // __syncthreads
    g.stamps[sm_id * NSTAMP + STAMP_START] = t_start;
    cnt_add(g.arrive, 1u); // release: this SM's resets (all its threads': the
                           // barrier above) before its arrival
#ifdef STATIC_L2_PREFETCH
    // while it waits: the batches of the layer's weights not claimed yet. The
    // CTAs that start first (on the SMs the previous kernel leaves first: PDL)
    // claim them; one that arrives last finds the barrier passed and claims
    // none. Launch gen claims on ctr[gen % 2]; SM 0 zeroes the other one below
    // (launch gen - 1, which used it, has ended: every SM is in this launch)
    // The next claim is in flight with each check of the barrier, so a CTA
    // leaves as soon as the check says so (it does not wait for a claim; a
    // claim left unused is a batch not prefetched)
    unsigned long long *const ctr = g.pf.ctr + (gen & 1u);
    unsigned long long b = atomicAdd(ctr, 1ull);
    bool more = true;
    while ((int)(cnt_ld(g.arrive) - gen * gridDim.x) < 0) {
      if (more) {
        more = pf_batch(g.pf, b);
        if (more) {
          b = atomicAdd(ctr, 1ull);
        }
      }
    } // acquire: every SM of this launch has arrived (wrap-safe)
    if (sm_id == 0) {
      g.pf.ctr[(gen & 1u) ^ 1u] = 0ull;
    }
#else
    while ((int)(cnt_ld(g.arrive) - gen * gridDim.x) < 0) {
    } // acquire: every SM of this launch has arrived (wrap-safe)
#endif
    if (sm_id == 0) {
      *g.gen_ptr = gen; // read by the next launch (after this one ends)
    }
    *g.start_barrier = gtime();
  }
#else
  // without STATIC_RESET_IN_KERNEL: the host resets the buffers and starts all
  // GPUs together; the start barrier lines the GPUs up (the layer's measured
  // span starts there): SM 0 of each GPU writes the launch number into its
  // hello slot on every GPU (this launch's set), every SM waits for all slots
  if (threadIdx.x == 0) {
    asm volatile("griddepcontrol.wait;" ::
                     : "memory"); // PDL: from here on global memory is read
                                  // (no-op without a programmatic launch)
    g.stamps[sm_id * NSTAMP + STAMP_START] = gtime();
    unsigned const gen = g.gen;
    dbuf_par =
        gen &
        1u; // the other threads read it after tmem_alloc_512's __syncthreads
    if (sm_id == 0) {
      push16(g,
             EXCHANGE_HELLO + (size_t)g.rank * 16,
             make_uint4(gen, gen, gen, gen));
    }
    for (int r = 0; r < GPUS; r++) {
      unsigned const volatile *hp =
          (unsigned const volatile *)(g.rv + EXCHANGE_HELLO + (size_t)r * 16 +
                                      rg_set());
      while (*hp < gen) {
      }
    }
    *g.start_barrier = gtime();
  }
#endif
  L.tb = tmem_alloc_512(); // its __syncthreads also publishes rt
#ifdef STATIC_RESET_IN_KERNEL
#ifdef STATIC_X_FLAG
  // H1: with x's flag bound, no wait here: the generated loop waits before the
  // CTA's first task that is not pre-wait safe (x_grid_wait), kernel_end at
  // the latest, so every CTA waits before it exits (the chain stays whole)
  if (threadIdx.x == 0) {
    x_state_s = g.xflag_in == nullptr ? 1u : 0u;
    if (g.xflag_in == nullptr) {
      asm volatile("griddepcontrol.wait;" ::: "memory");
    }
  }
  __syncthreads();
#else
  if (threadIdx.x == 0) {
    asm volatile("griddepcontrol.wait;" ::
                     : "memory"); // PDL: from here on the previous kernel's
                                  // output is read
  }
  __syncthreads();
#endif
#endif
  L.gl = 0;
  L.gi0 = 0;
  L.gi1 = 0;
  L.wt = 0;
  L.ring_relaid = false;
}

// after the task loop
#ifdef STATIC_X_FLAG
// H1: the wait for the kernel before's grid, once per CTA: before its first
// task that is not pre-wait safe (generated), and in kernel_end
__device__ __forceinline__ void x_grid_wait(KernelLocals &) {
  if (!(x_state_s & 1u)) {
    if (threadIdx.x == 0) {
      asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    __syncthreads(); // every thread has read the old state
    if (threadIdx.x == 0) {
      x_state_s |= 1u;
    }
    __syncthreads(); // the new one for every later check
  }
}
// H1: before a task that reads x: every post task of the kernel before has
// stored this launch's number into its word (after x: release; here acquire
// by threads 0 .. XFLAG_N - 1, then the barrier). Only that number passes: an
// older one is the last step's (wait); the kernels of a chain launch the same
// number of times, else this waits for ever (never a stale x). After the grid
// wait x is visible anyway. The proxy fence: x may be read by TMA (gemm_tile)
__device__ __forceinline__ void x_flag_poll(G const &g, KernelLocals &) {
  if (!(x_state_s & 3u)) { // (bit 0 is set whenever xflag_in is nullptr)
    unsigned const want = x_gen_s;
    if (threadIdx.x < XFLAG_N) {
#ifdef STATIC_X_FLAG_DEBUG
      long long const t0 = gtime();
#endif
      while (cnt_ld(g.xflag_in + threadIdx.x) != want) {
#ifdef STATIC_X_FLAG_DEBUG
        if (gtime() - t0 > 2000000000ll) {
          printf("x_flag_poll: CTA %d word %d holds %u, launch %u: no x\n",
                 (int)blockIdx.x,
                 (int)threadIdx.x,
                 cnt_ld(g.xflag_in + threadIdx.x),
                 want);
          __trap();
        }
#endif
      }
    }
    __syncthreads();
    asm volatile("fence.proxy.async.global;" ::: "memory");
    if (threadIdx.x == 0) {
      x_state_s |= 2u;
    }
    __syncthreads();
  }
}
// H1, the post: this task's columns of x and the streams stored (the caller's
// barrier before): its word := this launch's number (release, gpu scope)
__device__ __forceinline__ void x_flag_store(G const &g, int task) {
  if (g.xflag_out != nullptr) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(g.xflag_out + task),
                 "r"(x_gen_s)
                 : "memory");
  }
}
#endif

__device__ __forceinline__ void kernel_end(G const &g, KernelLocals &L) {
#ifdef STATIC_X_FLAG
  x_grid_wait(L); // H1: every CTA waits before it exits
#endif
  __syncthreads();
  if (threadIdx.x == 0) {
    g.stamps[L.sm_id * NSTAMP + STAMP_END] = gtime();
  }
  tmem_dealloc_512(L.tb);
}

} // namespace static_mk

// (11-20, host only, default off: host.cuh static_host_rebind; at the end of
// this file, so that no line of the default build moves) a module shared by
// several layers: each layer's own per-SM stamps and start-barrier word, for
// in-situ samples ("@stamps", "@start_barrier")
#ifdef STATIC_STAMP_REBIND
#define STATIC_STAMP_REBIND_(gg, name, p)                                      \
  if ((name) == "@stamps") {                                                   \
    (gg).stamps = reinterpret_cast<long long *>(p);                            \
    return true;                                                               \
  }                                                                            \
  if ((name) == "@start_barrier") {                                            \
    (gg).start_barrier = reinterpret_cast<long long *>(p);                     \
    return true;                                                               \
  }
#else
#define STATIC_STAMP_REBIND_(gg, name, p)
#endif
