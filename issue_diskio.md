# Disk I/O Minimization — Research

Branch: `experiment/diskio-research` (based on `dev` at `62419af`)

## TL;DR

The engine's disk I/O is **already well-engineered on the hottest path** (expert streaming uses coalesced O_DIRECT `pread` + `posix_fadvise` hints + LRU + pin cache + speculative prefetch). There are **4 concrete, bounded opportunities** to shave latency, ranked by ROI:

| # | Opportunity | Where | Frequency | Estimated win |
|---|---|---|---|---|
| 1 | **KV-cache write batching** (157 fwrites/token → 1) | `kv_disk_append` | per turn | cuts ~100s of syscalls/turn |
| 2 | **`/proc/meminfo` fopen storm** | `rss_gb()` | ~every 16 tokens (Linux) | eliminates recurring open/read/close |
| 3 | **Expert prefetch on Windows** (`PrefetchVirtualMemory`) | `expert_prefetch` | per miss | mmap path is Linux/macOS-only today |
| 4 | **KV-cache: buffered handle kept open** | `kv_disk_append` | per turn | kills open+fseek+close per turn |

There are also **2 non-opportunities** worth recording so we don't re-investigate: O_DIRECT for experts (correctly used today), and PagedAttention-style file layout (already single-file + indexed).

---

## How the engine does disk I/O today

There are **three I/O stacks**, behaving very differently:

| Stack | Mechanism | Frequency | Files |
|---|---|---|---|
| **Expert weights** (hottest) | `pread` on kept-open fds + `posix_fadvise`, optional `mmap` | per miss, every token | `st.h`, `glm.c:1328` |
| **KV cache** (`.coli_kv`) | `fopen` + `fwrite`/`fread` | per turn | `glm.c:3812-3889` |
| **Everything else** (config, tokenizer, stats, grammar) | `fopen` + `fread` | startup-only | scattered |

### Expert path (the hot path — already good)

`expert_load` (`glm.c:1328-1481`) has three sub-paths:

- **Default `pread` path** (`glm.c:1385-1472`): coalesces the 3 contiguous expert tensors (gate/up/down) into **one ~19 MB O_DIRECT `pread`** into a 16K-aligned slab (`glm.c:1447`). Falls back to 3 separate `pread`s only if non-contiguous. Scales are 3 tiny separate `pread`s (kilobytes). `posix_fadvise(DONTNEED)` evicts pages after if `g_drop`. **This is well-batched — one syscall for ~19 MB.**
- **`COLI_MMAP=1` path** (`glm.c:1352-1383`): `mmap` per shard fd (cached), `madvise(WILLNEED)` + synchronous page-touch loop. Zero-copy. **Default OFF, and Linux/macOS/FreeBSD-only** — no `MapViewOfFile` on Windows.
- **Prefetch hints**: `expert_prefetch` (`glm.c:1602-1609`) → `st_prefetch` (`st.h:178`) issues `posix_fadvise(WILLNEED)` — readahead hint only, no data read. Called from `moe` next-64-block lookahead, pilot, and SPEC.

### KV cache persistence (per turn — opportunity here)

`kv_disk_append` (`glm.c:3834-3855`), called once per turn:
1. `fopen("r+b")` — **reopens the file every turn**
2. `fseek` to append position
3. **per-position loop**: for each new token, `fwrite` the token i32, then **2 fwrites per layer** (Lc + Rc) + optional DSA Ic. With 78 layers that's **~157 fwrites per token appended**.
4. `fflush` (userspace only — **no fsync/fdatasync anywhere in the codebase**)
5. `fseek` back to header + `fwrite` the new nrec counter (crash-safe ordering)
6. `fclose`

Record size ~182 KB/token. On a long first turn this is **tens of thousands of small fwrites**. stdio buffering coalesces them into fewer `write` syscalls, but the userspace overhead remains.

### Recurring surprise: `/proc/meminfo`

`rss_gb()` (`glm.c:4625`) does `fopen("/proc/meminfo")` + fgets + fclose. Called from every STAT line and every 16-token heartbeat (`glm.c:3473, 3477`). On Linux this is an **open+read+close of procfs ~every 16 tokens**. (Windows uses `compat_meminfo`, no file — not affected.)

---

## What similar projects do

**llama.cpp** (the reference): `mmap`s the entire model read-only, uses `--mlock` to pin hot pages, streams layers to GPU via partial offload, and issues per-pass readahead of upcoming tensors (`llama-mmap.cpp`). Justine Tunney's mmap work: "load 100× faster using half as memory." Crucial finding from discussion #18758: **for MoE, mmap beats O_DIRECT** when the model fits in ~RAM — O_DIRECT takes "at least 10× longer" on repeated loads because it bypasses the page cache that serves re-faults for free.

**The general consensus across llama.cpp, vLLM, AirLLM, PRESERVE, HOBBIT, SolidAttention (FAST '26):**
- mmap + OS page cache as the backing store for an LRU is the proven recipe
- prefetch the *next* expert/layer while computing the current one — this is where the 0.5ms lives
- single indexed file (one `open()`) beats one-file-per-expert
- align tensors to 4KB (preferably 64KB) for clean page-fault boundaries + SSD geometry
- buffer sweet spot ~1MB; syscall cost ~1-5µs each, so batching matters at high repetition

---

## The 4 opportunities (ranked)

### Opportunity 1 — KV-cache write batching (HIGH ROI, LOW risk)

**Problem:** `kv_disk_append` does ~157 `fwrite` calls per appended token (1 token i32 + 2×78 layers). stdio buffering hides some of this, but on a long first-turn prefill (hundreds-thousands of tokens) this is tens of thousands of fwrites.

**Fix:** Build one contiguous record in a heap buffer (token + all layers' Lc/Rc/Ic for that position), then **a single `fwrite` per position** (or even one `fwrite` for the whole turn). The data is already laid out contiguously in memory per-layer (`coli_kv_row`), so a layered `memcpy` into a staging buffer + one write is straightforward.

**Win:** ~157× fewer fwrite calls per token. Even with stdio coalescing, the userspace loop overhead is real at scale.

### Opportunity 2 — `/proc/meminfo` fopen storm (MEDIUM ROI, trivial)

**Problem:** `rss_gb()` opens, reads, closes `/proc/meminfo` every ~16 tokens on Linux. Each is ~3 syscalls + path resolution.

**Fix:** Either (a) cache the value for N tokens (e.g. re-read at most once per second), or (b) keep the fd open and `rewind`+`fgets`. Trivial change.

### Opportunity 3 — Expert prefetch on Windows (MEDIUM ROI, bounded)

**Problem:** The `COLI_MMAP=1` path (which gives zero-copy expert access + free OS-cache re-faults) is **Linux/macOS/FreeBSD-only** — `glm.c:1301` guards it. On Windows, experts always go through the `pread` path, and `expert_prefetch` issues `posix_fadvise(WILLNEED)` which is a no-op shim on Windows (`compat.h`).

**Fix:** On Windows, implement the prefetch via `PrefetchVirtualMemory` (the Win32 analog of `MADV_WILLNEED`) on an mmap'd region, or via an async `ReadFile`+`OVERLAPPED` into a scratch buffer. This brings the Windows build closer to parity with the Linux mmap+prefetch story.

**Scope:** This is the largest of the four — it touches the Windows I/O path. Worth doing if Windows perf is a goal; skip if Linux is the target.

### Opportunity 4 — KV-cache: keep handle open (LOW-MEDIUM ROI, LOW risk)

**Problem:** `kv_disk_append` does `fopen`+...+`fclose` every turn. Handle creation is ~5-15µs of pure overhead (worse on Windows).

**Fix:** Open the KV file once (lazily on first append), keep the `FILE*` for the engine lifetime, just `fseek`+write each turn. Close on shutdown. Pair with Opportunity 1 for the write batching.

---

## Non-opportunities (recording so we don't re-investigate)

- **O_DIRECT for experts**: already correctly used (`st.h:83`, `DIRECT=1`). For an LRU+refetch pattern the page cache is your friend, but the engine offers both paths (O_DIRECT pread default + optional mmap) and the O_DIRECT coalesced read is already one syscall for ~19MB. Don't change this.
- **Single-file layout**: the engine already uses safetensors shards with kept-open fds + offset-indexed tensors (`st.h`). No per-expert open()/close() waste. Don't change this.
- **PagedAttention**: solves concurrency fragmentation this engine doesn't have (≤16 slots). Not applicable.

---

## Next steps

The highest-ROI, lowest-risk starting point is **Opportunity 1 (KV write batching) + Opportunity 4 (keep handle open)** — they're in the same function, both low-risk, and together they eliminate the per-turn open/close overhead and the per-token fwrite storm. Opportunity 2 is a trivial 5-minute fix we can bundle in.

Opportunity 3 (Windows prefetch) is the biggest single win but also the largest scope — separate effort, gated on whether Windows perf is a priority.

## Sources

- [justine.lol/mmap — Edge AI Just Got Faster](https://justine.lol/mmap/)
- [llama.cpp discussion #18758 — Mmap faster than direct I/O for MoE](https://github.com/ggml-org/llama.cpp/discussions/18758)
- [llama.cpp issue #20757 — Two-tier GPU+RAM expert cache](https://github.com/ggml-org/llama.cpp/issues/20757)
- [FAST '26 — Programmable Page Cache for LLM loading](https://www.usenix.org/system/files/fast26-liu-yubo.pdf)
- [FAST '26 — SolidAttention: SSD-based serving](https://www.usenix.org/system/files/fast26-zheng.pdf)
- [HOBBIT — Mixed precision expert offloading](https://arxiv.org/html/2411.01433v2)
- [posix_fadvise(2) — man7.org](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html)
- [madvise(2) — man7.org](https://man7.org/linux/man-pages/man2/madvise.2.html)
- [Microsoft Learn — File Buffering (FILE_FLAG_NO_BUFFERING)](https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering)
- [Microsoft Learn — PrefetchVirtualMemory](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-prefetchvirtualmemory)
- [What makes system calls expensive — codingconfessions.com](https://blog.codingconfessions.com/p/what-makes-system-calls-expensive)
- [Syscall overhead — Stack Overflow](https://stackoverflow.com/questions/8247331/syscall-overhead)

---

# Windows Implementation — branch `windows-optimizations` (2026-07-15)

## What landed (pread path, validated)

Two changes, both on the `pread` expert-load path (no mmap). Measured against the
existing `bench_budget*.txt` baselines (GLM-5.2 744B int4, 32 GB RAM, Core Ultra 9
185H, DRAFT=0, 32-token decode):

### 1. `compat_fadvise` WILLNEED cache-warmer (`c/compat.h`)

Replaced the Windows `posix_fadvise` no-op (was a `do{}while(0)` macro) with a real
readahead: an overlapped `ReadFile` into a throwaway scratch buffer that populates the
standby page cache, so the later synchronous `pread` faults from RAM not disk. Mirrors
the macOS `F_RDADVISE` shim (`compat.h:28-37`). DONTNEED stays a no-op (matches macOS;
Windows standby-list trimming self-regulates under pressure).

This re-arms the existing `expert_prefetch` → `st_prefetch` → `posix_fadvise(WILLNEED)`
chain on Windows: the next-block readahead in `moe()` and the PILOT cross-layer prefetch
hints now actually warm the cache instead of being silently discarded.

**Measured effect (budget=4, PIPE on):** hit rate 16.4% → 27.6%.

### 2. PIPE default ON for Windows (`c/glm.c`)

Flipped the async expert-load thread pool from default OFF to default ON on Windows
(`getenv("PIPE")?:1` under `_WIN32`, unchanged `:0` elsewhere). PIPE dispatches expert
`pread` loads onto worker threads so they overlap the expert matmul on the forward-pass
thread, instead of the blocking serial load-then-compute path. `PIPE=0` opts back out.

**Measured effect (budget=4):** expert-disk 65.9s → 54.3s (−18%), reaching **1.70 s/tok**
(under the 2 s/tok target; budget=4 baseline was 2.06 s/tok).

### Results table (DRAFT=0, 32-token decode, pread path)

| config | expert-disk | s/tok | hit% | tok/s |
|---|---|---|---|---|
| budget=4, no PIPE (existing baseline) | 65.9s | 2.06 | 16.4% | 0.33 |
| **budget=4 + PIPE (this PR)** | **54.3s** | **1.70** | **27.6%** | **0.34** |
| budget=6 + PIPE | 77.1s | 2.41 | 21.8% | 0.27 |

budget=4 + PIPE meets the ≤2 s/tok target. budget=6 (more experts/layer, higher quality)
misses it at 2.41 s/tok — the speed/quality tradeoff.

## What was tried and abandoned: Windows mmap (`COLI_MMAP` on `_WIN32`)

The original plan (informed by llama.cpp #18758: "mmap is ≥10× faster than O_DIRECT for
MoE") was to port the mmap expert path to Windows via `CreateFileMapping`/`MapViewOfFile`.
This was implemented and tested at length. **It was a measured regression and was reverted.**

### The attempt

Added a `_WIN32` branch to `map_of_fd` (`glm.c`) mapping each shard file read-only and
resolving experts as views into the mapping, mirroring the POSIX path. Also added
`PrefetchVirtualMemory` readahead and a `VirtualUnlock` eviction mechanism (the Windows
`posix_fadvise(DONTNEED)` analog — see SO#1880714; validated standalone to demote pages
to the standby list with a 2.3× faster re-fault).

### Why it regressed

**mmap'd expert pages bloat the process working set on Windows, which collapses the
expert cache.** This is a fundamental Windows-vs-Linux difference:

- On Linux, `mmap(MAP_SHARED)` file pages live in the kernel page cache (`buff/cache`),
  separate from `MemAvailable`, so the cache budget isn't fooled.
- On Windows, touched `MapViewOfFile` pages count against `ullAvailPhys` (what
  `compat_meminfo` reads for the budget). The CPU matmul touches every weight byte,
  faulting ~12 GB into the working set. `cap_for_ram()` then sees ~no free RAM and
  collapses the LRU cache cap.

Measured (budget=0, DRAFT=0, apples-to-apples):

| config | RAM_GB detected | cache cap | hit% | expert-disk | RSS |
|---|---|---|---|---|---|
| baseline (pread) | 21.4 | 1 | 9.3% | 133s | 15.0 GB |
| mmap, no eviction | **8.0** | 1 | **2.2%** | **240s** | **27.2 GB** |
| mmap + VirtualUnlock | 24.9 | 2 | 11.8% | 83s | 18.1 GB |
| mmap + reserve reductions | 24.6 | 4 | 21.8% | 80s | 20.1 GB |

The `VirtualUnlock` eviction recovered the regression (240s→83s), and dropping the
Linux-specific page-cache/slab reserves under mmap got it to parity with pread. But it
never clearly *beat* the simpler pread+PIPE path, and it added substantial complexity
(per-slot eviction tracking, reserve conditionals, `VirtualUnlock` on every slot recycle).
**The engine already moved off mmap to pread for this exact RSS bug** (`st.h:3-6`), and
the Windows port re-confirmed that decision.

### What else didn't work

- **Batched `PrefetchVirtualMemory`** for the mmap path: tested as a single batched
  readahead of all 64 missed experts' pages before the matmul. **Blocked instead of
  prefetching async** on this SSD — inflated `t_edisk` (80s→102s). Consistent with
  microsoft/Windows-Dev-Performance#108 ("PrefetchVirtualMemory does not prefetch").
  Reverted.
- **True I/O/compute overlap on the CPU path**: the Metal path has this ("submit
  resident experts to GPU before loading misses"), but the CPU path loads-then-computes
  serially. `PrefetchVirtualMemory` was the attempt to add it for mmap and failed. The
  pread path gets overlap via PIPE (which works), not via mmap prefetch.

### Conclusion

For this engine on Windows at this RAM budget (~32 GB, 370 GB model), **pread + PIPE +
compat_fadvise** is the right path. mmap remains valuable on Linux/macOS (where the page
cache doesn't inflate process RSS) but is not viable on Windows without a fundamentally
different cache-budget model that excludes mapped-file pages — left as future work.

## Sources added

- [SO#1880714 — VirtualUnlock releases mapped pages to standby list](https://stackoverflow.com/questions/1880714/createfilemapping-mapviewoffile-how-to-avoid-holding-up-the-system-memory)
- [Alois Kraus — The Mysterious Lost Memory (modified/standby list)](https://aloiskraus.wordpress.com/2017/02/26/the-mysterious-lost-memory-which-belongs-to-no-process/)
- [microsoft/Windows-Dev-Performance#108 — PrefetchVirtualMemory inconsistency](https://github.com/microsoft/Windows-Dev-Performance/issues/108)
- [llama.cpp #18758 — mmap faster than O_DIRECT for MoE (Linux)](https://github.com/ggml-org/llama.cpp/discussions/18758)
- [HN#35426679 — Why MMAP in llama.cpp hides true memory usage](https://news.ycombinator.com/item?id=35426679)
