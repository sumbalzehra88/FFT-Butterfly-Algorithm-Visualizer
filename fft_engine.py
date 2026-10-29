"""
fft_engine.py  —  Person 1
==========================

Floating-point Radix-2 Decimation-in-Time (DIT) FFT that exposes its internals.

Unlike numpy.fft.fft, which returns only the final spectrum, this engine
records *everything a hardware designer cares about*:

  * the bit-reversed input ordering,
  * the data vector after every stage (what sits in RAM between passes),
  * every butterfly: which two addresses it touches, which twiddle factor
    W_N^k it applies, and its input / product / output values.

The iterative in-place algorithm used here is exactly the structure that is
implemented on FPGAs and DSP chips:

    for each stage s = 1 .. log2(N):
        span   m    = 2^s           (size of the sub-DFTs being built)
        half   m/2                  (distance between butterfly partners)
        stride N/m                  (step through the twiddle ROM)
        for each group of m points:
            for j in 0 .. m/2-1:
                top    = base + j
                bottom = base + j + m/2
                t      = W_N^(j*stride) * x[bottom]
                x[top], x[bottom] = x[top] + t, x[top] - t

The function `butterfly_schedule()` generates only the *index* pattern of
this loop. It is the single source of truth shared by `fixed_point.py`
(which replays the same arithmetic in fixed point) and `memory_mapper.py`
(which turns the same indices into RAM addresses).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# A butterfly in the schedule: (stage, group, j, top_addr, bottom_addr, twiddle_k)
ScheduleEntry = Tuple[int, int, int, int, int, int]


# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #
def is_power_of_two(n: int) -> bool:
    """True if n is a positive power of two (1, 2, 4, 8, ...)."""
    return n > 0 and (n & (n - 1)) == 0


def num_stages(N: int) -> int:
    """Number of butterfly stages, log2(N). Raises for non power-of-two N."""
    if not is_power_of_two(N) or N < 2:
        raise ValueError(f"Radix-2 FFT needs N to be a power of two >= 2 (got {N}).")
    return N.bit_length() - 1


def bit_reverse(i: int, nbits: int) -> int:
    """Reverse the lowest `nbits` bits of integer i  (e.g. 001 -> 100)."""
    r = 0
    for _ in range(nbits):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r


def bit_reverse_permutation(N: int) -> np.ndarray:
    """perm[a] = natural index of the sample stored at RAM address a."""
    nbits = num_stages(N)
    return np.array([bit_reverse(i, nbits) for i in range(N)], dtype=int)


def twiddle_factor(k, N: int):
    """W_N^k = exp(-j*2*pi*k/N). Works on scalars or numpy arrays of k."""
    return np.exp(-2j * np.pi * np.asarray(k) / N)


def twiddle_table(N: int) -> np.ndarray:
    """The twiddle ROM contents: W_N^0 .. W_N^(N/2-1)."""
    return twiddle_factor(np.arange(N // 2), N)


def butterfly_schedule(N: int) -> List[List[ScheduleEntry]]:
    """
    Index-only description of every butterfly, grouped by stage.

    Returns a list (one entry per stage) of tuples
        (stage, group, j, top, bottom, twiddle_k)
    where `stage` is 1-based and `top`/`bottom` are in-place RAM addresses.
    """
    S = num_stages(N)
    schedule: List[List[ScheduleEntry]] = []
    for s in range(1, S + 1):
        m = 1 << s          # span of each sub-DFT
        half = m >> 1       # distance between the two butterfly inputs
        stride = N // m     # twiddle ROM step for this stage
        stage = []
        for g, base in enumerate(range(0, N, m)):
            for j in range(half):
                stage.append((s, g, j, base + j, base + j + half, j * stride))
        schedule.append(stage)
    return schedule


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class Butterfly:
    """Everything that happens inside one radix-2 butterfly node."""
    stage: int            # 1-based stage number
    index: int            # butterfly number within its stage (0 .. N/2-1)
    group: int            # which sub-DFT block it belongs to
    top: int              # RAM address of the upper input/output
    bottom: int           # RAM address of the lower input/output
    twiddle_k: int        # exponent k of W_N^k
    twiddle: complex      # value of W_N^k
    a_in: complex         # upper input
    b_in: complex         # lower input
    product: complex      # W * b_in  (the complex multiplier output)
    a_out: complex        # a_in + W*b_in
    b_out: complex        # a_in - W*b_in


@dataclass
class FFTResult:
    N: int
    input: np.ndarray                  # natural-order input x[n]
    reordered_input: np.ndarray        # bit-reversed input (RAM contents before stage 1)
    stage_values: np.ndarray           # shape (S+1, N); row 0 = reordered input, row s = after stage s
    butterflies: List[List[Butterfly]] = field(default_factory=list)  # per stage
    output: np.ndarray = None          # X[k], natural order

    @property
    def stages(self) -> int:
        return self.stage_values.shape[0] - 1


# --------------------------------------------------------------------------- #
# The FFT
# --------------------------------------------------------------------------- #
def fft_radix2_dit(x) -> FFTResult:
    """
    Iterative, in-place, radix-2 DIT FFT in double-precision floating point.

    Parameters
    ----------
    x : array-like (real or complex), length N = power of two

    Returns
    -------
    FFTResult with stage-by-stage values and every butterfly recorded.
    """
    x = np.asarray(x, dtype=complex).ravel()
    N = len(x)
    S = num_stages(N)

    perm = bit_reverse_permutation(N)
    data = x[perm].copy()                       # step 0: bit-reversed load

    stage_values = np.zeros((S + 1, N), dtype=complex)
    stage_values[0] = data
    all_butterflies: List[List[Butterfly]] = []

    for stage in butterfly_schedule(N):
        records = []
        for idx, (s, g, j, top, bot, k) in enumerate(stage):
            W = complex(twiddle_factor(k, N))
            a, b = data[top], data[bot]
            t = W * b
            data[top], data[bot] = a + t, a - t
            records.append(Butterfly(s, idx, g, top, bot, k, W, a, b, t, a + t, a - t))
        stage_values[stage[0][0]] = data
        all_butterflies.append(records)

    return FFTResult(N=N, input=x, reordered_input=x[perm],
                     stage_values=stage_values, butterflies=all_butterflies,
                     output=data.copy())


def verify_against_numpy(x) -> float:
    """Max absolute difference between this engine and numpy.fft.fft (sanity check)."""
    return float(np.max(np.abs(fft_radix2_dit(x).output - np.fft.fft(x))))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for n in (2, 4, 8, 16, 32, 64, 1024):
        err = verify_against_numpy(rng.standard_normal(n) + 1j * rng.standard_normal(n))
        print(f"N={n:5d}  max|error vs numpy| = {err:.2e}")