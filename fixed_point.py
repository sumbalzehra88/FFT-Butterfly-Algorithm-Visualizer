"""
fixed_point.py  —  Person 2
===========================

Fixed-point (two's-complement) arithmetic model of the FFT datapath.

Notation
--------
We use the "Qm.n" convention in which m INCLUDES the sign bit:

    Q1.15  -> 16-bit word, 1 sign/integer bit, 15 fractional bits
              range [-1, 1 - 2^-15],  LSB = 2^-15
    Q2.10  -> 12-bit word, range [-2, 2 - 2^-10]

What is modelled
----------------
* quantize()           : float -> fixed (round or truncate; saturate or wrap)
* fixed_point_fft()    : the same radix-2 DIT butterflies as fft_engine.py,
                         but every multiplier and adder output is re-quantized
                         to the datapath word length, exactly as an FPGA with
                         fixed-width registers would do.
                         Optional 1/2 scaling per stage (arithmetic shift right)
                         prevents overflow from bit-growth.
* bit-growth analysis  : how many integer bits each stage *would* need.
* error / SNR metrics  : truncation error, overflow counts, output SNR, ENOB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from fft_engine import (bit_reverse_permutation, butterfly_schedule,
                        fft_radix2_dit, num_stages, twiddle_factor)


# --------------------------------------------------------------------------- #
# Q-format description
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QFormat:
    int_bits: int    # integer bits INCLUDING the sign bit (>= 1)
    frac_bits: int   # fractional bits (>= 0)

    def __post_init__(self):
        if self.int_bits < 1 or self.frac_bits < 0:
            raise ValueError("QFormat needs int_bits >= 1 and frac_bits >= 0")

    # --- derived properties ------------------------------------------------ #
    @property
    def total_bits(self) -> int:
        return self.int_bits + self.frac_bits

    @property
    def lsb(self) -> float:
        """Weight of one least-significant bit (the quantization step)."""
        return 2.0 ** -self.frac_bits

    @property
    def max_code(self) -> int:
        return (1 << (self.total_bits - 1)) - 1

    @property
    def min_code(self) -> int:
        return -(1 << (self.total_bits - 1))

    @property
    def max_value(self) -> float:
        return self.max_code * self.lsb

    @property
    def min_value(self) -> float:
        return self.min_code * self.lsb

    @property
    def name(self) -> str:
        return f"Q{self.int_bits}.{self.frac_bits}"

    def __str__(self) -> str:
        return f"{self.name} ({self.total_bits}-bit)"

    # --- constructors ------------------------------------------------------ #
    @classmethod
    def from_bits(cls, total_bits: int, int_bits: int = 1) -> "QFormat":
        """Build a format from a total word length (what the precision slider sets)."""
        if total_bits <= int_bits:
            raise ValueError("total_bits must exceed int_bits")
        return cls(int_bits, total_bits - int_bits)

    @classmethod
    def from_string(cls, text: str) -> "QFormat":
        """Parse 'Q1.15' -> QFormat(1, 15)."""
        m, n = text.strip().upper().lstrip("Q").split(".")
        return cls(int(m), int(n))


# Handy presets that match common hardware word lengths.
PRESETS = {
    "Q1.7  (8-bit)": QFormat(1, 7),
    "Q1.11 (12-bit)": QFormat(1, 11),
    "Q1.15 (16-bit)": QFormat(1, 15),
    "Q1.31 (32-bit)": QFormat(1, 31),
}


def default_twiddle_format(total_bits: int) -> QFormat:
    """
    Twiddle ROM format. Twiddles lie in [-1, 1]; using 2 integer bits lets
    W = +1 be stored exactly (in Q1.x it would saturate to 1 - LSB).
    """
    return QFormat(2, max(total_bits - 2, 1))


# --------------------------------------------------------------------------- #
# Quantization
# --------------------------------------------------------------------------- #
@dataclass
class QuantResult:
    values: np.ndarray            # quantized values (float carrying exact fixed-point values)
    codes_re: np.ndarray          # integer two's-complement codes, real part
    codes_im: Optional[np.ndarray]  # integer codes, imaginary part (None for real input)
    overflow: np.ndarray          # bool per element: True if re OR im overflowed
    n_overflow: int               # number of overflowing scalar components
    error: np.ndarray             # quantized - original


def _quantize_real(x: np.ndarray, fmt: QFormat, rounding: str, overflow: str):
    """Quantize a real array. Returns (codes, overflow_mask)."""
    scaled = np.asarray(x, dtype=float) * (2.0 ** fmt.frac_bits)
    if rounding == "truncate":
        codes = np.floor(scaled)                 # two's-complement truncation = floor
    elif rounding == "round":
        codes = np.floor(scaled + 0.5)           # round half up
    else:
        raise ValueError("rounding must be 'round' or 'truncate'")
    codes = codes.astype(np.int64)

    ovf = (codes > fmt.max_code) | (codes < fmt.min_code)
    if overflow == "saturate":
        codes = np.clip(codes, fmt.min_code, fmt.max_code)
    elif overflow == "wrap":
        span = 1 << fmt.total_bits                # modular two's-complement wrap-around
        codes = ((codes - fmt.min_code) % span) + fmt.min_code
    else:
        raise ValueError("overflow must be 'saturate' or 'wrap'")
    return codes, ovf


def quantize(x, fmt: QFormat, rounding: str = "round",
             overflow: str = "saturate") -> QuantResult:
    """
    Quantize real or complex data to the given Q-format.
    Complex numbers are quantized component-wise (separate I and Q registers).
    """
    arr = np.atleast_1d(np.asarray(x))
    if np.iscomplexobj(arr):
        cr, ore = _quantize_real(arr.real, fmt, rounding, overflow)
        ci, oim = _quantize_real(arr.imag, fmt, rounding, overflow)
        values = (cr + 1j * ci) * fmt.lsb
        ovf = ore | oim
        n_ovf = int(ore.sum() + oim.sum())
    else:
        cr, ovf = _quantize_real(arr, fmt, rounding, overflow)
        ci = None
        values = cr * fmt.lsb
        n_ovf = int(ovf.sum())
    return QuantResult(values, cr, ci, ovf, n_ovf, values - arr)


# --------------------------------------------------------------------------- #
# Binary formatting helpers (for the UI)
# --------------------------------------------------------------------------- #
def value_to_code(value: float, fmt: QFormat) -> int:
    """Integer code of a value that is already on the fmt grid."""
    return int(np.round(value * (2.0 ** fmt.frac_bits)))


def code_to_binary(code: int, total_bits: int) -> str:
    """Two's-complement bit string of an integer code."""
    return format(code & ((1 << total_bits) - 1), f"0{total_bits}b")


def format_binary(value: float, fmt: QFormat) -> str:
    """'1.0110…' style string with the binary point placed after the integer bits."""
    bits = code_to_binary(value_to_code(value, fmt), fmt.total_bits)
    return bits[:fmt.int_bits] + "." + bits[fmt.int_bits:]


# --------------------------------------------------------------------------- #
# Error / noise metrics
# --------------------------------------------------------------------------- #
def snr_db(reference, test) -> float:
    """SNR = 10*log10( signal power / error power ). Returns +inf for a perfect match."""
    ref = np.asarray(reference)
    err = np.asarray(test) - ref
    p_sig = float(np.sum(np.abs(ref) ** 2))
    p_err = float(np.sum(np.abs(err) ** 2))
    if p_err == 0.0:
        return float("inf")
    if p_sig == 0.0:
        return float("-inf")
    return 10.0 * np.log10(p_sig / p_err)


def enob(snr: float) -> float:
    """Effective number of bits from SNR: (SNR - 1.76) / 6.02."""
    return (snr - 1.76) / 6.02


def theoretical_snr_db(total_bits: int) -> float:
    """Ideal SNR of a full-scale sine through a B-bit quantizer: 6.02*B + 1.76 dB."""
    return 6.02 * total_bits + 1.76


def quantization_error_stats(original, quantized, fmt: QFormat) -> dict:
    """Summary of the error introduced by one quantization step."""
    e = np.asarray(quantized) - np.asarray(original)
    comps = np.concatenate([e.real.ravel(), e.imag.ravel()]) if np.iscomplexobj(e) else e.ravel()
    return {
        "max_abs_error": float(np.max(np.abs(comps))),
        "rms_error": float(np.sqrt(np.mean(comps ** 2))),
        "mean_error (bias)": float(np.mean(comps)),
        "lsb": fmt.lsb,
        "theoretical_rms (round)": fmt.lsb / np.sqrt(12.0),
        "theoretical_bias (truncate)": -fmt.lsb / 2.0,
    }


# --------------------------------------------------------------------------- #
# Bit-growth analysis
# --------------------------------------------------------------------------- #
def required_int_bits(max_abs: float) -> int:
    """Integer bits (incl. sign) needed so that |v| <= max_abs never overflows."""
    if max_abs <= 0:
        return 1
    return max(1, int(np.floor(np.log2(max_abs))) + 2)


def growth_profile(stage_values: np.ndarray) -> List[dict]:
    """
    For each row of a (S+1, N) stage-value matrix, report the largest real/imag
    component and the integer bits needed to hold it.
    Each radix-2 stage can grow the magnitude by up to 2x (~1 bit); for complex
    data the worst-case component growth is 1 + sqrt(2) ≈ 2.414x (~1.27 bits).
    """
    rows, prev = [], None
    for s, row in enumerate(stage_values):
        peak = float(max(np.max(np.abs(row.real)), np.max(np.abs(row.imag))))
        bits = required_int_bits(peak)
        rows.append({"stage": s, "max_component": peak, "int_bits_needed": bits,
                     "growth_bits": 0 if prev is None else bits - prev})
        prev = bits
    return rows


def bit_growth_profile(x) -> List[dict]:
    """Bit growth of the UNSCALED floating-point FFT of x (what an unscaled datapath must hold)."""
    return growth_profile(fft_radix2_dit(x).stage_values)


# --------------------------------------------------------------------------- #
# Fixed-point FFT simulation
# --------------------------------------------------------------------------- #
@dataclass
class FixedButterfly:
    stage: int
    index: int
    top: int
    bottom: int
    twiddle_k: int
    twiddle: complex        # ideal W_N^k
    twiddle_q: complex      # value actually stored in the twiddle ROM
    a_in: complex
    b_in: complex
    product: complex        # quantized multiplier output
    a_out: complex          # quantized (and scaled) adder outputs
    b_out: complex
    overflow: bool          # any register in this butterfly overflowed


@dataclass
class FixedFFTResult:
    N: int
    fmt: QFormat
    twiddle_fmt: QFormat
    scale_per_stage: bool
    input_quant: QuantResult
    stage_values: np.ndarray            # (S+1, N) fixed-point datapath contents
    reference_stage_values: np.ndarray  # (S+1, N) ideal values with the SAME scaling
    butterflies: List[List[FixedButterfly]] = field(default_factory=list)
    overflow_per_stage: List[int] = field(default_factory=list)  # overflowing components
    output: np.ndarray = None           # raw fixed-point output (scaled)
    output_scale: float = 1.0           # multiply output by this to undo the 1/2-per-stage scaling

    @property
    def rescaled_output(self) -> np.ndarray:
        return self.output * self.output_scale

    @property
    def total_overflows(self) -> int:
        return self.input_quant.n_overflow + sum(self.overflow_per_stage)

    def stage_snr_db(self) -> List[float]:
        """SNR of the datapath contents after every stage (noise accumulation)."""
        return [snr_db(r, f) for r, f in zip(self.reference_stage_values, self.stage_values)]

    def stage_max_error_lsb(self) -> List[float]:
        err = self.stage_values - self.reference_stage_values
        comp = np.maximum(np.abs(err.real), np.abs(err.imag))
        return list(np.max(comp, axis=1) / self.fmt.lsb)


def fixed_point_fft(x, fmt: QFormat, twiddle_fmt: Optional[QFormat] = None,
                    rounding: str = "round", overflow: str = "saturate",
                    scale_per_stage: bool = True) -> FixedFFTResult:
    """
    Bit-accurate model of a radix-2 DIT FFT with fixed-width registers.

    Per butterfly:
        P   = Q( W_q * B )                 complex multiplier, re-quantized
        A'  = Q( (A + P) * g )             g = 1/2 if scale_per_stage else 1
        B'  = Q( (A - P) * g )
    The adder is assumed to have one guard bit, so overflow is judged AFTER the
    optional shift — exactly as in most FPGA FFT IP cores.
    A parallel ideal path (unquantized input, exact arithmetic, same scaling)
    is computed for error comparisons.
    """
    x = np.asarray(x, dtype=complex).ravel()
    N = len(x)
    S = num_stages(N)
    twiddle_fmt = twiddle_fmt or default_twiddle_format(fmt.total_bits)
    g = 0.5 if scale_per_stage else 1.0

    q_in = quantize(x, fmt, rounding, overflow)
    perm = bit_reverse_permutation(N)
    data = q_in.values[perm].astype(complex)
    ideal = x[perm].copy()

    stage_vals = [data.copy()]
    ref_vals = [ideal.copy()]
    all_bf: List[List[FixedButterfly]] = []
    ovf_per_stage: List[int] = []

    for stage in butterfly_schedule(N):
        tops = np.array([e[3] for e in stage])
        bots = np.array([e[4] for e in stage])
        ks = np.array([e[5] for e in stage])

        W = twiddle_factor(ks, N)
        Wq = quantize(W, twiddle_fmt, rounding, "saturate").values

        # --- fixed-point path (vectorised; butterflies in a stage are independent)
        a, b = data[tops].copy(), data[bots].copy()
        p = quantize(Wq * b, fmt, rounding, overflow)
        s_ = quantize((a + p.values) * g, fmt, rounding, overflow)
        d_ = quantize((a - p.values) * g, fmt, rounding, overflow)
        data[tops], data[bots] = s_.values, d_.values

        # --- ideal path
        ia, ib = ideal[tops].copy(), ideal[bots].copy()
        it = W * ib
        ideal[tops], ideal[bots] = (ia + it) * g, (ia - it) * g

        ovf_any = p.overflow | s_.overflow | d_.overflow
        ovf_per_stage.append(p.n_overflow + s_.n_overflow + d_.n_overflow)
        all_bf.append([
            FixedButterfly(stage[i][0], i, int(tops[i]), int(bots[i]), int(ks[i]),
                           complex(W[i]), complex(Wq[i]), complex(a[i]), complex(b[i]),
                           complex(p.values[i]), complex(s_.values[i]), complex(d_.values[i]),
                           bool(ovf_any[i]))
            for i in range(len(stage))
        ])
        stage_vals.append(data.copy())
        ref_vals.append(ideal.copy())

    return FixedFFTResult(
        N=N, fmt=fmt, twiddle_fmt=twiddle_fmt, scale_per_stage=scale_per_stage,
        input_quant=q_in, stage_values=np.array(stage_vals),
        reference_stage_values=np.array(ref_vals), butterflies=all_bf,
        overflow_per_stage=ovf_per_stage, output=data.copy(),
        output_scale=float(2 ** S) if scale_per_stage else 1.0,
    )


def output_snr_db(x, result: FixedFFTResult) -> float:
    """SNR of the fixed-point spectrum against the exact floating-point FFT."""
    return snr_db(np.fft.fft(np.asarray(x, dtype=complex)), result.rescaled_output)


def snr_vs_bitwidth(x, bit_range, int_bits: int = 1, rounding: str = "round",
                    overflow: str = "saturate", scale_per_stage: bool = True) -> List[dict]:
    """Sweep the datapath word length and record input and output SNR."""
    rows = []
    for B in bit_range:
        if B <= int_bits:
            continue
        fmt = QFormat.from_bits(B, int_bits)
        r = fixed_point_fft(x, fmt, rounding=rounding, overflow=overflow,
                            scale_per_stage=scale_per_stage)
        rows.append({"bits": B,
                     "input_snr_db": snr_db(x, r.input_quant.values),
                     "output_snr_db": output_snr_db(x, r),
                     "overflows": r.total_overflows})
    return rows


if __name__ == "__main__":
    n = np.arange(16)
    sig = 0.9 * np.sin(2 * np.pi * 3 * n / 16)
    for B in (8, 12, 16, 24):
        r = fixed_point_fft(sig, QFormat.from_bits(B))
        print(f"{B:2d}-bit  output SNR = {output_snr_db(sig, r):6.2f} dB  overflows = {r.total_overflows}")