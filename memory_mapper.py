"""
memory_mapper.py  —  Person 3
=============================

Memory-addressing model of an in-place radix-2 DIT FFT running on a single
N-word DUAL-PORT RAM (port A + port B) with a twiddle-factor ROM.

Timeline modelled (one row of the access log = one clock cycle):

  LOAD    N/2 cycles : samples arrive in natural order, two per cycle, and are
                       written to their BIT-REVERSED addresses
                       (x[2c] -> port A, x[2c+1] -> port B).
  COMPUTE per butterfly, 2 cycles:
            READ  cycle : port A reads `top`, port B reads `bottom`,
                          twiddle ROM reads W_N^k
            WRITE cycle : results are written back IN PLACE to the same
                          `top` / `bottom` addresses.
  UNLOAD  N/2 cycles : X[k] is read out in natural order, two per cycle.

Because the two butterfly operands are always at different addresses and are
written back exactly where they were read from, the schedule is free of
port conflicts and read/write hazards. `check_collisions()` proves it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np

from fft_engine import bit_reverse, butterfly_schedule, num_stages


# --------------------------------------------------------------------------- #
# Bit reversal
# --------------------------------------------------------------------------- #
def bit_reversal_table(N: int) -> List[Dict]:
    """
    One row per input sample: natural index -> bit-reversed RAM address.
    Also flags which addresses must be swapped for an in-place reorder.
    """
    nbits = num_stages(N)
    rows = []
    for i in range(N):
        r = bit_reverse(i, nbits)
        rows.append({
            "Sample index n": i,
            "Binary (n)": format(i, f"0{nbits}b"),
            "Reversed bits": format(r, f"0{nbits}b"),
            "RAM address": r,
            "In-place action": "stays" if r == i else (f"swap with {r}" if i < r else f"(swapped by {r})"),
        })
    return rows


def in_place_swap_pairs(N: int) -> List[tuple]:
    """Pairs (i, j), i < j, that must be swapped to bit-reverse an array in place."""
    nbits = num_stages(N)
    return [(i, bit_reverse(i, nbits)) for i in range(N) if i < bit_reverse(i, nbits)]


def output_mapping(N: int) -> List[Dict]:
    """For DIT with bit-reversed input, the output X[k] sits at address k (natural order)."""
    return [{"RAM address": k, "Holds": f"X[{k}]"} for k in range(N)]


# --------------------------------------------------------------------------- #
# Stage geometry
# --------------------------------------------------------------------------- #
def stage_parameters(N: int) -> List[Dict]:
    """Span, partner distance, twiddle stride and butterfly count per stage."""
    rows = []
    for s in range(1, num_stages(N) + 1):
        m = 1 << s
        rows.append({"Stage": s, "Sub-DFT size": m, "Partner distance": m // 2,
                     "Groups": N // m, "Butterflies/group": m // 2,
                     "Twiddle ROM stride": N // m,
                     "Twiddle ROM addresses": sorted({j * (N // m) for j in range(m // 2)})})
    return rows


def stage_partner_matrix(N: int) -> Dict[str, np.ndarray]:
    """
    Matrices of shape (S, N): for every stage and address, the address it is
    paired with, the group id, and the twiddle exponent applied to that node.
    """
    S = num_stages(N)
    partner = np.zeros((S, N), dtype=int)
    group = np.zeros((S, N), dtype=int)
    tw = np.zeros((S, N), dtype=int)
    for st in butterfly_schedule(N):
        for (s, g, j, top, bot, k) in st:
            partner[s - 1, top], partner[s - 1, bot] = bot, top
            group[s - 1, top] = group[s - 1, bot] = g
            tw[s - 1, top] = tw[s - 1, bot] = k
    return {"partner": partner, "group": group, "twiddle_k": tw}


def memory_contents(N: int) -> List[List[str]]:
    """
    Human-readable description of what each RAM address holds after the load
    and after every stage: after stage s the RAM holds N/2^s independent
    2^s-point DFTs; after the last stage it holds X[0..N-1].
    """
    S = num_stages(N)
    out = [[f"x[{bit_reverse(a, S)}]" for a in range(N)]]
    for s in range(1, S + 1):
        size = 1 << s
        if s == S:
            out.append([f"X[{a}]" for a in range(N)])
        else:
            out.append([f"DFT{size}#{a // size} bin {a % size}" for a in range(N)])
    return out


# --------------------------------------------------------------------------- #
# Cycle-by-cycle access log
# --------------------------------------------------------------------------- #
def generate_access_schedule(N: int, include_io: bool = True) -> List[Dict]:
    """Return the full dual-port RAM access log, one dict per clock cycle."""
    S = num_stages(N)
    rows: List[Dict] = []
    cycle = 0

    if include_io:
        for c in range(N // 2):
            i0, i1 = 2 * c, 2 * c + 1
            rows.append({"cycle": cycle, "phase": "LOAD", "stage": 0, "butterfly": None,
                         "op": "WRITE", "port_A_addr": bit_reverse(i0, S),
                         "port_B_addr": bit_reverse(i1, S), "twiddle_ROM_addr": None,
                         "note": f"x[{i0}] -> A, x[{i1}] -> B"})
            cycle += 1

    for stage in butterfly_schedule(N):
        for idx, (s, g, j, top, bot, k) in enumerate(stage):
            common = {"phase": f"STAGE {s}", "stage": s, "butterfly": idx,
                      "port_A_addr": top, "port_B_addr": bot}
            rows.append({"cycle": cycle, **common, "op": "READ", "twiddle_ROM_addr": k,
                         "note": f"read operands, fetch W_{N}^{k}"})
            cycle += 1
            rows.append({"cycle": cycle, **common, "op": "WRITE", "twiddle_ROM_addr": None,
                         "note": "A+WB -> top, A-WB -> bottom"})
            cycle += 1

    if include_io:
        for c in range(N // 2):
            rows.append({"cycle": cycle, "phase": "UNLOAD", "stage": S + 1, "butterfly": None,
                         "op": "READ", "port_A_addr": 2 * c, "port_B_addr": 2 * c + 1,
                         "twiddle_ROM_addr": None,
                         "note": f"X[{2 * c}] <- A, X[{2 * c + 1}] <- B"})
            cycle += 1
    return rows


def check_collisions(schedule: List[Dict], N: int) -> Dict:
    """
    Verify the access log against dual-port RAM rules:
      1. one access per port per cycle and port A != port B in the same cycle,
      2. LOAD writes every address exactly once (bit reversal is a permutation),
      3. in every stage each address is read exactly once and written exactly once,
      4. no write lands on an address before that address was read in the
         same stage (no write-after-read hazard / data clobbering).
    """
    issues: List[str] = []
    cycles_seen = set()

    for r in schedule:
        if r["cycle"] in cycles_seen:
            issues.append(f"cycle {r['cycle']}: more than one access per port")
        cycles_seen.add(r["cycle"])
        if r["port_A_addr"] == r["port_B_addr"]:
            issues.append(f"cycle {r['cycle']}: both ports hit address {r['port_A_addr']}")

    by_stage = defaultdict(list)
    for r in schedule:
        by_stage[r["phase"]].append(r)

    load = by_stage.get("LOAD", [])
    if load:
        addrs = [a for r in load for a in (r["port_A_addr"], r["port_B_addr"])]
        if sorted(addrs) != list(range(N)):
            issues.append("LOAD phase does not write each address exactly once")

    for s in range(1, num_stages(N) + 1):
        rows = by_stage.get(f"STAGE {s}", [])
        reads, writes = defaultdict(list), defaultdict(list)
        for r in rows:
            target = reads if r["op"] == "READ" else writes
            target[r["port_A_addr"]].append(r["cycle"])
            target[r["port_B_addr"]].append(r["cycle"])
        for a in range(N):
            if len(reads[a]) != 1 or len(writes[a]) != 1:
                issues.append(f"stage {s}: address {a} read {len(reads[a])}x, written {len(writes[a])}x")
            elif writes[a][0] <= reads[a][0]:
                issues.append(f"stage {s}: address {a} written before it was read")

    checks = [
        ("Port A and port B never address the same word in one cycle",
         not any("both ports" in i for i in issues)),
        ("Bit-reversed load is a perfect permutation",
         not any("LOAD" in i for i in issues)),
        ("Every address read once and written once per stage",
         not any("read" in i and "x," in i for i in issues)),
        ("No write-before-read (in-place hazard) in any stage",
         not any("before it was read" in i for i in issues)),
    ]
    return {"ok": not issues, "issues": issues, "checks": checks,
            "total_cycles": len(schedule)}


def cycle_count_summary(N: int) -> Dict:
    """Latency breakdown for this simple non-pipelined architecture."""
    S = num_stages(N)
    bf = (N // 2) * S
    return {"load_cycles": N // 2, "butterflies": bf, "compute_cycles": 2 * bf,
            "unload_cycles": N // 2, "total_cycles": N + 2 * bf}


if __name__ == "__main__":
    for n in (8, 16, 64):
        sched = generate_access_schedule(n)
        res = check_collisions(sched, n)
        print(f"N={n}: {res['total_cycles']} cycles, collision-free = {res['ok']}")