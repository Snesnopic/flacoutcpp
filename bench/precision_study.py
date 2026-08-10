#!/usr/bin/env python3
"""Can the LPC precision ladder be priced analytically instead of run?

Every candidate the ranked scan admits is fully encoded at all 8 precisions
(p = 8..15) -- 8 residual+Rice passes, and the header-based break never fires
in practice (measured 25480 = 8 x 3185 rungs entered, 0 pruned). That factor of
8 is the largest single multiplier on the cost of a candidate, and candidates
are what `-c`/`-p` buy. Removing it converts directly into search depth, which
is where the compression is: flat -c 8 leaves 0.114% of subframe bits on the
table, depth 23 leaves 0.077%, depth 85 leaves 0.050%.

The claim under test. For a predictor c, the windowed residual energy is

    E(c) = E_a + (c - a)' R (c - a)

with `a` the Levinson solution and R the windowed autocorrelation -- both
already computed for the candidate. So a rung's residual energy is an
O(order^2) quadratic form in the quantization error, not an O(bsize*order)
pass, and the rung's *total* cost is that plus order*precision bits of
coefficients, which is exact. If the argmin of that model matches the argmin of
the true cost often enough, the ladder can be predicted and only the winning
rung (or the best two) actually encoded.

Feed it dumps from a build with -DFLACOUT_DUMP_PRECISION:

    FLACOUT_PREC_DUMP_PATH=p.tsv ./flacoutcpp -q -n -R in.flac out.flac

What matters is not the model's accuracy on a rung but the *bits* a policy
built from it gives up, measured at the subframe winner -- an error on a rung
of a candidate that loses anyway is free.
"""

import argparse
import sys

import numpy as np

FIELDS = ["sf", "win", "ord", "bsize", "bps", "prec", "shift", "clamped",
          "cost", "ea", "drd", "wsq", "r0", "sd2", "drd1", "drd4"]


def load(paths):
    """Rows from every dump, with subframe ids made unique across files."""
    cols, off = [], 0
    for p in paths:
        with open(p) as fh:
            skip = 1 if fh.readline().startswith("sf\t") else 0
        a = np.genfromtxt(p, delimiter="\t", dtype=float, skip_header=skip,
                          invalid_raise=False)
        a[:, 0] += off
        off = a[:, 0].max() + 1
        cols.append(a)
    a = np.vstack(cols)
    return {n: a[:, i] for i, n in enumerate(FIELDS)}


def candidates(r):
    """Group rows into (subframe, window, order) ladders.

    Each ladder is one candidate's 8 rungs: the precisions, their true costs,
    and the analytic prediction's inputs.
    """
    key = (r["sf"].astype(np.int64) << 20) \
        | (r["win"].astype(np.int64) << 8) | r["ord"].astype(np.int64)
    idx = np.lexsort((r["prec"], key))
    k = key[idx]
    bounds = np.flatnonzero(np.diff(k)) + 1
    out = []
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(k)]):
        sl = idx[a:b]
        if len(sl) < 2:
            continue
        c = dict(sf=int(r["sf"][sl][0]), ord=int(r["ord"][sl][0]),
                 bsize=float(r["bsize"][sl][0]), r0=r["r0"][sl][0])
        for n in ("prec", "cost", "ea", "drd", "sd2", "drd1", "drd4", "shift"):
            c[n] = r[n][sl]
        out.append(c)
    return out


def variants():
    """Cheaper stand-ins for d'Rd, and the slope of the entropy term.

    The quadratic form is O(order^2) per rung. Everything here asks whether a
    coarser estimate ranks the ladder as well for less -- including the one
    that would need no quantization at all, treating the quantizer's error as
    uniform white noise of power (order/12)*4^-shift.
    """
    def cost(drd, slope=0.5):
        return lambda c: (c["ord"] * c["prec"]
                          + slope * (c["bsize"] - c["ord"])
                          * np.log2(np.maximum(c["ea"] + drd(c), 1e-300)))
    return [
        ("exact d'Rd                O(ord^2)", cost(lambda c: c["drd"])),
        ("exact d'Rd, slope 0.40    O(ord^2)", cost(lambda c: c["drd"], 0.40)),
        ("band |i-j|<=4             O(4*ord)", cost(lambda c: np.maximum(c["drd4"], 1e-300))),
        ("band |i-j|<=1             O(2*ord)", cost(lambda c: np.maximum(c["drd1"], 1e-300))),
        ("diagonal r0*sum d^2       O(ord)  ", cost(lambda c: c["r0"] * c["sd2"])),
        ("white noise, no quantize  O(1)    ",
         cost(lambda c: c["r0"] * c["ord"] / 12.0 * np.power(4.0, -c["shift"]))),
        ("coefficient term only             ", lambda c: c["ord"] * c["prec"]),
        ("energy term only                  ",
         lambda c: np.log2(np.maximum(c["ea"] + c["drd"], 1e-300))),
    ]


def predict(c):
    """Analytic cost of each rung, up to terms constant across the ladder.

    order*precision is the exact coefficient cost; the rest is the Gaussian
    entropy of the predicted residual energy. The window-energy normalisation
    and the subframe header are the same for every rung, so they are dropped --
    only the argmin matters.
    """
    e = np.maximum(c["ea"] + c["drd"], 1e-300)
    return c["ord"] * c["prec"] + 0.5 * (c["bsize"] - c["ord"]) * np.log2(e)


def subframe_losses(cands, pick):
    """Bits given up per subframe when each candidate evaluates `pick(c)` rungs.

    The subframe keeps the cheapest cost it actually evaluated; the reference
    is the cheapest over the whole ladder of every candidate. Errors on a
    candidate that loses anyway cost nothing, which is the point.
    """
    best_full, best_pol = {}, {}
    for c in cands:
        sf = c["sf"]
        full = c["cost"].min()
        pol = c["cost"][pick(c)].min()
        best_full[sf] = min(best_full.get(sf, np.inf), full)
        best_pol[sf] = min(best_pol.get(sf, np.inf), pol)
    return (np.array([best_pol[s] - best_full[s] for s in best_full]),
            float(sum(best_full.values())))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+")
    args = ap.parse_args()

    r = load(args.dumps)
    cands = candidates(r)
    rungs = sum(len(c["prec"]) for c in cands)
    nsf = len({c["sf"] for c in cands})
    print("%d subframes, %d candidates, %d rungs (%.2f per candidate)"
          % (nsf, len(cands), rungs, rungs / len(cands)))

    # How often the true best rung is each precision, and how often the
    # analytic model names it.
    true_best = np.array([c["prec"][np.argmin(c["cost"])] for c in cands])
    pred_rank = [np.argsort(predict(c), kind="stable") for c in cands]
    pred_best = np.array([c["prec"][pr[0]] for c, pr in zip(cands, pred_rank)])
    hit = np.array([c["cost"][pr[0]] == c["cost"].min()
                    for c, pr in zip(cands, pred_rank)])
    print("\ntrue winning precision: " + " ".join(
        "%d:%.0f%%" % (p, 100.0 * np.mean(true_best == p)) for p in range(8, 16)))
    print("model's pick         : " + " ".join(
        "%d:%.0f%%" % (p, 100.0 * np.mean(pred_best == p)) for p in range(8, 16)))
    print("model names a rung of the true minimum cost: %.1f%% of candidates"
          % (100.0 * hit.mean()))

    print("\n%-34s %7s %11s %9s  %s"
          % ("policy", "rungs", "bits lost", "vs total", "of ladder's value"))
    full_l, total = subframe_losses(cands, lambda c: slice(None))
    assert full_l.sum() == 0

    # The ladder's own value: what a single fixed precision gives up. If this
    # is small the ladder is not worth predicting -- it is worth deleting.
    fixed = {}
    for p in range(8, 16):
        l, _ = subframe_losses(
            cands, lambda c, p=p: np.array([int(np.argmin(np.abs(c["prec"] - p)))]))
        fixed[p] = l.sum()
    bestp = min(fixed, key=fixed.get)
    ladder_value = fixed[bestp]

    def row(name, nrung, lost):
        print("%-34s %7.2f %11d %8.4f%% %8.1f%%"
              % (name, nrung, lost, 100.0 * lost / total,
                 100.0 * lost / ladder_value if ladder_value else 0.0))

    for p in (bestp, 15):
        row("fixed precision %d" % p, 1.0, fixed[p])
    for k in (1, 2, 3, 4):
        lost = 0.0
        best_full, best_pol = {}, {}
        for c, pr in zip(cands, pred_rank):
            sf = c["sf"]
            best_full[sf] = min(best_full.get(sf, np.inf), c["cost"].min())
            best_pol[sf] = min(best_pol.get(sf, np.inf), c["cost"][pr[:k]].min())
        lost = sum(best_pol[s] - best_full[s] for s in best_full)
        row("analytic top-%d rungs" % k, float(k), lost)
    row("full ladder (reference)", rungs / len(cands), 0)
    print("\ntotal subframe bits: %d" % total)

    # Which parts of the model are load-bearing. Both terms are: dropping
    # either loses more than a fixed precision does.
    print("\n%-36s %9s %9s %9s" % ("drd approximation / model", "top-1", "top-2", "top-3"))
    for name, fn in variants():
        line = []
        for k in (1, 2, 3):
            best_full, best_pol = {}, {}
            for c in cands:
                pr = np.argsort(fn(c), kind="stable")
                sf = c["sf"]
                best_full[sf] = min(best_full.get(sf, np.inf), c["cost"].min())
                best_pol[sf] = min(best_pol.get(sf, np.inf), c["cost"][pr[:k]].min())
            line.append(sum(best_pol[s] - best_full[s] for s in best_full))
        print("  %-34s %9d %9d %9d" % (name, line[0], line[1], line[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
