#!/usr/bin/env python3
"""Is it better to spend the candidate budget unevenly across subframes?

The ranked search scans a fixed depth everywhere. But the depth it *needs*
varies enormously: on most subframes the analytic ordering is right within a
few candidates, and on a minority the winner sits far down the list. A flat
budget therefore overspends on the easy ones and gives up early on the hard
ones. This asks whether predicting that depth per subframe, and reallocating
the same total budget accordingly, finds more.

It answers three questions in order, and the first two do not involve a model
at all:

1.  What does perfect allocation buy? Budget assigned from the *true* depth,
    which no predictor can beat. If this is small the idea is dead regardless
    of how good a model might be.
2.  What does a trivial heuristic buy? Allocation from how tightly the top
    scores are clustered — near-ties mean the ordering is unreliable, so scan
    deeper — using no training whatsoever.
3.  What does a learned predictor buy, trained on one set of artists and
    tested on another?

Consumes the same dumps as bench/rank_study.py (build with
-DFLACOUT_DUMP_CANDIDATES, run with -c 100000 -p 0).
"""

import argparse
import sys

import numpy as np

FIELDS = ["sf", "win", "ord", "bsize", "bps", "lderr0", "lderr_ord", "wsq",
          "zf", "model_bps", "raw_bps", "var_raw", "score", "cost",
          "lderr_prev", "lderr_next", "lderr_1", "lderr_max", "max_ord"]


def load(path):
    with open(path) as fh:
        skip = 1 if fh.readline().startswith("sf\t") else 0
    arr = np.genfromtxt(path, delimiter="\t", dtype=float, skip_header=skip,
                        invalid_raise=False)
    return {n: arr[:, i] for i, n in enumerate(FIELDS)}


def group(r):
    """Per subframe: scores and costs in analytic order, plus block features.

    Everything here is knowable before the scan starts — the scores are what
    the ordering is built from — except `cost`, which is the ground truth the
    policies are scored against.
    """
    idx = np.argsort(r["sf"], kind="stable")
    sf = r["sf"][idx]
    bounds = np.flatnonzero(np.diff(sf)) + 1
    out = []
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(sf)]):
        sl = idx[a:b]
        sc, co = r["score"][sl], r["cost"][sl]
        o = np.argsort(sc, kind="stable")
        sc, co = sc[o], co[o]
        n = len(sc)
        # Score spread near the head, normalised by block size so it is in
        # bits-per-sample and comparable across subframes. A flat head means
        # the ordering is guessing.
        bs = r["bsize"][sl][0]
        top = sc[:min(16, n)] / bs
        feats = [
            bs, r["bps"][sl][0], np.log2(max(r["var_raw"][sl][0], 1e-30)), n,
            (top[-1] - top[0]) if len(top) > 1 else 0.0,
            top.std() if len(top) > 1 else 0.0,
            (sc[1] - sc[0]) / bs if n > 1 else 0.0,
            (sc[min(8, n - 1)] - sc[0]) / bs if n > 1 else 0.0,
            r["zf"][sl][o][0],
            r["ord"][sl][o][0],
        ]
        out.append((np.array(feats), sc, co))
    return out


def loss(groups, budgets):
    """Total bits above optimum when subframe i scans `budgets[i]` candidates."""
    tot = 0
    for (_, _, co), b in zip(groups, budgets):
        b = max(1, min(int(b), len(co)))
        tot += co[:b].min() - co.min()
    return tot


def allocate(need, mean_budget, ncands, lo=1):
    """Spread a fixed total budget in proportion to predicted difficulty.

    Proportional to the predicted depth itself, not to its rank: a subframe
    predicted to need 100 candidates should get roughly ten times what one
    predicted to need 10 gets, and a rank-based ramp cannot express that. The
    clip-and-redistribute loop keeps the total honest when subframes hit their
    own candidate count.
    """
    need = np.maximum(np.asarray(need, dtype=float), 1e-9)
    nc = np.asarray(ncands, dtype=float)
    total = float(mean_budget) * len(need)
    b = need * (total / need.sum())
    for _ in range(8):
        b = np.clip(b, lo, nc)
        deficit = total - b.sum()
        if abs(deficit) < 0.5:
            break
        room = nc - b
        free = room > 1e-9
        if not free.any():
            break
        b[free] += deficit * (need[free] / need[free].sum())
    return np.clip(b, lo, nc).astype(int)


def oracle_allocate(groups, mean_budget, lo=1):
    """Best possible split of a fixed budget, by marginal bits per candidate.

    Each subframe's loss falls in steps as it is scanned deeper, so this is a
    knapsack: repeatedly spend the next candidate wherever it buys the most
    bits. This is the ceiling for *any* allocation policy, perfect predictions
    included, and it is what tells us whether the idea is worth a model.
    """
    import heapq
    n = len(groups)
    total = int(mean_budget * n)
    b = [lo] * n
    curves = []
    for _, _, co in groups:
        run = np.minimum.accumulate(co)          # best cost within depth b
        curves.append(run - co.min())            # bits still lost at depth b
    heap = []
    for i, cur in enumerate(curves):
        if lo < len(cur):
            # gain from the next candidate, and how many it takes to get it
            j = lo
            while j < len(cur) and cur[j] == cur[lo - 1]:
                j += 1
            if j < len(cur):
                heapq.heappush(heap, (-(cur[lo - 1] - cur[j]) / (j - lo + 1), i, j))
    spent = lo * n
    while heap and spent < total:
        rate, i, j = heapq.heappop(heap)
        cost = j - b[i] + 1
        if spent + cost > total:
            continue
        spent += cost
        b[i] = j + 1
        cur = curves[i]
        k = b[i]
        while k < len(cur) and cur[k] == cur[b[i] - 1]:
            k += 1
        if k < len(cur):
            heapq.heappush(heap, (-(cur[b[i] - 1] - cur[k]) / (k - b[i] + 1), i, k))
    return np.array(b)


def report(name, groups, budgets, base, mean_b):
    l = loss(groups, budgets)
    print("  %-30s %12d %9.1f%%  (mean depth %.1f)"
          % (name, l, 100.0 * (base - l) / base if base else 0.0,
             np.mean(budgets)))
    return l


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--budgets", default="8,16,32")
    args = ap.parse_args()

    tr = group(load(args.train))
    te = group(load(args.test))
    print("train subframes: %d   test subframes: %d" % (len(tr), len(te)))

    # Depth actually needed: the first position at which the scan has the
    # cheapest candidate in hand.
    def depth(g):
        return np.array([int(np.argmin(co)) + 1 for _, _, co in g])

    dtr, dte = depth(tr), depth(te)
    print("true depth needed — train median %d p90 %d | test median %d p90 %d"
          % (np.median(dtr), np.percentile(dtr, 90),
             np.median(dte), np.percentile(dte, 90)))

    Xtr = np.array([f for f, _, _ in tr])
    Xte = np.array([f for f, _, _ in te])
    ncte = [len(co) for _, _, co in te]

    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(max_iter=300, random_state=0)
    model.fit(Xtr, np.log1p(dtr))
    pred = model.predict(Xte)

    def recoverable(groups, B):
        """Bits a subframe would still gain if scanned past a flat budget B.

        This, not depth, is what the oracle maximises: a subframe that needs
        depth 100 to save 2 bits should be starved, and one that saves 400 bits
        at depth 12 should be funded. Predicting depth funds the former.
        """
        return np.array([float(co[:min(B, len(co))].min() - co.min())
                         for _, _, co in groups])

    # Heuristic with no training: how tightly the top scores are clustered.
    # Column 6 is (score[1]-score[0]) per sample; a small gap means the top two
    # are near-tied and the ordering is not to be trusted.
    # Small gap between the top two scores => the ordering is guessing there,
    # so scan deeper. Inverted into a "difficulty" so it feeds allocate().
    gap = np.maximum(Xte[:, 6], 1e-9)
    tie = 1.0 / gap

    for B in [int(b) for b in args.budgets.split(",")]:
        flat = np.full(len(te), B)
        base = loss(te, flat)
        print("\n=== mean budget %d (total %d candidate evaluations) ==="
              % (B, B * len(te)))
        print("  %-30s %12s %9s" % ("policy", "bits lost", "vs flat"))
        report("flat (what ships today)", te, flat, base, B)
        report("oracle allocation (ceiling)", te,
               oracle_allocate(te, B), base, B)
        report("tie-gap heuristic, no training", te,
               allocate(tie, B, ncte), base, B)
        report("learned depth predictor", te,
               allocate(np.expm1(pred), B, ncte), base, B)

        # Predict recoverable *bits* and fund top-down: every subframe gets a
        # floor, and the remainder goes to whoever is predicted to gain most,
        # scanned as deep as the pool allows. This mirrors the oracle's own
        # objective rather than a proxy for it.
        vmod = HistGradientBoostingRegressor(max_iter=300, random_state=0)
        vmod.fit(Xtr, np.log1p(recoverable(tr, B)))
        vpred = np.expm1(vmod.predict(Xte))
        floor = max(1, B // 4)
        pool = B * len(te) - floor * len(te)
        bud = np.full(len(te), floor, dtype=float)
        for i in np.argsort(-vpred):
            if pool <= 0:
                break
            take = min(pool, max(0, ncte[i] - floor))
            bud[i] += take
            pool -= take
        report("learned recoverable-bits", te, bud, base, B)

        # What actually ships: react instead of predicting. Keep scanning
        # while candidates are still improving the best exact cost, stop after
        # `p` consecutive that are not. This spends the same kind of uneven
        # budget the oracle does, but decides using what the scan has already
        # revealed rather than guessing beforehand — which is strictly more
        # information than any pre-scan feature can carry.
        for p_ in (B, 2 * B):
            depths = []
            for _, _, co in te:
                best = np.inf
                since = 0
                d = 0
                for j, c in enumerate(co):
                    d = j + 1
                    if c < best:
                        best = c
                        since = 0
                    else:
                        since += 1
                        if d >= B and since >= p_:
                            break
                depths.append(d)
            depths = np.array(depths)
            report("reactive patience p=%d" % p_, te, depths, base, B)


if __name__ == "__main__":
    sys.exit(main())
