#!/usr/bin/env python3
"""Can a learned cost model rank LPC candidates better than the analytic one?

The ranked search fully evaluates only the first few candidates of an ordering
produced by a Gaussian-entropy estimate, so everything it wins or loses comes
down to one question: how far down that ordering does the eventual winner sit?
This measures that, then asks whether a model trained on the encoder's own
(features -> exact cost) pairs orders them better.

Feed it dumps from a build with -DFLACOUT_DUMP_CANDIDATES, produced with a
candidate limit high enough that nothing is cut, e.g.

    FLACOUT_DUMP_PATH=train.tsv ./flacoutcpp -q -n -R -c 100000 -p 0 in.flac out.flac

Train and test must be different audio — ideally different artists. A model
that has seen the test material will flatter itself, and the whole point is to
find out whether the ordering generalises.

What matters is *rank of the winner*, not prediction accuracy: a model with
worse RMSE that puts the winner at rank 0 more often is the better model here,
because the encoder never sees the predictions, only the order they impose.
"""

import argparse
import math
import sys

import numpy as np

FIELDS = ["sf", "win", "ord", "bsize", "bps", "lderr0", "lderr_ord", "wsq",
          "zf", "model_bps", "raw_bps", "var_raw", "score", "cost",
          "lderr_prev", "lderr_next", "lderr_1", "lderr_max", "max_ord"]


def load(path):
    # Explicit float dtype: letting genfromtxt infer gives string columns the
    # moment a header line is present, and every column here is numeric.
    with open(path) as fh:
        skip = 1 if fh.readline().startswith("sf\t") else 0
    arr = np.genfromtxt(path, delimiter="\t", dtype=float, skip_header=skip,
                        invalid_raise=False)
    return {name: arr[:, i] for i, name in enumerate(FIELDS)}


def features(r):
    """Everything derivable at ranking time, before any exact cost is known.

    Ratios and logs rather than raw magnitudes: the absolute scale of an
    autocorrelation depends on the signal's loudness, which says nothing about
    which candidate wins, and a tree splitting on raw energy would be learning
    the corpus's mastering rather than the encoder.
    """
    eps = 1e-30
    lderr0 = np.maximum(r["lderr0"], eps)
    lderr_o = np.maximum(r["lderr_ord"], eps)
    wsq = np.maximum(r["wsq"], eps)
    n = r["bsize"]

    prev = np.maximum(r["lderr_prev"], eps)
    nxt = np.maximum(r["lderr_next"], eps)
    e1 = np.maximum(r["lderr_1"], eps)
    emax = np.maximum(r["lderr_max"], eps)

    # The Levinson recursion produces an error at *every* order and the scorer
    # keeps one number from it. These are the discarded shape:
    #   k2   reflection coefficient at this order, |k|^2 = 1 - E_m/E_{m-1};
    #        how much this order just bought.
    #   gain_next  whether the next order is still buying anything, i.e.
    #        whether we are at the knee of the curve or past it.
    #   head_room  distance to the best any order reaches, so a candidate can
    #        be judged against its own window's ceiling rather than absolutely.
    #   pred1  first-order predictability, a cheap proxy for how tonal the
    #        block is.
    k2 = np.clip(1.0 - lderr_o / prev, 0.0, 1.0)
    gain_next = np.log2(nxt / lderr_o)
    head_room = np.log2(lderr_o / emax)
    pred1 = np.log2(e1 / lderr0)

    var_e = lderr_o / wsq
    return np.column_stack([
        k2,
        gain_next,
        head_room,
        pred1,
        np.log2(prev / lderr_o),                   # gain this order bought
        r["max_ord"],
        r["ord"] / np.maximum(r["max_ord"], 1.0),  # position along the curve
        r["ord"],
        n,
        r["bps"],
        r["zf"],
        np.log2(np.maximum(var_e, eps)),           # what the current model uses
        np.log2(lderr_o / lderr0),                 # fraction of energy left
        np.log2(np.maximum(r["var_raw"], eps)),
        r["model_bps"],
        r["raw_bps"],
        r["model_bps"] - r["raw_bps"],             # how much the window claims
        r["ord"] / n,                # coefficient overhead share
        r["win"],                    # window identity
        r["score"] / n,              # current score, normalised
    ])


def winner_ranks(order_key, sf, cost):
    """For each subframe, the position of the true cheapest candidate when
    candidates are sorted by `order_key` ascending."""
    ranks = []
    idx = np.argsort(sf, kind="stable")
    sf_s, key_s, cost_s = sf[idx], order_key[idx], cost[idx]
    bounds = np.flatnonzero(np.diff(sf_s)) + 1
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(sf_s)]):
        k, c = key_s[a:b], cost_s[a:b]
        order = np.argsort(k, kind="stable")
        best = np.argmin(c[order])
        ranks.append(best)
    return np.array(ranks)


def winner_ranks_pos(sf, key):
    """Position of each candidate within its subframe under `key` ascending."""
    pos = np.empty(len(sf), dtype=np.int64)
    idx = np.argsort(sf, kind="stable")
    sf_s = sf[idx]
    bounds = np.flatnonzero(np.diff(sf_s)) + 1
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(sf_s)]):
        sl = idx[a:b]
        order = np.argsort(key[sl], kind="stable")
        pos[sl[order]] = np.arange(b - a)
    return pos


def summarise(name, ranks, budgets):
    out = ["%-22s n=%d  median=%d  mean=%.1f  p95=%d"
           % (name, len(ranks), int(np.median(ranks)), ranks.mean(),
              int(np.percentile(ranks, 95)))]
    hit = "  ".join("<=%d: %5.1f%%" % (b, 100.0 * (ranks < b).mean())
                    for b in budgets)
    out.append("  " + hit)
    return "\n".join(out)


def bytes_left(sf, cost, ranks_by_order, budget):
    """Bits lost per subframe by stopping after `budget` candidates, summed.

    This is the quantity the encoder actually pays: the gap between the
    cheapest candidate inside the budget and the cheapest overall.
    """
    idx = np.argsort(sf, kind="stable")
    sf_s, cost_s = sf[idx], cost[idx]
    key_s = ranks_by_order[idx]
    bounds = np.flatnonzero(np.diff(sf_s)) + 1
    lost = 0
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(sf_s)]):
        k, c = key_s[a:b], cost_s[a:b]
        order = np.argsort(k, kind="stable")
        got = c[order][:budget].min()
        lost += got - c.min()
    return lost


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--budgets", default="1,4,8,16,32")
    ap.add_argument("--trees", type=int, default=300)
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]

    tr, te = load(args.train), load(args.test)
    print("train: %d candidates / %d subframes"
          % (len(tr["sf"]), len(np.unique(tr["sf"]))))
    print("test:  %d candidates / %d subframes"
          % (len(te["sf"]), len(np.unique(te["sf"]))))

    Xtr, Xte = features(tr), features(te)

    # Learn the analytic score's *error*, not the cost itself. Predicting cost
    # outright asks the model to rebuild a function the encoder already has a
    # decent closed form for, and it optimises squared error over the whole
    # corpus — which is not the objective. Ordering is, and ordering only ever
    # compares candidates inside one subframe. Fitting the residual keeps the
    # incumbent's structure and spends the model's capacity on where it is
    # wrong; ranking by score + correction can then only differ from the
    # analytic ordering where the model has something to say.
    #
    # Per sample, because a residual in bits scales with block size and the
    # model would otherwise re-learn block size as a proxy for magnitude.
    ytr = (tr["cost"] - tr["score"]) / tr["bsize"]

    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(max_iter=args.trees, learning_rate=0.1,
                                          random_state=0)
    model.fit(Xtr, ytr)
    pred = te["score"] + model.predict(Xte) * te["bsize"]

    cur = winner_ranks(te["score"], te["sf"], te["cost"])
    new = winner_ranks(pred, te["sf"], te["cost"])
    print("\n=== rank of the true winner (held-out audio) ===")
    print(summarise("analytic score", cur, budgets))
    print(summarise("learned model", new, budgets))

    print("\n=== bits left on the table at a given budget ===")
    print("%-10s %14s %14s %9s" % ("budget", "analytic", "learned", "recovered"))
    for b in budgets:
        la = bytes_left(te["sf"], te["cost"], te["score"], b)
        ln = bytes_left(te["sf"], te["cost"], pred, b)
        rec = (100.0 * (la - ln) / la) if la else 0.0
        print("%-10d %14d %14d %8.1f%%" % (b, la, ln, rec))

    # The two orderings fail in different places, so try using each where it
    # is strong: the analytic score for the head of the list, the model to
    # order everything after it. If the encoder's scan is going to run past
    # the head anyway (which patience makes it do), only the tail ordering
    # decides what it finds there.
    print("\n=== hybrid: analytic for the first K, model after ===")
    print("%-8s %-8s %14s %9s" % ("K", "budget", "bits left", "vs analytic"))
    for K in (4, 8, 16):
        for b in (16, 32, 64):
            if b <= K:
                continue
            hy = np.where(winner_ranks_pos(te["sf"], te["score"]) < K,
                          te["score"], 1e18 + pred)
            lh = bytes_left(te["sf"], te["cost"], hy, b)
            la = bytes_left(te["sf"], te["cost"], te["score"], b)
            print("%-8d %-8d %14d %8.1f%%"
                  % (K, b, lh, 100.0 * (la - lh) / la if la else 0.0))

    idx = np.argsort(te["sf"], kind="stable")
    sf_s, cost_s = te["sf"][idx], te["cost"][idx]
    bounds = np.flatnonzero(np.diff(sf_s)) + 1
    best_total = sum(cost_s[a:b].min()
                     for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(sf_s)]))
    print("\noracle total (best candidate every subframe): %d bits" % best_total)
    print("for scale, 1%% of that is %d bits = %d bytes"
          % (best_total // 100, best_total // 800))


if __name__ == "__main__":
    sys.exit(main())
