"""Paired McNemar test between two prediction dumps over the same test set.

Usage: python scripts/run_mcnemar.py <candidate.npz> <baseline.npz>
n10 = baseline wrong & candidate right; n01 = the reverse.
"""
import argparse
import json

from devnet.evaluate import load_predictions, mcnemar_between, metrics_from_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("baseline")
    args = ap.parse_args()

    result = mcnemar_between(args.candidate, args.baseline)
    for name, path in [("candidate", args.candidate), ("baseline", args.baseline)]:
        _, preds, labels = load_predictions(path)
        m = metrics_from_predictions(preds, labels)
        result[f"{name}_accuracy"] = m["accuracy"]
        result[f"{name}_n_errors"] = m["n_errors"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
