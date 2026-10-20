import argparse
import json

from app.monitoring import get_model_performance


def main() -> int:
    parser = argparse.ArgumentParser(description="Check recent production model quality")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--min-labeled", type=int, default=30)
    parser.add_argument("--min-recall", type=float, default=0.70)
    parser.add_argument("--model-version", default=None)
    args = parser.parse_args()

    metrics = get_model_performance(args.days, args.model_version)
    print(json.dumps(metrics, indent=2))

    if metrics["labeled_count"] < args.min_labeled:
        print("STATUS=INSUFFICIENT_DATA")
        return 0

    recall = metrics["recall"]
    if recall is not None and recall < args.min_recall:
        print("STATUS=RETRAIN_REVIEW_REQUIRED")
        return 2

    print("STATUS=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
