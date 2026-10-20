import argparse

from app.model_registry import promote_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote or roll back a registered model")
    parser.add_argument("model_version")
    args = parser.parse_args()

    if not promote_model(args.model_version):
        print(f"Cannot promote {args.model_version}: version is missing or already production")
        return 1

    print(f"Production model is now {args.model_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
