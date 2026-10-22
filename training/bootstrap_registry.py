"""Bootstrap the model registry from the model currently loaded by the runtime.

Use this once when introducing the registry to an existing deployment. The
operation is idempotent: rerunning it does not create duplicate model versions.
It refuses to overwrite a different PRODUCTION model because that would hide a
real runtime/registry consistency problem.
"""

from sqlalchemy import text

from app.database import get_engine
from app.ml import runtime


GET_VERSION = text("""
SELECT model_version, status, artifact_path, threshold
FROM ml_model_registry
WHERE model_version = :version
""")

GET_PRODUCTION = text("""
SELECT model_version
FROM ml_model_registry
WHERE status = 'PRODUCTION'
LIMIT 1
""")

INSERT_PRODUCTION = text("""
INSERT INTO ml_model_registry (
    model_version,
    artifact_path,
    status,
    threshold,
    validation_recall,
    validation_precision,
    validation_accuracy,
    training_rows,
    promoted_at
) VALUES (
    :version,
    :artifact_path,
    'PRODUCTION',
    :threshold,
    NULL,
    NULL,
    NULL,
    0,
    CURRENT_TIMESTAMP
)
""")


def main() -> None:
    loaded = runtime.current()

    with get_engine().begin() as connection:
        existing = connection.execute(
            GET_VERSION, {"version": loaded.version}
        ).mappings().first()

        if existing:
            if existing["status"] != "PRODUCTION":
                raise RuntimeError(
                    f"Runtime model {loaded.version} already exists in registry "
                    f"with status={existing['status']}; refusing to change it automatically"
                )
            print(f"Registry already contains runtime model {loaded.version} as PRODUCTION")
            return

        production = connection.execute(GET_PRODUCTION).scalar_one_or_none()
        if production is not None:
            raise RuntimeError(
                f"Registry says {production} is PRODUCTION but runtime loaded {loaded.version}. "
                "Resolve this inconsistency explicitly before bootstrapping."
            )

        connection.execute(INSERT_PRODUCTION, {
            "version": loaded.version,
            "artifact_path": loaded.artifact_path,
            "threshold": loaded.threshold,
        })

    print(f"Bootstrapped runtime model {loaded.version} as PRODUCTION")
    print(f"Artifact: {loaded.artifact_path}")
    print(f"Threshold: {loaded.threshold:.4f}")
    print("Historical validation metrics are left NULL because the registry did not own that training run.")


if __name__ == "__main__":
    main()
