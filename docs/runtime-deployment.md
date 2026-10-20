# Runtime model deployment

V6 closes the gap between the SQL model registry and the model actually serving predictions.

## Flow

Candidate artifact -> validate -> promote registry row -> POST `/models/reload-production` -> load artifact into memory -> validate interface -> atomic reference swap -> new requests use the new model.

## Why load before swap

The runtime loads the candidate artifact and verifies that it exposes `predict_proba` before acquiring the swap lock. If loading fails, the currently serving model reference is unchanged. This is the key safety property for runtime rollback/deployment.

## Runtime inspection

`GET /models/runtime` returns the exact model version, threshold and artifact path currently loaded by this API process.

Do not infer runtime state only from the registry. Registry state describes the intended production model; `/models/runtime` describes what this process is actually serving.

## Deploy

1. Train a candidate and register it.
2. Review validation metrics.
3. Promote it transactionally in the registry.
4. Ensure the versioned artifact path is mounted/available to every API instance.
5. Call `POST /models/reload-production` on each instance, or perform a rolling restart with startup synchronization.
6. Check `/health/ready` and `/models/runtime`.
7. Send smoke-test predictions.
8. Monitor errors, latency and model metrics.

## Rollback

Re-promote the previous RETIRED version, then reload production on each instance. Because the old versioned artifact is retained, rollback does not require retraining.

## Multi-instance boundary

The in-memory swap is atomic within one Python process. It is not a distributed transaction across multiple containers. A real multi-instance deployment should use rolling deployment/orchestration or a deployment controller that reloads each instance and verifies its runtime version before completing traffic rollout.

## Artifact storage

Local versioned paths are suitable for this project. Larger systems normally use durable object/artifact storage and immutable artifact checksums. A registry record should point to an immutable artifact rather than a mutable `latest.joblib` file.
