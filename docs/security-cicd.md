# Security and CI/CD

## Endpoint classes

Public/internal service endpoints:
- `POST /predict`
- health endpoints
- `GET /models/runtime` (operational metadata only; restrict at the network layer if required)

Administrative endpoints require `X-Admin-API-Key`:
- `POST /models/reload-production`
- `PATCH /predictions/{id}/actual-fault`
- monitoring endpoints

The service fails closed when `ADMIN_API_KEY` is missing. A real production deployment should store this value in a secret manager or protected deployment environment, not in Git or a Docker image.

API-key authentication is deliberately a small V7 boundary, not a full identity system. Enterprise deployments should normally place the service behind SSO/OIDC, an API gateway, mTLS, or an internal zero-trust access layer and use roles for operator/admin permissions.

## CI gates

Every pull request runs:
1. dependency installation
2. model training smoke step
3. pytest
4. Docker image build

The workflow has read-only repository contents permission. CI does not deploy automatically from an unreviewed pull request.

## Recommended deployment pipeline

PR -> tests -> review -> merge -> build immutable image -> vulnerability scan/signing -> staging -> smoke test -> approval -> production rollout -> readiness check -> monitor -> rollback if needed.

The repository currently implements the build/test gates. Registry publishing and production deployment should be added only after a real target registry/server and credentials are available; credentials must be GitHub environment secrets, never committed configuration.
