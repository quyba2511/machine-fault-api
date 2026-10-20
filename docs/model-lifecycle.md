# Model lifecycle: candidate, production and rollback

## State machine

`CANDIDATE -> PRODUCTION -> RETIRED`

A retired version can be promoted again for rollback. SQL Server enforces that at most one registry row has `PRODUCTION` status.

## Retrain

After enough trusted ground truth has accumulated:

```bash
python -m training.retrain --version v1.1 --threshold 0.5
```

The command trains on labeled production history, evaluates on a held-out validation split, writes a versioned artifact + metadata, and registers the version as `CANDIDATE`.

A candidate must not become production merely because training completed. Review validation metrics and business cost first.

## Promote

```bash
python scripts/promote_model.py v1.1
```

Promotion is transactional. The current production row is retired and the selected candidate becomes production in the same SQL transaction.

## Rollback

If v1.1 behaves badly after deployment:

```bash
python scripts/promote_model.py v1.0
```

The same operation retires v1.1 and makes v1.0 production again.

## Important limitation of V5

The registry controls lifecycle metadata, but the running API still loads the local configured artifact. A deployment/reload step must synchronize the promoted registry version to the runtime artifact before traffic switches. A later release can add an artifact store and atomic runtime model reload.

## Enterprise principle

Retraining, model approval, deployment and rollback are separate operations. This separation prevents a newly trained model from silently replacing a known production model without validation and an audit trail.
