# Production deployment

## Target flow

Client -> Nginx :80 -> Docker -> FastAPI :8000 -> SQL Server

The API container binds only to `127.0.0.1:8000` on the host. Nginx is the public/internal-network entry point.

## Ubuntu prerequisites

Install Docker Engine + Compose plugin, Nginx, Microsoft ODBC Driver 18 for SQL Server, and unixODBC dependencies as appropriate for the Ubuntu release.

## Configuration

Copy `.env.example` to `.env` and set the real `DATABASE_URL`. Never commit `.env`.

Run `sql/001_create_prediction_history.sql` against the target database before enabling persistence.

## Start

```bash
docker compose up -d --build
```

Check:

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
```

## Nginx

Copy `deploy/nginx.conf` into an enabled Nginx site, validate with `nginx -t`, then reload Nginx.

## systemd

Place the repository at `/opt/machine-fault-api`, copy `deploy/machine-fault-api.service` to `/etc/systemd/system/`, then enable/start it with systemctl.

## Health semantics

- `/health/live`: process is alive. It intentionally does not query SQL Server.
- `/health/ready`: model can load and, when persistence is enabled, SQL Server answers `SELECT 1`.

This distinction prevents a temporary database outage from causing the container/process itself to be treated as dead while still allowing a load balancer or orchestrator to stop sending traffic to an unready instance.

## Request tracing

Nginx supplies `X-Request-ID`; FastAPI echoes it and logs method, path, status and duration. This gives operators a correlation key when investigating a failed business request across proxy and application logs.
