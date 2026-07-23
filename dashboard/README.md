# SME-FPI Dashboard

Interactive Dash application for the SME Financing Pain Index project.

## Local Run

Run from the repository root:

```powershell
python -m pip install -r requirements.txt
python dashboard/app.py
```

Open:

```text
http://127.0.0.1:8050
```

If port `8050` is already busy:

```powershell
python dashboard/run_8051.py
```

Runtime settings can also be supplied through environment variables:

```powershell
$env:DASH_HOST = "127.0.0.1"
$env:PORT = "8051"
$env:DASH_DEBUG = "false"
python dashboard/app.py
```

`PORT` must be between 1 and 65535. Debug mode is disabled by default.

## Container Run

Build from the repository root:

```bash
docker build -t sme-fpi-dashboard .
docker run --rm -p 8050:8050 sme-fpi-dashboard
```

The image installs the deployment requirements, copies the committed processed
data, runs as a non-root user, and serves `dashboard.wsgi:application` with
Gunicorn. Override the exposed service port when required:

```bash
docker run --rm -e PORT=9000 -p 9000:9000 sme-fpi-dashboard
```

## Data

The app reads dashboard-ready files from `data/processed/` and uses visual assets from `dashboard/assets/` and `figures/`.

Local submission exports such as `dashboard/data/`, `dashboard/reports/`, and copied final-report files are intentionally ignored because they duplicate repository-level artifacts.

## Main Entry Point

- `app.py`: Dash layout, callbacks, and tab content.
- `runtime.py`: validated host, port, and debug configuration.
- `wsgi.py`: production WSGI entry point.
- `assets/style_premium.css`: dashboard styling.
- `run_8051.py`: helper runner for an alternate local port.
