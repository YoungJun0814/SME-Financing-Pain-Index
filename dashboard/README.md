# SME-FPI Dashboard

Interactive Dash application for the SME Financing Pain Index project.

## Run

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

## Data

The app reads dashboard-ready files from `data/processed/` and uses visual assets from `dashboard/assets/` and `figures/`.

Local submission exports such as `dashboard/data/`, `dashboard/reports/`, and copied final-report files are intentionally ignored because they duplicate repository-level artifacts.

## Main Entry Point

- `app.py`: Dash layout, callbacks, and tab content.
- `assets/style_premium.css`: dashboard styling.
- `run_8051.py`: helper runner for an alternate local port.
