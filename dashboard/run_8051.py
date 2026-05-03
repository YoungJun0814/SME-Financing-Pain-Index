from pathlib import Path
import importlib.util
import traceback


APP_PATH = Path(__file__).resolve().with_name("app.py")
SPEC = importlib.util.spec_from_file_location("sme_fpi_dashboard_app", APP_PATH)
APP_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(APP_MODULE)
app = APP_MODULE.app


if __name__ == "__main__":
    try:
        app.run(host="127.0.0.1", port=8051, debug=False)
    except Exception:
        Path("server_8051.error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
