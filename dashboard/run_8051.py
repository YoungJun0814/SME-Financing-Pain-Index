from pathlib import Path
import traceback

from app import app


if __name__ == "__main__":
    try:
        app.run(host="127.0.0.1", port=8051, debug=False)
    except Exception:
        Path("server_8051.error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
