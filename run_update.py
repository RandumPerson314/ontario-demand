"""Runs update.py and, if it crashes anywhere, records the crash on the website's status banner
(keeping the last good data) before failing the job, so a problem is never silent."""
import json
import os
import runpy
import traceback

try:
    runpy.run_path("update.py", run_name="__main__")
except SystemExit:
    raise
except BaseException:
    tb = traceback.format_exc()
    print(tb)
    path = "docs/demand_log.js"
    try:
        txt = open(path).read().strip() if os.path.exists(path) else ""
        dec = json.JSONDecoder()
        rows, meta = [], {}
        if txt.startswith("window.LOG="):
            rows, i = dec.raw_decode(txt[len("window.LOG="):])
            rest = txt[len("window.LOG=") + i:]
            if "window.LOG_META=" in rest:
                meta = dec.raw_decode(rest[rest.index("window.LOG_META=") + len("window.LOG_META="):])[0]
        meta["status"] = [s for s in meta.get("status", []) if not s["name"].startswith("Update job")] + [
            dict(name="Update job", ok=False, detail=tb.strip().splitlines()[-1][:300])]
        with open(path, "w") as fh:               # 'updated' is left alone, so the page still says how old the data is
            fh.write("window.LOG=" + json.dumps(rows, separators=(",", ":")) + ";window.LOG_META=" + json.dumps(meta) + ";")
    finally:
        raise
