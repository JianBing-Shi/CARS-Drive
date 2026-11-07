"""Read-only inventory of locally extracted nuPlan camera archives and DB overlap."""

import json
import sqlite3
from pathlib import Path


def main():
    split_root = Path("H:/nuplan/dataset/nuplan-v1.1/splits")
    expected = {"CAM_F0", "CAM_B0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2"}
    db_sets = {s: {p.stem: p for p in (split_root / s).glob("*.db")} for s in ("mini", "train", "val")}
    report = {"mode": "read-only; directory presence is not a full per-frame completeness check", "shards": []}
    for outer in sorted(split_root.glob("*camera*")):
        if not outer.is_dir():
            continue
        inner = outer / outer.name
        base = inner if inner.is_dir() else outer
        logs = []
        for p in sorted(base.iterdir()):
            if not p.is_dir() or not p.name.startswith("20"):
                continue
            channels = {c.name for c in p.iterdir() if c.is_dir() and c.name.startswith("CAM_")}
            matches = [s for s, dbs in db_sets.items() if p.name in dbs]
            row = {"log": p.name, "channels": sorted(channels), "has_8_channel_dirs": expected <= channels, "db_splits": matches}
            if matches:
                db = db_sets[matches[0]][p.name]
                try:
                    with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as con:
                        location = con.execute("SELECT location FROM log LIMIT 1").fetchone()
                        row["location"] = location[0] if location else None
                        samples = con.execute("SELECT cam.channel, img.filename_jpg FROM image img JOIN camera cam ON cam.token=img.camera_token GROUP BY cam.channel").fetchall()
                    row["sample_image_exists"] = {}
                    for channel, filename in samples:
                        filename = Path(filename.replace("\\", "/"))
                        candidates = [base / filename, p / channel / filename.name]
                        row["sample_image_exists"][channel] = any(candidate.is_file() for candidate in candidates)
                except sqlite3.Error as exc:
                    row["db_error"] = str(exc)
            logs.append(row)
        report["shards"].append({"shard": outer.name, "base": str(base), "log_count": len(logs), "logs": logs})
    out = Path(__file__).parent / "evidence" / "local_camera_inventory_2026-10-19.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out), "shards": [{"shard": s["shard"], "logs": s["log_count"], "eight_channel_dirs": sum(r["has_8_channel_dirs"] for r in s["logs"]), "cities": sorted({r.get("location", "unknown") for r in s["logs"]}), "db_overlap": {split: sum(split in r["db_splits"] for r in s["logs"]) for split in db_sets}} for s in report["shards"]]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
