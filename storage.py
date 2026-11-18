"""Persist completed experiment folders in PostgreSQL when DATABASE_URL is set."""

import io
import json
import os
import re
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}-[0-9a-f]{8}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS stroke_experiments (
    run_id text PRIMARY KEY,
    created_at timestamptz NOT NULL,
    experiment_name text NOT NULL,
    settings jsonb NOT NULL,
    archive bytea NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
)
"""


def database_configured():
    return bool(os.environ.get("DATABASE_URL"))


def connect():
    import psycopg

    url = os.environ["DATABASE_URL"]
    options = {} if "sslmode=" in url else {"sslmode": "require"}
    return psycopg.connect(url, autocommit=True, connect_timeout=10, **options)


def initialize_database(connection):
    connection.execute(SCHEMA)


def archive_run(folder):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for path in sorted(folder.iterdir()):
            if path.is_file() and not path.name.endswith(".tmp"):
                archive.write(path, path.name)
    return buffer.getvalue()


def save_run_to_database(folder):
    if not database_configured():
        return False
    from psycopg.types.json import Jsonb

    settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
    with connect() as connection:
        initialize_database(connection)
        connection.execute(
            """
            INSERT INTO stroke_experiments (run_id, created_at, experiment_name, settings, archive)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
                created_at = EXCLUDED.created_at,
                experiment_name = EXCLUDED.experiment_name,
                settings = EXCLUDED.settings,
                archive = EXCLUDED.archive,
                updated_at = now()
            """,
            (folder.name, settings["created_utc"], settings["experiment_name"],
             Jsonb(settings), archive_run(folder)),
        )
    return True


def restore_run(root, run_id, contents):
    if not RUN_ID.fullmatch(run_id):
        raise ValueError(f"Invalid stored run ID: {run_id}")
    target = root / run_id
    if (target / "settings.json").is_file():
        return False
    target.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(io.BytesIO(contents)) as archive:
            for member in archive.infolist():
                if member.is_dir() or Path(member.filename).name != member.filename:
                    raise ValueError("The stored experiment archive contains an invalid path.")
                (target / member.filename).write_bytes(archive.read(member))
        if not (target / "settings.json").is_file():
            raise ValueError("The stored experiment archive is incomplete.")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return True


def sync_runs_from_database(root):
    if not database_configured():
        return 0
    root.mkdir(parents=True, exist_ok=True)
    restored = 0
    with connect() as connection:
        initialize_database(connection)
        for run_id, archive in connection.execute(
                "SELECT run_id, archive FROM stroke_experiments ORDER BY created_at"):
            restored += restore_run(root, run_id, bytes(archive))
    return restored


def upload_existing_runs(root):
    if not database_configured():
        raise RuntimeError("Set DATABASE_URL before uploading existing runs.")
    uploaded = 0
    for settings in root.glob("*/settings.json"):
        uploaded += save_run_to_database(settings.parent)
    return uploaded
