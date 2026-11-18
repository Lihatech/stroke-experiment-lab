"""Upload existing local experiments to the configured PostgreSQL database."""

from experiment import RUNS
from storage import upload_existing_runs

print(f"Uploaded {upload_existing_runs(RUNS)} experiments from {RUNS}.")
