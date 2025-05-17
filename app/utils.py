import csv
import os
from datetime import datetime
import pandas as pd

LOG_FILE = "../access_log.csv"

def log_access(user_name, status, reason=""):
    """Append an access attempt to the log CSV file."""
    fieldnames = ["timestamp", "user_name", "status", "reason"]

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_name": user_name,
        "status": status,
        "reason": reason
    }

    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

def load_access_logs(csv_file="../access_log.csv"):
    if not os.path.exists(csv_file):
        return pd.DataFrame(columns=["timestamp", "user_name", "status", "reason"])
    return pd.read_csv(csv_file)
