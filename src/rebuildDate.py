import os
import argparse

from ml_tools.trackdatabase import TrackDatabase
from config.config import Config
from datetime import timedelta
from datetime import date
from pathlib import Path
from dateutil.parser import parse as parse_date

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="Directory of cptv files")
args = parser.parse_args()
args.data_dir = Path(args.data_dir)
latest_date = None
for db_clip in args.data_dir.glob(f"**/*.cptv"):
    file_name = db_clip.name
    hyphen = file_name.index("-")
    date_s = file_name[hyphen + 1 : hyphen + 16]
    cptv_dt = parse_date(date_s)
    if latest_date is None or cptv_dt > latest_date:
        latest_date = cptv_dt


month_ago = latest_date - timedelta(days=30 * 6)
month_ago = month_ago.strftime("%Y-%m-%d 00:00:00")
print(month_ago)
