import os
import argparse

from ml_tools.trackdatabase import TrackDatabase
from config.config import Config
from datetime import timedelta
from datetime import date

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config-file", help="Path to config file to use")
args = parser.parse_args()

config = Config.load_from_file(args.config_file)
db_file = os.path.join(config.tracks_folder, "dataset.hdf5")
db = TrackDatabase(db_file)
latest_date = db.latest_date()
month_ago = latest_date - timedelta(days=30)
month_ago = month_ago.strftime("%Y-%m-%d 00:00:00")
print(month_ago)
