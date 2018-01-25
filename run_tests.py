# simple script to run various tests
import os

# run difficult cases
python classify.py missclassified -p -vv -i -o c:/cac/runs/difficult -m models/model_hq_joint

# run a weeks unseen data
python classify.py all -p -vv -i --start-date 2018-01-01 --end-date 2018-01-07 -o c:/cac/runs/hq_joint -m models/model_hq_joint