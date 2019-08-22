import h5py
import sys

if len(sys.argv) < 2:
    print("h5py file is required")
    sys.exit(0)

db_name = sys.argv[1]
print("showing clips and tracks in {}".format(db_name))
db = h5py.File(db_name, mode="r")
clips = db["clips"]
for clip_id in clips:
    clip = clips[clip_id]
    print("clip {} start time {}".format(clip_id, clip.attrs.get("start_time")))
    for c_attr in clip:
        if c_attr != "frames":
            track = clip[c_attr]
            t_attr = track.attrs
            print(
                "track {} has {} frames is animal {} with confidence {}".format(
                    c_attr,
                    t_attr.get("frames"),
                    t_attr.get("what"),
                    t_attr.get("confidence"),
                )
            )
