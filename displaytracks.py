import h5py
import sys


def main():
    if len(sys.argv) < 2:
        print("h5py file is required")
        sys.exit(0)
    db_name = sys.argv[1]
    clip_id = None
    if len(sys.argv) == 3:
        clip_id = sys.argv[2]

    print("showing clips and tracks in {} clip id {}".format(db_name, clip_id))
    db = h5py.File(db_name, mode="r")
    if clip_id:
        display_clip(db, clip_id)
    else:
        display_clips(db)
    db.close()


def display_clips(db):
    clips = db["clips"]
    for clip_id in clips:
        display_clip(db, clip_id)


def display_clip(db, clip_id):
    clip = db["clips"][clip_id]
    print("clip {} start time {}".format(clip_id, clip.attrs.get("start_time")))
    for c_attr in clip:
        if c_attr != "frames":
            track = clip[c_attr]
            t_attr = track.attrs
            print(
                "track {} has {} frames is animal {} with confidence {}".format(
                    c_attr,
                    t_attr.get("frames"),
                    t_attr.get("tag"),
                    t_attr.get("confidence"),
                )
            )


if __name__ == "__main__":
    main()
