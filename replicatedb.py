import h5py
import numpy as np
from config.config import Config
import os

gzip_compression = {"compression": "gzip"}


def copy(old_db, new_db):
    print("copying db")

    f = h5py.File(old_db, "r")
    copy = h5py.File(new_db, "w")

    clips = f["clips"]
    copy_clips = copy.create_group("clips")
    for clip_id in clips:
        print("Copying", clip_id)
        og_frames = {}
        copy_group(clips, copy_clips, clip_id, og_frames, True)
        frame_keys = list(og_frames.keys())
        frame_keys.sort()
        copy_og = copy_clips[clip_id].create_group("original_frames")
        for k, v in og_frames.items():
            copy_frame = copy_og.create_dataset(
                str(k), shape=v.shape, chunks=v.shape, **gzip_compression, dtype=v.dtype
            )
            copy_frame[:] = v
        # break
    copy.close()
    f.close()


def copy_group(f, copy, key, og_frames, upper=False):
    copy_g = copy.create_group(key)
    group = f[key]
    for g in group:
        if g == "original" or (upper and g == "frames"):
            start_frame = 0
            if g == "original":
                start_frame = group.attrs.get("start_frame")
            f_data = group[g]
            for frame_num in f_data:
                if frame_num not in og_frames:
                    og_frames[int(frame_num) + start_frame] = f_data[frame_num]
            continue
        # print("D or G",g,isinstance(group[g], h5py.Dataset), isinstance(group[g], h5py.Group))
        if isinstance(group[g], h5py.Group):
            copy_group(group, copy_g, g, og_frames)
        else:
            # print("create new dataset", g)
            ds = group[g]
            # print(ds.shape, ds.chunks, ds.dtype, ds.compression)
            copy_ds = copy_g.create_dataset(
                g, shape=ds.shape, chunks=ds.chunks, **gzip_compression, dtype=ds.dtype
            )
            copy_ds[:] = ds

    for k, v in group.attrs.items():
        # if  isinstance(v, np.ndarray):
        #     print("numpy array", k)
        # print("key is",k,v)
        copy_g.attrs[k] = v


def main():
    config = Config.load_from_file(None)
    db_file = os.path.join(config.tracks_folder, "dataset.hdf5")
    copy(db_file, os.path.join(config.tracks_folder, "copied.hdf5"))


if __name__ == "__main__":
    main()
