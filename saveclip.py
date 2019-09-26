import sys
import os
from piclassifier.clipsaver import clip_to_mp4, extract_clip, cptv_to_mp4


def main():
    if len(sys.argv) < 3:
        print("h5py file and clip id arguments are required ")
        sys.exit(0)

    db_name = sys.argv[1]
    clip_id = sys.argv[2]
    if len(sys.argv) == 4:
        raw = sys.argv[3]
    if clip_id=="c":
        for folder_path, _, files in os.walk(db_name):
            for name in files:
                if os.path.splitext(name)[1] == ".cptv":
                    full_path = os.path.join(folder_path, name)
                    cptv_to_mp4(full_path)

    elif raw == "t":
        extract_clip(db_name, clip_id, "pithermal")
    else:
        clip_to_mp4(db_name, clip_id, "pithermal")


if __name__ == "__main__":
    main()
