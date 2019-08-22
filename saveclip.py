import sys

from piclassifier.clipsaver import clip_to_mp4, extract_clip


def main():
    if len(sys.argv) < 3:
        print("h5py file and clip id arguments are required ")
        sys.exit(0)

    db_name = sys.argv[1]
    clip_id = sys.argv[2]
    
    raw = sys.argv[3]
    if raw == "t":
     	extract_clip(db_name, clip_id, "pithermal")
    else:	
    	clip_to_mp4(db_name, clip_id, "pithermal")


if __name__ == "__main__":
    main()
