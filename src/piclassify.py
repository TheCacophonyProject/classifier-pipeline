#!/usr/bin/env python

"""
Script to classify animals within a CPTV video file.
"""

import multiprocessing
multiprocessing.set_forkserver_preload(["numpy"])
if __name__ == "__main__":
    multiprocessing.set_start_method('forkserver')
    from piclassifier.piclassify import main
    main()
