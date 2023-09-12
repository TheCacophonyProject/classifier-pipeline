"" """
classifier-pipeline - this is a server side component that manipulates cptv
files and to create a classification model of animals present
Copyright (C) 2018, The Cacophony Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import logging
import time
from multiprocessing import Process, Queue
import traceback
from ml_tools import tools

from load.clip import Clip, ClipStats
import numpy as np
import json
from pathlib import Path

import h5py
from cptv import CPTVReader
import yaml
from load.cliptrackextractor import is_affected_by_ffc
from track.track import Track
from track.region import Region

FPS = 9


def process_job(loader, queue, out_dir, config):
    i = 0
    while True:
        i += 1
        filename = queue.get()
        logging.info("Processing %s", filename)
        try:
            if filename == "DONE":
                break
            else:
                loader.process_file(str(filename), out_dir, config)
            if i % 50 == 0:
                logging.info("%s jobs left", queue.qsize())
        except Exception as e:
            logging.error("Process_job error %s %s", filename, e)
            traceback.print_exc()


class ClipLoader:
    def __init__(self, config):
        self.config = config

        # number of threads to use when processing jobs.
        self.workers_threads = 9

    def process_all(self, root, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        job_queue = Queue()
        processes = []
        for i in range(max(1, self.workers_threads)):
            p = Process(
                target=process_job,
                args=(self, job_queue, out_dir, self.config),
            )
            processes.append(p)
            p.start()

        file_paths = []
        file_paths = [
            "/data2/cptv-files/04/1041937-20210511-121936-ospri14.cptv",
            "/data2/cptv-files/10/1018117-20210509-072813-ospri16.cptv",
            "/data2/cptv-files/30/966222-20210507-194112-ospri15.cptv",
            "/data2/cptv-files/31/1044746-20210513-053540-ospri13.cptv",
            "/data2/cptv-files/37/1045184-20210511-195058-ospri15b.cptv",
            "/data2/cptv-files/50/1045175-20210510-144946-ospri15b.cptv",
            "/data2/cptv-files/54/966372-20210507-064904-ospri14.cptv",
            "/data2/cptv-files/56/963147-20210507-195146-ospri16.cptv",
            "/data2/cptv-files/5a/1044733-20210512-052901-ospri13.cptv",
            "/data2/cptv-files/5f/966388-20210511-161714-ospri14.cptv",
            "/data2/cptv-files/6a/966380-20210509-130558-ospri14.cptv",
            "/data2/cptv-files/76/966254-20210511-195058-ospri15.cptv",
            "/data2/cptv-files/77/966212-20210507-075725-ospri15.cptv",
            "/data2/cptv-files/7a/1041919-20210506-104438-ospri14.cptv",
            "/data2/cptv-files/88/1018109-20210507-201137-ospri16.cptv",
            "/data2/cptv-files/8c/966577-20210509-073450-ospri13.cptv",
            "/data2/cptv-files/93/966278-20210509-193700-ospri13.cptv",
            "/data2/cptv-files/b7/966215-20210507-080133-ospri15.cptv",
            "/data2/cptv-files/d6/966284-20210510-051611-ospri13.cptv",
            "/data2/cptv-files/e2/1018131-20210512-200355-ospri16.cptv",
            "/data2/cptv-files/ee/966301-20210509-061051-ospri18.cptv",
            "/data2/cptv-files/f4/966440-20210509-050302-ospri19.cptv",
            "/data2/cptv-files/f8/1041942-20210512-094255-ospri14.cptv",
            "/data2/cptv-files/fa/966286-20210510-194800-ospri13.cptv",
            "/data2/cptv-files/ff/1041830-20210510-195348-ospri18.cptv",
            "/data2/cptv-files/02/1041943-20210512-195937-ospri14.cptv",
            "/data2/cptv-files/10/966227-20210508-125127-ospri15.cptv",
            "/data2/cptv-files/16/966211-20210507-075659-ospri15.cptv",
            "/data2/cptv-files/18/966386-20210511-161533-ospri14.cptv",
            "/data2/cptv-files/38/1044778-20210513-055534-ospri18.cptv",
            "/data2/cptv-files/40/966217-20210507-080338-ospri15.cptv",
            "/data2/cptv-files/42/1041999-20210508-050804-ospri19.cptv",
            "/data2/cptv-files/48/966276-20210509-073450-ospri13.cptv",
            "/data2/cptv-files/49/966194-20210506-200652-ospri18.cptv",
            "/data2/cptv-files/5d/1041901-20210510-195742-ospri13.cptv",
            "/data2/cptv-files/5e/1044769-20210512-045109-ospri18.cptv",
            "/data2/cptv-files/76/1044730-20210511-194315-ospri13.cptv",
            "/data2/cptv-files/76/966299-20210513-193136-ospri13.cptv",
            "/data2/cptv-files/7b/1041859-20210507-181116-ospri13.cptv",
            "/data2/cptv-files/7d/966326-20210513-055534-ospri18.cptv",
            "/data2/cptv-files/99/963158-20210509-081301-ospri16.cptv",
            "/data2/cptv-files/9a/963164-20210510-062212-ospri16.cptv",
            "/data2/cptv-files/a2/966465-20210510-091256-ospri19.cptv",
            "/data2/cptv-files/b1/1041993-20210510-103940-ospri15.cptv",
            "/data2/cptv-files/b3/1041912-20210506-051110-ospri14.cptv",
            "/data2/cptv-files/b7/963159-20210509-081430-ospri16.cptv",
            "/data2/cptv-files/b9/1045186-20210511-200334-ospri15b.cptv",
            "/data2/cptv-files/c2/966235-20210510-074924-ospri15.cptv",
            "/data2/cptv-files/c3/1018132-20210513-051020-ospri16.cptv",
            "/data2/cptv-files/c8/966327-20210513-183804-ospri18.cptv",
            "/data2/cptv-files/cf/966279-20210509-194014-ospri13.cptv",
            "/data2/cptv-files/d0/963145-20210507-201137-ospri16.cptv",
            "/data2/cptv-files/d2/966204-20210506-101423-ospri15.cptv",
            "/data2/cptv-files/d3/963148-20210507-201137-ospri16.cptv",
            "/data2/cptv-files/db/966323-20210512-195102-ospri18.cptv",
            "/data2/cptv-files/de/911538-20210506-050448-ospri13.cptv",
            "/data2/cptv-files/e0/1044768-20210511-080124-ospri18.cptv",
            "/data2/cptv-files/e4/966378-20210508-201151-ospri14.cptv",
            "/data2/cptv-files/ed/966271-20210508-161624-ospri13.cptv",
            "/data2/cptv-files/f6/966283-20210510-051502-ospri13.cptv",
            "/data2/cptv-files/ff/1041900-20210510-195558-ospri13.cptv",
            "/data2/cptv-files/12/966379-20210508-201208-ospri14.cptv",
            "/data2/cptv-files/16/1041979-20210507-194112-ospri15.cptv",
            "/data2/cptv-files/18/966288-20210510-195742-ospri13.cptv",
            "/data2/cptv-files/25/966316-20210511-080124-ospri18.cptv",
            "/data2/cptv-files/28/1041861-20210507-200543-ospri13.cptv",
            "/data2/cptv-files/2a/966265-20210506-200730-ospri13.cptv",
            "/data2/cptv-files/37/1045188-20210512-103645-ospri15b.cptv",
            "/data2/cptv-files/3b/966289-20210511-194315-ospri13.cptv",
            "/data2/cptv-files/3e/966382-20210511-070749-ospri14.cptv",
            "/data2/cptv-files/56/1041863-20210508-053816-ospri13.cptv",
            "/data2/cptv-files/5d/1041969-20210507-075725-ospri15.cptv",
            "/data2/cptv-files/60/1041978-20210507-080746-ospri15.cptv",
            "/data2/cptv-files/70/966381-20210510-122402-ospri14.cptv",
            "/data2/cptv-files/76/966296-20210513-175715-ospri13.cptv",
            "/data2/cptv-files/77/1041825-20210509-200701-ospri18.cptv",
            "/data2/cptv-files/79/1045167-20210510-104135-ospri15b.cptv",
            "/data2/cptv-files/93/966264-20210513-050034-ospri15.cptv",
            "/data2/cptv-files/96/1044775-20210512-195102-ospri18.cptv",
            "/data2/cptv-files/a1/963162-20210510-050638-ospri16.cptv",
            "/data2/cptv-files/a2/1044767-20210510-195615-ospri18.cptv",
            "/data2/cptv-files/ab/1018116-20210509-070444-ospri16.cptv",
            "/data2/cptv-files/ae/1018130-20210511-193327-ospri16.cptv",
            "/data2/cptv-files/b2/1041972-20210507-080133-ospri15.cptv",
            "/data2/cptv-files/bd/966228-20210508-200918-ospri15.cptv",
            "/data2/cptv-files/c5/1044759-20210510-195516-ospri18.cptv",
            "/data2/cptv-files/cd/1044748-20210513-133141-ospri13.cptv",
            "/data2/cptv-files/d3/1044750-20210513-175715-ospri13.cptv",
            "/data2/cptv-files/d6/1041973-20210507-080301-ospri15.cptv",
            "/data2/cptv-files/d6/966231-20210509-082208-ospri15.cptv",
            "/data2/cptv-files/d9/1041878-20210509-193406-ospri13.cptv",
            "/data2/cptv-files/db/966224-20210508-082604-ospri15.cptv",
            "/data2/cptv-files/e1/1041915-20210506-085408-ospri14.cptv",
            "/data2/cptv-files/e7/966315-20210510-195615-ospri18.cptv",
            "/data2/cptv-files/e8/1041829-20210510-195226-ospri18.cptv",
            "/data2/cptv-files/ed/1041820-20210509-061051-ospri18.cptv",
            "/data2/cptv-files/ef/966270-20210508-053816-ospri13.cptv",
            "/data2/cptv-files/0e/966390-20210512-094255-ospri14.cptv",
            "/data2/cptv-files/11/1041931-20210508-201208-ospri14.cptv",
            "/data2/cptv-files/17/859177-20210531-050121-Wallaby2.cptv",
            "/data2/cptv-files/24/966260-20210512-104021-ospri15.cptv",
            "/data2/cptv-files/2c/966371-20210506-195004-ospri14.cptv",
            "/data2/cptv-files/2d/1045178-20210511-071533-ospri15b.cptv",
            "/data2/cptv-files/2e/1041881-20210509-193700-ospri13.cptv",
            "/data2/cptv-files/32/966290-20210512-052901-ospri13.cptv",
            "/data2/cptv-files/39/911563-20210506-200508-ospri18.cptv",
            "/data2/cptv-files/40/966309-20210510-195130-ospri18.cptv",
            "/data2/cptv-files/4e/966364-20210506-051034-ospri14.cptv",
            "/data2/cptv-files/5d/966280-20210509-200001-ospri13.cptv",
            "/data2/cptv-files/78/1041894-20210510-051611-ospri13.cptv",
            "/data2/cptv-files/7b/966442-20210509-050901-ospri19.cptv",
            "/data2/cptv-files/83/1044779-20210513-183804-ospri18.cptv",
            "/data2/cptv-files/86/966579-20210510-091256-ospri19.cptv",
            "/data2/cptv-files/8e/1041866-20210508-194550-ospri13.cptv",
            "/data2/cptv-files/97/1041892-20210510-051502-ospri13.cptv",
            "/data2/cptv-files/a0/966460-20210510-073439-ospri19.cptv",
            "/data2/cptv-files/b0/966300-20210506-200652-ospri18.cptv",
            "/data2/cptv-files/b4/1041998-20210507-063341-ospri19.cptv",
            "/data2/cptv-files/c9/1041874-20210509-073450-ospri13.cptv",
            "/data2/cptv-files/d7/1044757-20210513-193136-ospri13.cptv",
            "/data2/cptv-files/d9/966209-20210507-070351-ospri15.cptv",
            "/data2/cptv-files/db/966453-20210510-050110-ospri19.cptv",
            "/data2/cptv-files/dd/966580-20210510-181820-ospri19.cptv",
            "/data2/cptv-files/fa/966251-20210511-161658-ospri15.cptv",
            "/data2/cptv-files/fe/966310-20210510-195226-ospri18.cptv",
            "/data2/cptv-files/10/1041961-20210506-101327-ospri15.cptv",
            "/data2/cptv-files/12/1042005-20210509-065930-ospri19.cptv",
            "/data2/cptv-files/12/966362-20210506-050204-ospri14.cptv",
            "/data2/cptv-files/14/966203-20210506-101327-ospri15.cptv",
            "/data2/cptv-files/17/1044774-20210512-193245-ospri18.cptv",
            "/data2/cptv-files/19/911558-20210506-200108-ospri16.cptv",
            "/data2/cptv-files/1a/963161-20210510-050555-ospri16.cptv",
            "/data2/cptv-files/3b/1041974-20210507-080338-ospri15.cptv",
            "/data2/cptv-files/3e/966246-20210510-144946-ospri15.cptv",
            "/data2/cptv-files/47/966456-20210510-053936-ospri19.cptv",
            "/data2/cptv-files/4e/859167-20210530-223648-Stoat hunter.cptv",
            "/data2/cptv-files/55/966277-20210509-193406-ospri13.cptv",
            "/data2/cptv-files/5d/1041927-20210508-164957-ospri14.cptv",
            "/data2/cptv-files/5f/859192-20210531-051011-Wallaby2.cptv",
            "/data2/cptv-files/60/966457-20210510-054115-ospri19.cptv",
            "/data2/cptv-files/6c/1041884-20210509-194014-ospri13.cptv",
            "/data2/cptv-files/76/966258-20210512-103645-ospri15.cptv",
            "/data2/cptv-files/78/966306-20210509-200701-ospri18.cptv",
            "/data2/cptv-files/79/1041914-20210506-085335-ospri14.cptv",
            "/data2/cptv-files/7a/966221-20210507-080746-ospri15.cptv",
            "/data2/cptv-files/7c/966220-20210507-080628-ospri15.cptv",
            "/data2/cptv-files/8f/1018129-20210511-072043-ospri16.cptv",
            "/data2/cptv-files/91/1041858-20210506-200730-ospri13.cptv",
            "/data2/cptv-files/98/966304-20210509-195731-ospri18.cptv",
            "/data2/cptv-files/9d/1041897-20210510-194800-ospri13.cptv",
            "/data2/cptv-files/b4/966292-20210512-194515-ospri13.cptv",
            "/data2/cptv-files/b6/966236-20210510-103940-ospri15.cptv",
            "/data2/cptv-files/ba/1045194-20210513-050034-ospri15b.cptv",
            "/data2/cptv-files/ba/966287-20210510-195558-ospri13.cptv",
            "/data2/cptv-files/c0/1041991-20210510-074807-ospri15.cptv",
            "/data2/cptv-files/c1/966249-20210511-071533-ospri15.cptv",
            "/data2/cptv-files/d0/1042001-20210509-050422-ospri19.cptv",
            "/data2/cptv-files/d4/1041966-20210507-070351-ospri15.cptv",
            "/data2/cptv-files/eb/966294-20210513-053540-ospri13.cptv",
            "/data2/cptv-files/f4/966373-20210507-195517-ospri14.cptv",
            "/data2/cptv-files/fc/1041968-20210507-075659-ospri15.cptv",
            "/data2/cptv-files/00/966311-20210510-195348-ospri18.cptv",
            "/data2/cptv-files/02/1041938-20210511-161533-ospri14.cptv",
            "/data2/cptv-files/0e/963171-20210513-051020-ospri16.cptv",
            "/data2/cptv-files/11/966383-20210511-075223-ospri14.cptv",
            "/data2/cptv-files/14/1045180-20210511-161658-ospri15b.cptv",
            "/data2/cptv-files/1c/966387-20210511-161657-ospri14.cptv",
            "/data2/cptv-files/20/1044776-20210512-195705-ospri18.cptv",
            "/data2/cptv-files/26/966272-20210508-194550-ospri13.cptv",
            "/data2/cptv-files/33/1041987-20210509-082123-ospri15.cptv",
            "/data2/cptv-files/36/966445-20210509-065930-ospri19.cptv",
            "/data2/cptv-files/42/966459-20210510-063212-ospri19.cptv",
            "/data2/cptv-files/4d/966230-20210509-082123-ospri15.cptv",
            "/data2/cptv-files/54/966449-20210509-200113-ospri19.cptv",
            "/data2/cptv-files/5b/963156-20210509-072813-ospri16.cptv",
            "/data2/cptv-files/62/1041977-20210507-080628-ospri15.cptv",
            "/data2/cptv-files/6f/1044736-20210512-053041-ospri13.cptv",
            "/data2/cptv-files/73/966375-20210508-164957-ospri14.cptv",
            "/data2/cptv-files/78/1041975-20210507-080510-ospri15.cptv",
            "/data2/cptv-files/7a/1041922-20210506-194349-ospri14.cptv",
            "/data2/cptv-files/91/966368-20210506-104438-ospri14.cptv",
            "/data2/cptv-files/92/966229-20210509-082030-ospri15.cptv",
            "/data2/cptv-files/b4/966363-20210506-050414-ospri14.cptv",
            "/data2/cptv-files/ce/966365-20210506-051110-ospri14.cptv",
            "/data2/cptv-files/da/963168-20210511-072043-ospri16.cptv",
            "/data2/cptv-files/e0/1044739-20210512-194515-ospri13.cptv",
            "/data2/cptv-files/f6/963155-20210509-070444-ospri16.cptv",
            "/data2/cptv-files/00/966462-20210510-081247-ospri19.cptv",
            "/data2/cptv-files/14/966266-20210507-181116-ospri13.cptv",
            "/data2/cptv-files/15/966223-20210508-075658-ospri15.cptv",
            "/data2/cptv-files/18/1044771-20210512-192556-ospri18.cptv",
            "/data2/cptv-files/18/966391-20210512-195937-ospri14.cptv",
            "/data2/cptv-files/1c/966303-20210509-195659-ospri18.cptv",
            "/data2/cptv-files/21/966454-20210510-053921-ospri19.cptv",
            "/data2/cptv-files/28/1041934-20210511-070749-ospri14.cptv",
            "/data2/cptv-files/38/966438-20210507-063341-ospri19.cptv",
            "/data2/cptv-files/3d/1042006-20210509-195938-ospri19.cptv",
            "/data2/cptv-files/3e/1041924-20210507-064904-ospri14.cptv",
            "/data2/cptv-files/3f/1041828-20210510-195130-ospri18.cptv",
            "/data2/cptv-files/52/963143-20210507-195146-ospri16.cptv",
            "/data2/cptv-files/54/1041932-20210509-130558-ospri14.cptv",
            "/data2/cptv-files/61/966446-20210509-195938-ospri19.cptv",
            "/data2/cptv-files/70/1041941-20210511-200144-ospri14.cptv",
            "/data2/cptv-files/78/1042007-20210509-195954-ospri19.cptv",
            "/data2/cptv-files/7e/1044744-20210510-195348-ospri18.cptv",
            "/data2/cptv-files/86/966190-20210506-054851-ospri15.cptv",
            "/data2/cptv-files/90/966370-20210506-194349-ospri14.cptv",
            "/data2/cptv-files/93/1045174-20210510-144856-ospri15b.cptv",
            "/data2/cptv-files/97/1041823-20210509-195731-ospri18.cptv",
            "/data2/cptv-files/97/966298-20210513-180114-ospri13.cptv",
            "/data2/cptv-files/9d/966448-20210509-195954-ospri19.cptv",
            "/data2/cptv-files/b7/1041996-20210506-154018-ospri19.cptv",
            "/data2/cptv-files/bc/1045166-20210510-104045-ospri15b.cptv",
            "/data2/cptv-files/cf/1018122-20210510-050555-ospri16.cptv",
            "/data2/cptv-files/db/966216-20210507-080301-ospri15.cptv",
            "/data2/cptv-files/e0/1041907-20210506-050204-ospri14.cptv",
            "/data2/cptv-files/e4/963169-20210511-193327-ospri16.cptv",
            "/data2/cptv-files/e6/1041886-20210509-200001-ospri13.cptv",
            "/data2/cptv-files/f4/966441-20210509-050422-ospri19.cptv",
            "/data2/cptv-files/0b/1041984-20210508-125127-ospri15.cptv",
            "/data2/cptv-files/1b/1041822-20210509-195659-ospri18.cptv",
            "/data2/cptv-files/25/966313-20210510-195516-ospri18.cptv",
            "/data2/cptv-files/27/859169-20210531-032903-Stoat hunter.cptv",
            "/data2/cptv-files/2d/963170-20210512-200355-ospri16.cptv",
            "/data2/cptv-files/33/966452-20210509-200219-ospri19.cptv",
            "/data2/cptv-files/3b/966436-20210506-154018-ospri19.cptv",
            "/data2/cptv-files/48/966319-20210512-192556-ospri18.cptv",
            "/data2/cptv-files/49/1045190-20210512-104021-ospri15b.cptv",
            "/data2/cptv-files/4e/1018119-20210509-081301-ospri16.cptv",
            "/data2/cptv-files/52/966237-20210510-104045-ospri15.cptv",
            "/data2/cptv-files/53/963144-20210507-195146-ospri16.cptv",
            "/data2/cptv-files/54/1041864-20210508-161624-ospri13.cptv",
            "/data2/cptv-files/56/1018123-20210510-050638-ospri16.cptv",
            "/data2/cptv-files/5c/966322-20210512-193245-ospri18.cptv",
            "/data2/cptv-files/65/966324-20210512-195705-ospri18.cptv",
            "/data2/cptv-files/71/859170-20210531-032936-Stoat hunter.cptv",
            "/data2/cptv-files/71/966389-20210511-200144-ospri14.cptv",
            "/data2/cptv-files/81/1018120-20210509-081430-ospri16.cptv",
            "/data2/cptv-files/82/1041904-20210511-194315-ospri13.cptv",
            "/data2/cptv-files/83/1041909-20210506-050414-ospri14.cptv",
            "/data2/cptv-files/86/966295-20210513-133141-ospri13.cptv",
            "/data2/cptv-files/93/966366-20210506-085335-ospri14.cptv",
            "/data2/cptv-files/ac/966268-20210507-200543-ospri13.cptv",
            "/data2/cptv-files/be/966463-20210510-083521-ospri19.cptv",
            "/data2/cptv-files/c5/966234-20210510-074807-ospri15.cptv",
            "/data2/cptv-files/d0/1042000-20210509-050302-ospri19.cptv",
            "/data2/cptv-files/d6/1041981-20210508-082604-ospri15.cptv",
            "/data2/cptv-files/de/1041925-20210507-195517-ospri14.cptv",
            "/data2/cptv-files/e3/1041930-20210508-201151-ospri14.cptv",
            "/data2/cptv-files/f3/911554-20210506-064036-ospri16.cptv",
            "/data2/cptv-files/11/963151-20210509-063524-ospri16.cptv",
            "/data2/cptv-files/16/1041923-20210506-195004-ospri14.cptv",
            "/data2/cptv-files/1a/966385-20210511-121936-ospri14.cptv",
            "/data2/cptv-files/33/1041911-20210506-051034-ospri14.cptv",
            "/data2/cptv-files/3b/1041896-20210510-193644-ospri13.cptv",
            "/data2/cptv-files/45/966291-20210512-053041-ospri13.cptv",
            "/data2/cptv-files/54/859166-20210530-223423-Stoat hunter.cptv",
            "/data2/cptv-files/57/1042002-20210509-050901-ospri19.cptv",
            "/data2/cptv-files/5a/1041933-20210510-122402-ospri14.cptv",
            "/data2/cptv-files/7d/966218-20210507-080510-ospri15.cptv",
            "/data2/cptv-files/8e/966198-20210506-054851-ospri15.cptv",
            "/data2/cptv-files/98/966285-20210510-193644-ospri13.cptv",
            "/data2/cptv-files/9a/1041819-20210506-200652-ospri18.cptv",
            "/data2/cptv-files/9b/966245-20210510-144856-ospri15.cptv",
            "/data2/cptv-files/a3/966317-20210512-045109-ospri18.cptv",
            "/data2/cptv-files/b4/966256-20210511-200334-ospri15.cptv",
            "/data2/cptv-files/bd/1041992-20210510-074924-ospri15.cptv",
            "/data2/cptv-files/c6/966439-20210508-050804-ospri19.cptv",
            "/data2/cptv-files/f4/1045179-20210511-161153-ospri15b.cptv",
            "/data2/cptv-files/f5/966238-20210510-104135-ospri15.cptv",
            "/data2/cptv-files/f6/1044754-20210513-180114-ospri13.cptv",
            "/data2/cptv-files/fb/966367-20210506-085408-ospri14.cptv",
        ]
        #
        # for folder_path, _, files in os.walk(root):
        #     for name in files:
        #         if os.path.splitext(name)[1] in [".cptv"]:
        #             full_path = os.path.join(folder_path, name)
        #             file_paths.append(full_path)
        # # allows us know the order of processing
        file_paths.sort()
        for file_path in file_paths:
            job_queue.put(file_path)

        logging.info("Processing %d", job_queue.qsize())
        for i in range(len(processes)):
            job_queue.put("DONE")
        for process in processes:
            try:
                process.join()
            except KeyboardInterrupt:
                logging.info("KeyboardInterrupt, terminating.")
                for process in processes:
                    process.terminate()
                exit()

    def process_file(self, filename, out_dir, config):
        start = time.time()
        filename = Path(filename)
        logging.info(f"processing %s", filename)
        metadata_file = filename.with_suffix(".txt")
        if not metadata_file.exists():
            logging.error("No meta data found for %s", metadata_file)
            return

        metadata = tools.load_clip_metadata(metadata_file)
        r_id = metadata["id"]
        out_file = out_dir / f"{r_id}.hdf5"
        tracker_version = metadata.get("tracker_version")
        logging.info("Tracker version is %s", tracker_version)
        if out_file.exists() and tracker_version > 9:
            logging.warning("Already loaded %s", filename)
            # going to add some missing fierlds
            # with h5py.File(out_file, "a") as f:
            #     if f.attrs.get("device_id") is None:
            #         f.attrs["device_id"] = metadata["deviceId"]
            # return
        if len(metadata.get("Tracks")) == 0:
            logging.error("No tracks found for %s", filename)
            return

        clip = Clip(config.tracking["thermal"], filename)
        clip.load_metadata(
            metadata,
            config.load.tag_precedence,
        )

        with h5py.File(out_file, "w") as f:
            try:
                logging.info("creating clip %s", clip.get_id())

                clip_id = str(clip.get_id())

                clip_node = f
                triggered_temp_thresh = None
                camera_model = None
                with open(clip.source_file, "rb") as f:
                    reader = CPTVReader(f)
                    clip.set_res(reader.x_resolution, reader.y_resolution)
                    if clip.from_metadata:
                        for track in clip.tracks:
                            track.crop_regions()
                    if reader.model:
                        camera_model = reader.model.decode()
                    clip.set_model(camera_model)

                    # if we have the triggered motion threshold should use that
                    # maybe even override dynamic threshold with this value
                    if reader.motion_config:
                        motion = yaml.safe_load(reader.motion_config)
                        triggered_temp_thresh = motion.get("triggeredthresh")
                        if triggered_temp_thresh:
                            clip.temp_thresh = triggered_temp_thresh

                    video_start_time = reader.timestamp.astimezone(Clip.local_tz)
                    clip.set_video_stats(video_start_time)

                    frames = clip_node.create_group("frames")
                    ffc_frames = []
                    stats = ClipStats()
                    cropped_stats = ClipStats()
                    num_frames = 0
                    cptv_frames = []
                    region_adjust = 0
                    for frame in reader:
                        if frame.background_frame:
                            back_node = frames.create_dataset(
                                "background",
                                frame.pix.shape,
                                chunks=frame.pix.shape,
                                compression="gzip",
                                dtype=frame.pix.dtype,
                            )
                            back_node[:, :] = frame.pix
                            # pre tracker verison 10 there was a bug where back frame was counted in region frame number
                            #  so frame 0 is actually the background frame, no tracks shouild ever start at 0
                            if tracker_version < 10:
                                region_adjust = -1
                                logging.info("Adjusting regions by %s", region_adjust)
                            continue
                        ffc = is_affected_by_ffc(frame)
                        if ffc:
                            ffc_frames.append(num_frames)

                        cptv_frames.append(frame.pix)
                        stats.add_frame(frame.pix)
                        cropped_stats.add_frame(clip.crop_rectangle.subimage(frame.pix))
                        num_frames += 1
                    cptv_frames = np.uint16(cptv_frames)
                    thermal_node = frames.create_dataset(
                        "thermals",
                        cptv_frames.shape,
                        chunks=(1, *cptv_frames.shape[1:]),
                        compression="gzip",
                        dtype=cptv_frames.dtype,
                    )

                    thermal_node[:, :, :] = cptv_frames
                    stats.completed()
                    cropped_stats.completed()
                    group_attrs = clip_node.attrs
                    group_attrs["clip_id"] = r_id
                    group_attrs["num_frames"] = np.uint16(num_frames)
                    group_attrs["ffc_frames"] = np.uint16(ffc_frames)
                    group_attrs["device_id"] = metadata["deviceId"]
                    stationId = metadata.get("stationId", 0)
                    group_attrs["station_id"] = stationId
                    group_attrs["crop_rectangle"] = np.uint8(
                        clip.crop_rectangle.to_ltrb()
                    )
                    group_attrs["max_temp"] = np.uint16(cropped_stats.max_temp)
                    group_attrs["min_temp"] = np.uint16(cropped_stats.min_temp)
                    group_attrs["mean_temp"] = np.uint16(cropped_stats.mean_temp)
                    group_attrs["frame_temp_min"] = np.uint16(
                        cropped_stats.frame_stats_min
                    )
                    group_attrs["frame_temp_max"] = np.uint16(
                        cropped_stats.frame_stats_max
                    )
                    group_attrs["frame_temp_median"] = np.uint16(
                        cropped_stats.frame_stats_median
                    )
                    group_attrs["frame_temp_mean"] = np.uint16(
                        cropped_stats.frame_stats_mean
                    )
                    group_attrs["start_time"] = clip.video_start_time.isoformat()
                    group_attrs["res_x"] = clip.res_x
                    group_attrs["res_y"] = clip.res_y
                    if camera_model is not None:
                        group_attrs["model"] = clip.camera_model

                    if triggered_temp_thresh is not None:
                        group_attrs["temp_thresh"] = triggered_temp_thresh

                    if clip.tags:
                        clip_tags = []
                        for track in clip.tags:
                            if track["what"]:
                                clip_tags.append(track["what"])
                            elif track["detail"]:
                                clip_tags.append(track["detail"])
                        group_attrs["tags"] = clip_tags

                tracks_group = clip_node.create_group("tracks")

                tracks = metadata.get("Tracks", [])
                for track in tracks:
                    track_id = track["id"]

                    track_group = tracks_group.create_group(str(track_id))

                    node_attrs = track_group.attrs
                    node_attrs["id"] = track_id
                    tags = track.get("tags", [])
                    tag = Track.get_best_human_tag(
                        tags, self.config.load.tag_precedence, 0
                    )

                    master_tag = [
                        t
                        for t in tags
                        if t.get("automatic")
                        and not isinstance(t.get("data", ""), str)
                        and t.get("data", {}).get("name") == "Master"
                    ]
                    if len(master_tag) > 0:
                        master_tag = master_tag[0]
                        node_attrs["ai_tag"] = master_tag["what"]
                        node_attrs["ai_tag_confidence"] = master_tag["confidence"]

                    if tag is not None:
                        node_attrs["human_tag"] = tag["what"]
                        node_attrs["human_tag_confidence"] = tag["confidence"]

                    human_tags = [
                        (t.get("what"), t["confidence"])
                        for t in tags
                        if t.get("automatic", False) != True
                    ]
                    if len(human_tags) > 0:
                        node_attrs["human_tags"] = [h[0] for h in human_tags]
                        node_attrs["human_tags_confidence"] = np.float32(
                            [h[1] for h in human_tags]
                        )

                    start = None
                    end = None

                    prev_frame = None
                    regions = []
                    for i, r in enumerate(track.get("positions")):
                        if isinstance(r, list):
                            region = Region.region_from_array(r[1])
                            if region.frame_number is None:
                                if i == 0:
                                    frame_number = round(r[0] * FPS)
                                    region.frame_number = frame_number
                                else:
                                    region.frame_number = prev_frame + 1
                        else:
                            region = Region.region_from_json(r)
                        if region.frame_number is None:
                            if "frameTime" in r:
                                if i == 0:
                                    region.frame_number = round(r["frameTime"] * 9)
                                else:
                                    region.frame_number = prev_frame + 1
                        # new_f = region.frame_number + region_adjust
                        prev_frame = region.frame_number
                        region.frame_number = region.frame_number + region_adjust
                        assert region.frame_number >= 0
                        regions.append(region.to_array())
                        if start is None:
                            start = region.frame_number
                        end = region.frame_number
                    node_attrs["start_frame"] = start
                    node_attrs["end_frame"] = min(num_frames, end)

                    region_array = np.uint16(regions)
                    regions = track_group.create_dataset(
                        "regions",
                        region_array.shape,
                        chunks=region_array.shape,
                        compression="gzip",
                        dtype=region_array.dtype,
                    )
                    regions[:, :] = region_array
            except:
                logging.error("Error saving file %s", filename, exc_info=True)
                f.close()
                out_file.unlink()
