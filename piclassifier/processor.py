"""
classifier-pipeline - this is a server side component that manipulates cptv
files and to create a classification model of animals present
Copyright (C) 2020, The Cacophony Project

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


class Processor:
    def process_frame(self, lepton_frame):
        raise Exception("process_frame method must be overwritten in sub class.")

    def get_recent_frame(self):
        raise Exception("get_recent_frame method must be overwritten in sub class.")

    def disconnected(self):
        raise Exception("disconnected method must be overwritten in sub class.")

    @property
    def res_x(self):
        raise Exception("res_x property must be overwritten in sub class.")

    @property
    def res_y(self):
        raise Exception("res_y property must be overwritten in sub class.")

    @property
    def output_dir(self):
        raise Exception("output_dir property must be overwritten in sub class.")
