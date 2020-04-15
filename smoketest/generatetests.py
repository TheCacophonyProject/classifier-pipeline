import attr
import os
import yaml
import json
import argparse
import requests
from urllib.parse import urljoin
from pathlib import Path
from testconfig import TestRecording, TestConfig


class API:
    def __init__(self, args, server):
        self._baseurl = server
        self._loginname = args.user
        self._token = self._get_jwt(args.password, "user")
        self._auth_header = {"Authorization": self._token}

    def _get_jwt(self, password, logintype):
        nameProp = logintype + "name"

        url = urljoin(self._baseurl, "/authenticate_" + logintype)
        r = requests.post(url, data={nameProp: self._loginname, "password": password})

        if r.status_code == 200:
            return r.json().get("token")
        elif r.status_code == 422:
            raise ValueError(
                "Could not log on as '{}'.  Please check {} name.".format(
                    self._loginname, logintype
                )
            )
        elif r.status_code == 401:
            raise ValueError(
                "Could not log on as '{}'.  Please check password.".format(
                    self._loginname
                )
            )
        else:
            r.raise_for_status()

    def _check_response(self, r):
        if r.status_code in (400, 422):
            j = r.json()
            messages = j.get("messages", j.get("message", ""))
            raise IOError("request failed ({}): {}".format(r.status_code, messages))
        r.raise_for_status()
        return r.json()

    def download_raw(self, recording_id):
        return self._download_recording(recording_id, "downloadRawJWT")

    def query_rec(self, recording_id):
        url = urljoin(self._baseurl, "/api/v1/recordings/{}".format(recording_id))
        r = requests.get(url, headers=self._auth_header)
        return self._check_response(r)

    def _download_recording(self, recording_id, jwt_key):
        d = self.query_rec(recording_id)
        return self._download_signed(d[jwt_key])

    def _download_signed(self, token):
        r = requests.get(
            urljoin(self._baseurl, "/api/v1/signedUrl"),
            params={"jwt": token},
            stream=True,
        )
        r.raise_for_status()
        yield from r.iter_content(chunk_size=4096)

    def get_tracks(self, recording_id):
        url = urljoin(
            self._baseurl, "/api/v1/recordings/{}/tracks".format(recording_id)
        )
        r = requests.get(url, headers=self._auth_header)
        return self._check_response(r)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_folder", help="Root folder to place downloaded files in.")
    parser.add_argument("user", help="API server username")
    parser.add_argument("password", help="API server password")
    parser.add_argument(
        "-s",
        "--server",
        default="https://api.cacophony.org.nz",
        help="CPTV file server URL",
    )
    parser.add_argument(
        "-t",
        "--test_file",
        default="tracking-tests.yml",
        help="File to save generated tests to",
    )
    parser.add_argument(
        "ids", nargs="+", help="List of recording ids to download",
    )

    return parser.parse_args()


def iter_to_file(filename, source, overwrite=True):
    if not overwrite and Path(filename).is_file():
        print("{} already exists".format(filename))
        return False
    with open(filename, "wb") as f:
        for chunk in source:
            f.write(chunk)
    return True


def main():
    args = parse_args()
    print(args.server)
    api = API(args, args.server)
    out_dir = Path(args.out_folder)
    tests = []
    for rec_id in args.ids:
        rec_meta = api.query_rec(rec_id)
        tracks = api.get_tracks(rec_id)
        fullpath = out_dir / rec_id
        tests.append(
            TestRecording.load_from_meta(rec_meta["recording"], tracks, fullpath)
        )
        if iter_to_file(
            fullpath.with_suffix(".cptv"),
            api._download_signed(rec_meta["downloadRawJWT"]),
        ):
            print("Saved {} - {}.cptv".format(rec_id, fullpath))
        else:
            print("error saving {}".format(rec_id))
    test_data = TestConfig(tests, args.server)
    if os.path.exists(args.test_file):
        os.remove(args.test_file)
    with open(args.test_file, "w") as f:
        yaml.dump(test_data, f)


if __name__ == "__main__":
    main()
