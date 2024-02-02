"""
# Download latest model relase from  https://github.com/TheCacophonyProject/AI-Model
# with prefix server-
# place this in /etc/cacophony/classifier.yaml
"""

import sys
import requests
from pathlib import Path
import subprocess
import yaml
import json


def get_headers():
    return {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def run_command(cmd, timeout=None):
    proc = subprocess.run(
        cmd,
        shell=True,
        encoding="ascii",
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return proc.stdout


def get_releases():
    r = requests.get(
        "https://api.github.com/repos/TheCacophonyProject/AI-Model/releases",
        headers=get_headers(),
    )
    r.raise_for_status()

    return r.json()


def download_file(url, name, extract_to):
    print(f"Downloading {url}")
    response = requests.get(url)
    open(name, "wb").write(response.content)
    out_dir = Path("latest-model")
    out_dir.mkdir(exist_ok=True)
    print(f"Extracting {name} to {extract_to}")
    extract_to.mkdir()
    run_command(f"tar -xzf {name} -C {extract_to} --strip-components=1")
    Path(name).unlink()


def main():
    releases = get_releases()
    releases = [
        release
        for release in releases
        if release["tag_name"].startswith("server") and not release["prerelease"]
    ]
    releases = sorted(releases, key=lambda r: r["tag_name"], reverse=True)
    if len(releases) == 0:
        print("No releases found for server")
        sys.exit(0)
    release = releases[0]
    print("Using release ", release["tag_name"])
    if len(release["assets"]) == 0:
        print("Release has no files")
        sys.exit(0)
    asset = release["assets"][0]

    # check existing models
    existing_models = Path("/etc/cacophony/models")
    model_versions = []
    if existing_models.exists():
        for dir in existing_models.iterdir():
            if dir.is_dir():
                model_versions.append(dir.name)
    else:
        existing_models.mkdir()
    print("Have existing", model_versions)
    if release["tag_name"] in model_versions:
        print("Already have latest model")
        sys.exit(0)

    # Download and extract new model
    print("Downloading model", asset["name"])
    model_dir = existing_models / release["tag_name"]
    download_file(asset["browser_download_url"], asset["name"], model_dir)

    # Get model metadata to extract model name
    meta_data = model_dir.glob("*.json")
    print("looking for meta in ", "latest-model")
    meta_data = next(meta_data)
    print("Loading metadata ", meta_data)
    with open(meta_data) as stream:
        model_meta = json.load(stream)
    model_name = model_meta["name"]
    print("Model name is ", model_name)

    config_file = "/etc/cacophony/classifier.yaml"
    print("Loading ", config_file)
    with open(config_file) as stream:
        config = yaml.safe_load(stream)
    if config is None:
        config = {"classify": {"models": []}}

    if config.get("classify") is None:
        config["classify"] = {}
    if len(config.get("classify", {}).get("models", [])) == 0:
        config_model = {
            "name": model_name,
            "model_file": str(model_dir / "saved_model.pb"),
            "id": 1,
        }
        config["classify"]["models"] = [config_model]
    else:
        config_model = config["classify"]["models"][0]
        config_model["name"] = model_name
        config_model["model_file"] = str(model_dir / "saved_model.pb")
        config_model["id"] = config_model["id"] + 1

    with open(config_file, "w") as stream:
        raw = yaml.dump(config, stream)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error running ", e)
