import json
import os
from pathlib import Path

import yaml

from detr.utils.contants import ROOT


def relpath(path: str | Path, rel_path: str | Path = ROOT) -> str:
    """Return relative path"""
    return os.path.relpath(path, rel_path)


def load_yaml(path: Path | str) -> dict:
    """Load yaml file to dict"""
    with open(path) as file:
        yaml_dct = yaml.safe_load(file)
    return yaml_dct


def save_yaml(dct: dict | list, path: Path | str) -> None:
    """Save dict as yaml file"""
    with open(path, "w") as file:
        yaml.dump(dct, file)


def save_json(data: dict | list, path: Path | str):
    """Save data to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(filepath: Path | str) -> dict:
    with open(filepath) as file:
        data = json.load(file)
        return data


def save_txt_to_file(txt: str, filename: Path | str):
    """Save to txt file"""
    with open(filename, "w") as file:
        file.write(txt)


def list_folders(directory_path: Path | str) -> list[str]:
    path = Path(str(directory_path))
    folders = [str(f) for f in path.iterdir() if f.is_dir()]
    return folders
