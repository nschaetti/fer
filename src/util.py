"""Generic JSON/PKL persistence helpers."""

import os
import json
import pickle


def save_json(save_dir, name, item):
    """Persist an object to JSON if a target directory is provided.

    Args:
        save_dir: Directory path for outputs.
        name: Base filename without extension.
        item: JSON-serializable Python object.
    """
    if save_dir is not None:
        os.makedirs(f"{save_dir}/", exist_ok=True)
        with open(f"{save_dir}/{name}.json", "w") as f:
            json.dump(item, f)
        # end with
    # end if
# end def save_json

def load_json(load_dir, name):
    """Load a JSON file if the directory is provided; otherwise return None.

    Args:
        load_dir: Directory containing the JSON artifact.
        name: Base filename without extension.

    Returns:
        Parsed Python object or `None` if no directory is supplied.
    """
    if load_dir is not None:
        with open(f"{load_dir}/{name}.json", "r") as f:
            return json.load(f)
        # end with
    else:
        return None
    # end if
# end def load_json

def save_pkl(save_dir, name, item):
    """Persist a Python object as pickle within the target directory.

    Args:
        save_dir: Directory path for outputs.
        name: Base filename without extension.
        item: Python object to pickle.
    """
    if save_dir is not None:
        os.makedirs(f"{save_dir}/", exist_ok=True)
        with open(f"{save_dir}/{name}.pkl", "wb") as f:
            pickle.dump(item, f)
        # end with
    # end if
# end def save_pkl


def load_pkl(load_dir, name):
    """Load a pickle file when the directory argument is provided.

    Args:
        load_dir: Directory containing the pickle artifact.
        name: Base filename without extension.

    Returns:
        Python object from the pickle file or `None` if not provided.
    """
    if load_dir is not None:
        with open(f"{load_dir}/{name}.pkl", "rb") as f:
            return pickle.load(f)
        # end with
    else:
        return None
    # end if
# end def load_pkl
