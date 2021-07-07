from pathlib import Path

def create_folder(path):
    """
    Creating Folder given the path that can be nested

    Args:
        path: The path to the folder. 
    """
    Path(path).mkdir(parents=True, exist_ok=True)
