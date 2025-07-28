import subprocess
import os
import sys
import shutil
import inspect
from datetime import datetime


def create_results_folder(base_dir="data", prefix="run"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"{prefix}_{timestamp}"
    full_path = os.path.join(base_dir, folder_name)
    os.makedirs(full_path, exist_ok=False)  # Raises error if folder somehow exists
    return full_path

def export_full_conda_env(target_folder, env_name=None):
    """
    Export the full current conda environment (including packages, versions, and Python)
    to environment.yml in the target folder.

    Parameters
    ----------
    target_folder : str
        Path to the folder where environment.yml will be saved.
    env_name : str or None
        Name of the conda environment to export. If None, uses the active environment.
    """
    os.makedirs(target_folder, exist_ok=True)
    output_path = os.path.join(target_folder, "environment.yml")

    if env_name is None:
        cmd = ["conda", "env", "export", "-f", output_path]
    else:
        cmd = ["conda", "env", "export", "--name", env_name, "-f", output_path]

    print(f"Exporting full conda environment to {output_path} ...")
    try:
        subprocess.run(cmd, check=True)
        print("Export successful!")
    except subprocess.CalledProcessError as e:
        print("Error exporting conda environment:", e)
        sys.exit(1)



def export_pip_requirements(target_folder):
    """
    Export installed pip packages to requirements.txt in the target folder.

    Parameters
    ----------
    target_folder : str
        Path to the folder where requirements.txt will be saved.
    """
    os.makedirs(target_folder, exist_ok=True)
    output_path = os.path.join(target_folder, "requirements.txt")

    print(f"Exporting pip requirements to {output_path} ...")
    try:
        with open(output_path, "w") as f:
            subprocess.run(["pip", "freeze"], stdout=f, check=True)
        print("Export successful!")
    except subprocess.CalledProcessError as e:
        print("Error exporting pip requirements:", e)
        sys.exit(1)


def export_git_commit(target_folder, filename="git_commit.txt"):
    """
    Export the current git commit hash to a text file.

    Parameters
    ----------
    target_folder : str
        Path to the folder where the commit hash file will be saved.
    filename : str
        Name of the file to store the commit hash.
    """
    os.makedirs(target_folder, exist_ok=True)
    output_path = os.path.join(target_folder, filename)

    try:
        # Get the current commit hash
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

        with open(output_path, "w") as f:
            f.write(commit_hash + "\n")

        print(f"Current commit hash '{commit_hash}' exported to {output_path}")
    except subprocess.CalledProcessError as e:
        print("Error retrieving git commit hash:", e)
        sys.exit(1)


def copy_problem_setup(module, target_folder, new_name=None):
    # Find the source .py file for the imported module
    src = inspect.getsourcefile(module) or module.__file__
    if src is None:
        raise RuntimeError("Cannot locate source file for module.")

    # Make sure destination exists
    os.makedirs(target_folder, exist_ok=True)

    # Destination path (optional rename)
    dst_name = new_name if new_name is not None else os.path.basename(src)
    dst = os.path.join(target_folder, dst_name)

    # Copy (copy2 preserves timestamp/metadata)
    shutil.copy2(src, dst)

def copy_main_script(target_folder, src):
    """
    Copies the currently executing main script to the specified folder.
    
    Args:
        results_folder (str): The destination folder path.
        new_name (str, optional): New filename (e.g., "main_backup.py"). If None, keep original filename.
        
    Returns:
        str: Path to the copied file.
    """

    os.makedirs(target_folder, exist_ok=True)
    dst_name = os.path.basename(src)
    dst = os.path.join(target_folder, dst_name)
    shutil.copy2(src, dst)