import subprocess
import os
import sys


def export_conda_env(target_folder, env_name=None):
    """
    Export the current conda environment to environment.yml in the target folder.

    Parameters
    ----------
    target_folder : str
        Path to the folder where environment.yml will be saved.
    env_name : str or None
        Name of the conda environment to export. If None, uses the active environment.
    """
    os.makedirs(target_folder, exist_ok=True)
    output_path = os.path.join(target_folder, "environment.yml")

    # Construct the conda command
    if env_name is None:
        cmd = ["conda", "env", "export", "--from-history", "-f", output_path]
    else:
        cmd = ["conda", "env", "export", "--name", env_name, "--from-history", "-f", output_path]

    print(f"Exporting conda environment to {output_path} ...")
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