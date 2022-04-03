"""
Small script to find out if we are on colab and clone the git repository in this case
"""

import subprocess
import os
import glob
import shutil

repository = "https://github.com/nikoladze/LMU_DA_ML"

if "colab" in str(get_ipython()) and not os.path.exists("git_folder"):
    subprocess.run(["git", "clone", repository, "git_folder"])
    for path in glob.glob("git_folder/*"):
        shutil.move(path, ".")
