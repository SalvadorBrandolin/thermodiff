import subprocess
import sys

def install_in_colab():
    if "google.colab" in sys.modules:
        subprocess.run(
            ["pip", "install", "git+https://github.com/SalvadorBrandolin/thermodiff"],
            check=True
        )
