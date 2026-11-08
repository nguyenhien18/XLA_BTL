"""Apply the locally stored, verified StarDist model; no training is performed."""
from pipeline import run_cli

if __name__ == "__main__":
    run_cli("stardist_python")
