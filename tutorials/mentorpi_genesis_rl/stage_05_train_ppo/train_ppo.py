from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from sim_mentor_pi.car_train import main


if __name__ == "__main__":
    main()
