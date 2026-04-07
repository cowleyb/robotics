from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_DATA_ROOT = PROJECT_ROOT / "data" / "lerobot"
TRAIN_OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "train"


@dataclass(frozen=True)
class StageConfig:
    number: int
    name: str
    instruction: str
    obstacle_count: int

    @property
    def label(self) -> str:
        return f"stage {self.number} ({self.name})"

    @property
    def dataset_name(self) -> str:
        return f"robotics-sim-stage-{self.number}-{self.name}"

    @property
    def repo_id(self) -> str:
        return f"local/{self.dataset_name}"

    @property
    def dataset_root(self) -> Path:
        return LEROBOT_DATA_ROOT / self.dataset_name

    @property
    def train_output_root(self) -> Path:
        return TRAIN_OUTPUTS_ROOT / f"stage-{self.number}-{self.name}"


STAGE_CONFIGS = {
    1: StageConfig(
        number=1,
        name="goal-only",
        instruction="drive to the yellow goal",
        obstacle_count=0,
    ),
    2: StageConfig(
        number=2,
        name="light-obstacles",
        instruction="drive to the yellow goal while avoiding obstacles",
        obstacle_count=3,
    ),
    3: StageConfig(
        number=3,
        name="dense-obstacles",
        instruction="drive to the yellow goal while avoiding obstacles",
        obstacle_count=10,
    ),
}


def get_stage_config(stage: int) -> StageConfig:
    try:
        return STAGE_CONFIGS[stage]
    except KeyError as exc:
        available = ", ".join(str(number) for number in sorted(STAGE_CONFIGS))
        raise ValueError(f"Unknown stage {stage}. Available stages: {available}.") from exc


def find_latest_checkpoint(stage_config: StageConfig) -> Path:
    checkpoints = list(stage_config.train_output_root.glob("*/*/checkpoints/*/pretrained_model"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No trained checkpoint found under {stage_config.train_output_root}"
        )
    return max(checkpoints, key=lambda path: path.stat().st_mtime)
