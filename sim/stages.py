from dataclasses import dataclass
from pathlib import Path

from lerobot.utils.constants import PRETRAINED_MODEL_DIR

from sim.world import GPSSensorConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_DATA_ROOT = PROJECT_ROOT / "data"
TRAIN_OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "train"


@dataclass(frozen=True)
class StageConfig:
    number: int
    name: str
    instruction: str
    obstacle_count: int
    gps_sensor_config: GPSSensorConfig

    @property
    def label(self) -> str:
        return f"stage {self.number} ({self.name})"

    @property
    def dataset_name(self) -> str:
        return f"stage-{self.number}-{self.name}"

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
        gps_sensor_config=GPSSensorConfig(
            position_noise_std_m=0.12,
            initial_bias_std_m=0.15,
            bias_walk_std_m=0.008,
            outage_probability_per_update=0.02,
            outage_duration_range_s=(0.4, 1.0),
        ),
    ),
    2: StageConfig(
        number=2,
        name="light-obstacles",
        instruction="drive to the yellow goal while avoiding obstacles",
        obstacle_count=3,
        gps_sensor_config=GPSSensorConfig(
            position_noise_std_m=0.18,
            initial_bias_std_m=0.22,
            bias_walk_std_m=0.012,
            outage_probability_per_update=0.03,
            outage_duration_range_s=(0.6, 1.5),
            stale_accuracy_growth_m_per_s=0.9,
        ),
    ),
    3: StageConfig(
        number=3,
        name="dense-obstacles",
        instruction="drive to the yellow goal while avoiding obstacles",
        obstacle_count=10,
        gps_sensor_config=GPSSensorConfig(
            update_period_s=0.25,
            position_noise_std_m=0.28,
            initial_bias_std_m=0.28,
            bias_walk_std_m=0.018,
            outage_probability_per_update=0.06,
            outage_duration_range_s=(1.0, 2.5),
            stale_accuracy_growth_m_per_s=1.2,
        ),
    ),
}


def get_stage_config(stage: int) -> StageConfig:
    try:
        return STAGE_CONFIGS[stage]
    except KeyError as exc:
        available = ", ".join(str(number) for number in sorted(STAGE_CONFIGS))
        raise ValueError(
            f"Unknown stage {stage}. Available stages: {available}."
        ) from exc


def find_latest_checkpoint(stage_config: StageConfig) -> Path:
    output_root = stage_config.train_output_root
    checkpoints = [
        path
        for path in output_root.rglob(PRETRAINED_MODEL_DIR)
        if path.is_dir() and path.parent.parent.name == "checkpoints"
    ]
    if not checkpoints:
        raise FileNotFoundError(
            f"No trained checkpoint found under {output_root}"
        )
    return max(checkpoints, key=lambda path: path.stat().st_mtime)
