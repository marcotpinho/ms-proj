from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from torch import device
    from src.entities.DistanceModel import DistanceModel

@dataclass
class Config:
    max_iterations: int = 1000
    max_time: int = 60
    random_seed: int = 42
    archive_size: int = 40
    front_selection_prob: float = 0.9
    map_file: str = None
    db_path: str = None
    algorithm: str = "unique_vis"
    out_dir: str = "out/"
    img_dir: str = "imgs/"
    save_to_db: bool = False
    plot_results: bool = False
    save_results: bool = False

    predict_distances: bool = False
    model: Optional["DistanceModel"] = None
    model_device: Optional["device"] = None

CONFIG = Config()