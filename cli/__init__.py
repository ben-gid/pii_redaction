"""Dataset registry shared by prepare_ds.py and train.py.

Keeping the save location, label info filename, and run name in one place stops the
prepared artifacts and the training inputs from drifting apart.
"""

from pydantic import BaseModel


class DatasetConfig(BaseModel, frozen=True):
    hf_name: str      # source dataset on the Hub
    dir_name: str     # save_to_disk directory under data/
    label_info: str   # label info JSON under data/
    run_suffix: str   # models/<variant>-<run_suffix>


DATASETS = {
    "ai4privacy": DatasetConfig(
        hf_name="ai4privacy/pii-masking-300k",
        dir_name="cleaned_ai4privacy_300k_pii",
        label_info="label_info.json",
        run_suffix="bf16-weighted-trainer",
    ),
    "nemotron": DatasetConfig(
        hf_name="nvidia/Nemotron-PII",
        dir_name="nemotron-pii",
        label_info="nemotron-label_info.json",
        run_suffix="nemotron",
    ),
}
