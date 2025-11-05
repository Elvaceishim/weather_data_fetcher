from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    repo_id = os.environ.get("MODEL_REPO_ID", "theelvace/weather-data-fetcher-models")
    files_env = os.environ.get("MODEL_FILES", "rain_xgb_tuned.joblib")
    target_dir = Path(os.environ.get("MODEL_DIR", "models"))

    target_dir.mkdir(parents=True, exist_ok=True)

    filenames = [name.strip() for name in files_env.split() if name.strip()]
    if not filenames:
        print("MODEL_FILES is empty; nothing to download.")
        return

    for filename in filenames:
        print(f"Downloading {filename} from {repo_id} ...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Saved to {local_path}")


if __name__ == "__main__":
    main()
