import json

import numpy as np

from run.base_model.evaluate_model_zoo_ce import _write_json


def test_validation_json_accepts_numpy_scalars(tmp_path) -> None:
    output = tmp_path / "validation.json"
    _write_json(
        output,
        {
            "dataset_size": np.int64(64),
            "validation_ce": np.float32(0.65),
        },
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["dataset_size"] == 64
    assert payload["validation_ce"] == np.float32(0.65).item()
    assert not output.with_suffix(".json.tmp").exists()
