import json

from base_model.training import (
    checkpoint_config,
    iter_checkpoint_arrays,
    load_checkpoint_metadata,
    resolve_checkpoint_path,
    summarize_checkpoint,
)


def _write_fake_checkpoint(root, step="123"):
    step_dir = root / step
    metadata_dir = step_dir / "metadata"
    state_dir = step_dir / "state"
    metadata_dir.mkdir(parents=True)
    state_dir.mkdir()
    (step_dir / "_CHECKPOINT_METADATA").write_text("{}", encoding="utf-8")
    (metadata_dir / "metadata").write_text(
        json.dumps(
            {
                "config": {
                    "ssm_type": "mamba3",
                    "d_model": 1024,
                    "n_layers": 6,
                    "msg_seq_len": 500,
                    "use_book_data": True,
                    "opt_config": "muon",
                }
            }
        ),
        encoding="utf-8",
    )
    (state_dir / "_METADATA").write_text(
        json.dumps(
            {
                "tree_metadata": {
                    "('step',)": {"value_metadata": {"write_shape": []}},
                    "('params', 'encoder', 'kernel')": {
                        "value_metadata": {"write_shape": [2, 3]}
                    },
                    "('opt_state', 'count')": {
                        "value_metadata": {"write_shape": []}
                    },
                    "('opt_state', 'muon', 'encoder', 'kernel')": {
                        "value_metadata": {"value_type": "None", "skip_deserialize": True}
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return step_dir


def test_resolve_checkpoint_accepts_manager_root(tmp_path):
    step_dir = _write_fake_checkpoint(tmp_path)
    resolved = resolve_checkpoint_path(tmp_path)
    assert resolved.step == 123
    assert resolved.step_path == step_dir


def test_resolve_checkpoint_accepts_concrete_step_dir(tmp_path):
    step_dir = _write_fake_checkpoint(tmp_path)
    resolved = resolve_checkpoint_path(step_dir)
    assert resolved.manager_path == tmp_path
    assert resolved.step == 123


def test_checkpoint_summary_reads_metadata_and_state_tree(tmp_path):
    _write_fake_checkpoint(tmp_path)
    metadata = load_checkpoint_metadata(tmp_path, step=123)
    assert checkpoint_config(metadata)["ssm_type"] == "mamba3"

    arrays = list(iter_checkpoint_arrays(tmp_path, step=123))
    assert [array.name for array in arrays] == [
        "step",
        "params.encoder.kernel",
        "opt_state.count",
    ]

    summary = summarize_checkpoint(tmp_path)
    assert summary["model"] == "mamba3"
    assert summary["leaf_counts"] == {"opt_state": 1, "params": 1, "step": 1}
    assert summary["metadata_elements"]["params"] == 6


def test_iter_checkpoint_arrays_skips_none_valued_optimizer_slots(tmp_path):
    # Muon (and similar per-parameter-group optimizers) record a `None` slot
    # for params an inner optimizer state doesn't track. Those aren't real
    # arrays and must not be counted as 1-element leaves.
    _write_fake_checkpoint(tmp_path)
    arrays = list(iter_checkpoint_arrays(tmp_path, step=123))
    assert "opt_state.muon.encoder.kernel" not in [array.name for array in arrays]


def test_summarize_checkpoint_verify_shapes_falls_back_without_backing_store(tmp_path):
    # verify_shapes=True tries TensorStore for the true global shape but must
    # degrade gracefully (not raise) when there's no real array store to open,
    # e.g. this synthetic fixture which only has JSON metadata on disk.
    _write_fake_checkpoint(tmp_path)
    summary = summarize_checkpoint(tmp_path, verify_shapes=True)
    assert summary["metadata_elements"]["params"] == 6
