from base_model.training.checkpoint_selection import convergence_step, select_checkpoints


def test_select_checkpoints_uses_validation_only() -> None:
    results = [
        {"step": 1_000, "validation_ce": 0.72},
        {"step": 2_000, "validation_ce": 0.651},
        {"step": 3_000, "validation_ce": 0.61},
        {"step": 4_000, "validation_ce": 0.62},
    ]

    selected = select_checkpoints(results, [0.70, 0.65, 0.60], tolerance=0.01)

    assert selected["best_validation"] == {"step": 3_000, "validation_ce": 0.61}
    assert selected["iso_ce"]["0.7"]["step"] == 1_000
    assert selected["iso_ce"]["0.65"]["step"] == 2_000
    assert selected["iso_ce"]["0.65"]["status"] == "matched"
    assert selected["iso_ce"]["0.6"]["status"] == "outside_tolerance"
    assert selected["iso_ce"]["0.6"]["target_reached"] is False


def test_convergence_step_applies_five_thousand_step_window() -> None:
    results = [
        {"step": 1_000, "validation_ce": 0.70},
        {"step": 5_000, "validation_ce": 0.66},
        {"step": 6_000, "validation_ce": 0.695},
        {"step": 10_000, "validation_ce": 0.655},
    ]

    assert convergence_step(results) == 6_000
