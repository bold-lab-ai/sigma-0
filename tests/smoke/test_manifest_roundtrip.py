from lobpipeline.manifests import RunManifest, ScoreManifest


def test_manifest_models() -> None:
    run = RunManifest(run_name="smoke", source_path="inference/smoke")
    score = ScoreManifest(run_name=run.run_name, score_path="scores/smoke")
    assert score.run_name == "smoke"
    assert run.source_path.endswith("smoke")

