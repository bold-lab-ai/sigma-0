from lobpipeline.pipeline.downstream_metrics import downstream_metric_names
from lobpipeline.lob_bench.scoring import score_run


def test_score_schema_metric_order() -> None:
    assert downstream_metric_names() == ("l1", "wasserstein", "ks")


def test_score_plan_metric_order() -> None:
    plan = score_run("smoke")
    assert plan.metrics == ("l1", "wasserstein", "ks")
