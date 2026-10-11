from lobpipeline.lob_bench.metrics import metric_names


def test_lobbench_metric_names() -> None:
    assert metric_names() == ("l1", "wasserstein", "ks")

