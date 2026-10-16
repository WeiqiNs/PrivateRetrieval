import numpy as np

from privret.quantize import quantize


def test_quantize_scales_by_the_doc_maximum_and_clips_queries() -> None:
    quantized = quantize(
        docs=np.array([[0.5, -0.25], [0.1, 0.0]], dtype=np.float32),
        queries=np.array([[0.6, -0.1]], dtype=np.float32),
    )
    assert quantized.scale == 254.0
    assert quantized.docs.tolist() == [[127, -64], [25, 0]]
    assert quantized.queries.tolist() == [[127, -25]]
    assert quantized.clipped == 1
