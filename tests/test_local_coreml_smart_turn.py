#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LocalCoreMLSmartTurnAnalyzer predicts synchronously, as BaseSmartTurn calls it.

coremltools, torch and transformers are not installed here, so the module is
imported with stand-ins for those three names only, each doing the arithmetic
the method needs with numpy. The code under test is the real
``_predict_endpoint``, driven through the real
``BaseSmartTurn._process_speech_segment``, which calls it synchronously and
subscripts the result.
"""

import importlib
import sys
import types
import warnings
from unittest.mock import patch

import numpy as np

from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn

MODULE = "pipecat.audio.turn.smart_turn.local_coreml_smart_turn"


def _softmax(t, dim):
    e = np.exp(t - np.max(t, axis=dim, keepdims=True))
    return e / e.sum(axis=dim, keepdims=True)


def _stand_ins() -> dict:
    torch = types.ModuleType("torch")
    torch.tensor = lambda x: np.asarray(x, dtype=np.float64)
    torch.nn = types.SimpleNamespace(functional=types.SimpleNamespace(softmax=_softmax))
    coremltools = types.ModuleType("coremltools")
    transformers = types.ModuleType("transformers")
    transformers.AutoFeatureExtractor = object
    return {"torch": torch, "coremltools": coremltools, "transformers": transformers}


class _FeatureExtractor:
    def __call__(self, audio, **kwargs):
        return {"input_features": np.asarray(audio)[None, :]}


class _Model:
    def predict(self, inputs):
        return {"logits": np.array([[0.0, 4.0]])}  # class 1 ("complete") wins


def _analyzer():
    with patch.dict(sys.modules, _stand_ins()):
        sys.modules.pop(MODULE, None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            module = importlib.import_module(MODULE)
        cls = module.LocalCoreMLSmartTurnAnalyzer
    analyzer = object.__new__(cls)  # skip model loading; the method is what is under test
    BaseSmartTurn.__init__(analyzer, sample_rate=16_000)
    analyzer.set_sample_rate(16_000)
    analyzer._turn_processor = _FeatureExtractor()
    analyzer._turn_model = _Model()
    return analyzer


def test_prediction_reaches_the_base_as_data():
    analyzer = _analyzer()
    for _ in range(10):
        analyzer.append_audio(np.full(320, 1000, dtype=np.int16).tobytes(), True)

    state, metrics = analyzer._process_speech_segment(analyzer._audio_buffer)

    assert state == EndOfTurnState.COMPLETE
    assert metrics is not None and metrics.probability > 0.9
