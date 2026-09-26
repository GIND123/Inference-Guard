"""Adversarial stress-testing suite for Milestone 1: Risk Modeling.

Tests edge cases, boundary inputs, degeneracies, and completeness axioms for:
- ModernBertRiskClassifier (src/risk_model/model.py)
- compute_ece and TemperatureScaler (src/risk_model/calibration.py)
- IntegratedGradientsExplainer (src/risk_model/explainer.py)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.product.risk_bands import (
    ATTRIBUTES,
    CUE_IMPORTANCE_FLOOR,
    MAX_CUES_SHOWN,
    Cue,
)
from src.risk_model.calibration import (
    CalibrationMetrics,
    TemperatureScaler,
    compute_ece,
)
from src.risk_model.explainer import IntegratedGradientsExplainer
from src.risk_model.model import ModernBertRiskClassifier


