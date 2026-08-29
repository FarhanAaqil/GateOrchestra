"""
gate/classifier.py
==================
Learned gate classifiers.

Architecture: Abstract base class + 3 concrete implementations
  LogRegGate   – logistic regression (interpretable baseline)
  GBTGate      – gradient boosted trees (expected best performer)
  MLPGate      – multi-layer perceptron (only if GBT underperforms)

All classifiers wrap scikit-learn estimators and use GateFeatures as input.

Person 3 owns this file.
"""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from shared.schemas import GateDecision, GateFeatures

logger = logging.getLogger(__name__)

# Feature ordering — MUST be consistent across train and predict
FEATURE_NAMES: list[str] = [
    "consistency_score",
    "probe_tokens",
    "question_word_count",
    "entity_count",
    "clause_count",
    "has_context",
    "estimated_depth",
    "estimated_parallel",
]


def features_to_array(features: GateFeatures) -> np.ndarray:
    """Convert a GateFeatures object to a 1D numpy array (model input)."""
    return np.array(
        [
            features.consistency_score,
            features.probe_tokens,
            features.question_word_count,
            features.entity_count,
            features.clause_count,
            float(features.has_context),
            features.estimated_depth or 0.0,
            features.estimated_parallel or 0.0,
        ],
        dtype=float,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base class
# ─────────────────────────────────────────────────────────────────────────────


class GateClassifier(ABC):
    """Abstract base for all gate classifiers.

    Contract:
      - train() fits the model on labeled (features, labels) pairs
      - predict() returns a GateDecision given features and k
      - save() / load() persist and restore the model
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._is_trained: bool = False

    @abstractmethod
    def train(
        self,
        features: list[GateFeatures],
        labels: list[str],  # "STOP" | "ESCALATE"
    ) -> None:
        """Fit the classifier on labeled training data."""
        ...

    @abstractmethod
    def predict(self, features: GateFeatures, k: int, probe_tokens: int) -> GateDecision:
        """Predict gate decision for a single task.

        Args:
            features:     Extracted GateFeatures for the task.
            k:            Token budget multiplier (token_budget_cap = k × probe_tokens).
            probe_tokens: Probe token count (used to compute token_budget_cap).
        """
        ...

    def save(self, path: str | Path) -> None:
        """Serialize the classifier to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved {self.name} to {path}")

    @classmethod
    def load(cls, path: str | Path) -> GateClassifier:
        """Deserialize a classifier from disk."""
        from typing import cast  # noqa: PLC0415

        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        logger.info(f"Loaded gate classifier from {path}")
        return cast(GateClassifier, obj)

    def _make_decision(
        self,
        task_id: str,
        label: str,
        confidence: float,
        k: int,
        probe_tokens: int,
    ) -> GateDecision:
        """Helper to build a GateDecision with correct token_budget_cap logic."""
        return GateDecision(
            task_id=task_id,
            decision=label,
            confidence=round(confidence, 4),
            token_budget_cap=k * probe_tokens if label == "ESCALATE" else None,
            gate_type="learned",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Concrete implementations
# ─────────────────────────────────────────────────────────────────────────────


class LogRegGate(GateClassifier):
    """Logistic Regression gate — interpretable, fast, good baseline."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        super().__init__("LogRegGate")
        self._scaler = StandardScaler()
        self._model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        self._classes: list[str] | None = None

    def train(self, features: list[GateFeatures], labels: list[str]) -> None:
        X = np.stack([features_to_array(f) for f in features])
        y = np.array(labels)
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._classes = list(self._model.classes_)
        self._is_trained = True
        logger.info(f"LogRegGate trained on {len(labels)} examples. Classes: {self._classes}")

    def predict(self, features: GateFeatures, k: int, probe_tokens: int) -> GateDecision:
        if not self._is_trained:
            raise RuntimeError("Call train() before predict()")
        x = features_to_array(features).reshape(1, -1)
        x_scaled = self._scaler.transform(x)
        label = str(self._model.predict(x_scaled)[0])
        proba = self._model.predict_proba(x_scaled)[0]
        idx = self._classes.index(label)  # type: ignore[union-attr]
        confidence = float(proba[idx])
        return self._make_decision(features.task_id, label, confidence, k, probe_tokens)


class GBTGate(GateClassifier):
    """Gradient Boosted Trees gate — expected best performing classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
    ) -> None:
        super().__init__("GBTGate")
        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
        )
        self._classes: list[str] | None = None

    def train(self, features: list[GateFeatures], labels: list[str]) -> None:
        X = np.stack([features_to_array(f) for f in features])
        y = np.array(labels)
        self._model.fit(X, y)
        self._classes = list(self._model.classes_)
        self._is_trained = True
        importances = dict(zip(FEATURE_NAMES, self._model.feature_importances_, strict=True))
        logger.info(f"GBTGate trained. Feature importances: {importances}")

    def predict(self, features: GateFeatures, k: int, probe_tokens: int) -> GateDecision:
        if not self._is_trained:
            raise RuntimeError("Call train() before predict()")
        x = features_to_array(features).reshape(1, -1)
        label = str(self._model.predict(x)[0])
        proba = self._model.predict_proba(x)[0]
        idx = self._classes.index(label)  # type: ignore[union-attr]
        confidence = float(proba[idx])
        return self._make_decision(features.task_id, label, confidence, k, probe_tokens)

    def feature_importances(self) -> dict[str, float]:
        """Return feature importance scores (useful for analysis in Week 11)."""
        if not self._is_trained:
            raise RuntimeError("Call train() first")
        return dict(zip(FEATURE_NAMES, self._model.feature_importances_, strict=True))


class MLPGate(GateClassifier):
    """Small MLP gate — only use if GBT underperforms significantly."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (32, 16),
        max_iter: int = 500,
    ) -> None:
        super().__init__("MLPGate")
        self._scaler = StandardScaler()
        self._model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        )
        self._classes: list[str] | None = None

    def train(self, features: list[GateFeatures], labels: list[str]) -> None:
        X = np.stack([features_to_array(f) for f in features])
        y = np.array(labels)
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._classes = list(self._model.classes_)
        self._is_trained = True
        logger.info(f"MLPGate trained. Best val loss: {self._model.best_loss_:.4f}")

    def predict(self, features: GateFeatures, k: int, probe_tokens: int) -> GateDecision:
        if not self._is_trained:
            raise RuntimeError("Call train() before predict()")
        x = features_to_array(features).reshape(1, -1)
        x_scaled = self._scaler.transform(x)
        label = str(self._model.predict(x_scaled)[0])
        proba = self._model.predict_proba(x_scaled)[0]
        idx = self._classes.index(label)  # type: ignore[union-attr]
        confidence = float(proba[idx])
        return self._make_decision(features.task_id, label, confidence, k, probe_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[GateClassifier]] = {
    "logreg": LogRegGate,
    "gbt": GBTGate,
    "mlp": MLPGate,
}


def make_classifier(name: str, **kwargs) -> GateClassifier:
    """Instantiate a gate classifier by name.

    Args:
        name:   One of "logreg", "gbt", "mlp".
        **kwargs: Passed to the classifier constructor.

    Returns:
        An untrained GateClassifier instance.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown classifier {name!r}. Choose from: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
