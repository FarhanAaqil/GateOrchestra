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
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from gate.feature_extractor import feature_backend
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


# Abstract base class


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
    def predict(
        self,
        features: GateFeatures,
        k: int,
        probe_tokens: int,
        threshold: float | None = None,
    ) -> GateDecision:
        """Predict gate decision for a single task.

        Args:
            features:     Extracted GateFeatures for the task.
            k:            Token budget multiplier (token_budget_cap = k × probe_tokens).
            probe_tokens: Probe token count (used to compute token_budget_cap).
            threshold:    Optional probability threshold for ESCALATE decision.
        """
        ...

    def save(self, path: str | Path) -> None:
        """Serialize the classifier to disk, recording how it was built.

        The pickle records the feature backend (spaCy or regex) and the
        scikit-learn version, which :meth:`load` checks against the current
        environment.
        """
        self.build_info = {
            "feature_backend": feature_backend(),
            "sklearn_version": sklearn.__version__,
        }
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved {self.name} to {path} ({self.build_info})")

    @classmethod
    def load(cls, path: str | Path) -> GateClassifier:
        """Deserialize a classifier and warn if it was built in a different environment."""
        target_path = Path(path)
        with target_path.open("rb") as f:
            obj = pickle.load(f)
        build_info = getattr(obj, "build_info", None)
        if build_info is None:
            logger.warning(
                f"Gate at {path} has no build info; its feature backend and "
                "scikit-learn version are unknown."
            )
        else:
            current = {"feature_backend": feature_backend(), "sklearn_version": sklearn.__version__}
            for key, value in current.items():
                if build_info.get(key) != value:
                    logger.warning(
                        f"Gate at {path} was built with {key}={build_info.get(key)!r} "
                        f"but this environment has {value!r}; predictions may differ."
                    )
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
            decision=cast(Literal["STOP", "ESCALATE"], label),
            confidence=round(confidence, 4),
            token_budget_cap=k * probe_tokens if label == "ESCALATE" else None,
            gate_type="learned",
        )


# Concrete implementations


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

    def predict(
        self,
        features: GateFeatures,
        k: int,
        probe_tokens: int,
        threshold: float | None = None,
    ) -> GateDecision:
        if not self._is_trained:
            raise RuntimeError("Call train() before predict()")
        x = features_to_array(features).reshape(1, -1)
        x_scaled = self._scaler.transform(x)
        proba = self._model.predict_proba(x_scaled)[0]

        if threshold is not None and self._classes and "ESCALATE" in self._classes:
            esc_idx = self._classes.index("ESCALATE")
            esc_prob = float(proba[esc_idx])
            if esc_prob >= threshold:
                label = "ESCALATE"
                confidence = esc_prob
            else:
                label = "STOP"
                stop_idx = self._classes.index("STOP") if "STOP" in self._classes else (1 - esc_idx)
                confidence = float(proba[stop_idx])
        else:
            label = str(self._model.predict(x_scaled)[0])
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

    def predict(
        self,
        features: GateFeatures,
        k: int,
        probe_tokens: int,
        threshold: float | None = None,
    ) -> GateDecision:
        if not self._is_trained:
            raise RuntimeError("Call train() before predict()")
        x = features_to_array(features).reshape(1, -1)
        proba = self._model.predict_proba(x)[0]

        if threshold is not None and self._classes and "ESCALATE" in self._classes:
            esc_idx = self._classes.index("ESCALATE")
            esc_prob = float(proba[esc_idx])
            if esc_prob >= threshold:
                label = "ESCALATE"
                confidence = esc_prob
            else:
                label = "STOP"
                stop_idx = self._classes.index("STOP") if "STOP" in self._classes else (1 - esc_idx)
                confidence = float(proba[stop_idx])
        else:
            label = str(self._model.predict(x)[0])
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
            early_stopping=False,
        )
        self._classes: list[str] | None = None

    def train(self, features: list[GateFeatures], labels: list[str]) -> None:
        X = np.stack([features_to_array(f) for f in features])
        # Encode string labels to integers — workaround for sklearn MLPClassifier
        # early_stopping bug that calls np.isnan() on string label arrays.
        unique_labels = sorted(set(labels))
        label_to_int = {lab: i for i, lab in enumerate(unique_labels)}
        y_int = np.array([label_to_int[lab] for lab in labels])
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y_int)
        # Restore string class names in the expected order
        self._classes = unique_labels
        self._is_trained = True
        best_loss = getattr(self._model, "best_loss_", None)
        loss_str = f"{best_loss:.4f}" if best_loss is not None else "N/A"
        logger.info(f"MLPGate trained. Best val loss: {loss_str}")

    def predict(
        self,
        features: GateFeatures,
        k: int,
        probe_tokens: int,
        threshold: float | None = None,
    ) -> GateDecision:
        if not self._is_trained:
            raise RuntimeError("Call train() before predict()")
        x = features_to_array(features).reshape(1, -1)
        x_scaled = self._scaler.transform(x)
        # Model predicts integer class indices; decode back to string label
        int_pred = int(self._model.predict(x_scaled)[0])
        label = self._classes[int_pred]  # type: ignore[index]
        proba = self._model.predict_proba(x_scaled)[0]
        confidence = float(proba[int_pred])
        return self._make_decision(features.task_id, label, confidence, k, probe_tokens)


class ConstantGate(GateClassifier):
    """Always predicts one label.

    Produced by calibration when every training label is the same class, so
    there is nothing to learn. Reporting this gate (rather than substituting a
    gate trained elsewhere) keeps a "the gate always stops" result visible.
    """

    def __init__(self, label: str = "STOP") -> None:
        super().__init__("ConstantGate")
        if label not in ("STOP", "ESCALATE"):
            raise ValueError(f"label must be 'STOP' or 'ESCALATE', got {label!r}")
        self.label = label

    def train(self, features: list[GateFeatures], labels: list[str]) -> None:
        unique = set(labels)
        if len(unique) != 1:
            raise ValueError(f"ConstantGate needs single-class labels, got {sorted(unique)}")
        self.label = unique.pop()
        self._is_trained = True

    def predict(
        self,
        features: GateFeatures,
        k: int,
        probe_tokens: int,
        threshold: float | None = None,
    ) -> GateDecision:
        if not self._is_trained:
            raise RuntimeError("Call train() before predict()")
        return self._make_decision(features.task_id, self.label, 1.0, k, probe_tokens)


def benchmark_gate_paths(models_dir: Path) -> list[Path]:
    """Gates written by benchmark calibration (best_gate_seed_*.pkl), newest first."""
    return sorted(
        models_dir.glob("best_gate_seed_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True
    )


def gate_provenance(path: Path, gate: GateClassifier) -> dict[str, Any]:
    """Describe where a saved gate came from."""
    seed = re.search(r"best_gate_seed_(\d+)\.pkl$", path.name)
    return {
        "path": str(path),
        "seed": int(seed.group(1)) if seed else None,
        "classifier": gate.name,
        "degenerate": isinstance(gate, ConstantGate),
        "build_info": getattr(gate, "build_info", None),
    }


def load_provenanced_gate(path: Path) -> GateClassifier | None:
    """Load a trained gate that records how it was built; None otherwise.

    Gates without build info predate calibration recording it, so nobody can
    say what data trained them (some were trained on simulated data).
    """
    try:
        gate = GateClassifier.load(path)
    except Exception as err:
        logger.warning(f"Could not load gate from {path}: {err}")
        return None
    if not getattr(gate, "_is_trained", False):
        return None
    if getattr(gate, "build_info", None) is None:
        logger.warning(f"Skipping gate {path}: no build info, provenance unknown")
        return None
    return gate


def load_benchmark_gate(models_dir: Path) -> tuple[GateClassifier, dict[str, Any]] | None:
    """Newest benchmark-calibrated gate with known provenance, or None."""
    for path in benchmark_gate_paths(models_dir):
        gate = load_provenanced_gate(path)
        if gate is not None:
            return gate, gate_provenance(path, gate)
    return None


# Factory

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
