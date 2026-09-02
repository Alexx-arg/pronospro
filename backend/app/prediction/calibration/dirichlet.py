# ruff: noqa: N803, N806
"""Dirichlet calibration (Sprint 5.4)."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.prediction.calibration.base import (
    CLIP_EPS,
    _array_to_probas,
    _clip_for_log,
    _probas_to_array,
    _softmax,
    _validate_probas_and_targets,
)
from app.prediction.contracts import CalibratorKind, MatchProbabilities

LAMBDA_L2_DEFAULT: float = 0.1


class DirichletCalibrator:
    """Dirichlet calibration: ``p_cal = softmax(W log p + b)``, ``W=I+A``."""

    kind: CalibratorKind = CalibratorKind.DIRICHLET

    def __init__(
        self,
        *,
        lambda_l2: float = LAMBDA_L2_DEFAULT,
        max_iter: int = 500,
    ) -> None:
        if lambda_l2 < 0:
            raise ValueError(f"lambda_l2 must be >=0, got {lambda_l2}")
        self.lambda_l2: float = float(lambda_l2)
        self.max_iter: int = int(max_iter)
        # Params: W shape (3,3), b shape (3,)
        self.W: Any = np.eye(3, dtype=np.float64)
        self.b: Any = np.zeros(3, dtype=np.float64)
        self._fitted: bool = False

    # ------------------------------------------------------------------
    def _pack(self, A: Any, b: Any) -> Any:
        return np.concatenate([A.reshape(-1), b.reshape(-1)])

    def _unpack(self, x: Any) -> tuple[Any, Any]:
        A = x[:9].reshape(3, 3)
        b = x[9:12].reshape(3)
        return A, b

    def _nll(self, x: Any, log_p: Any, targets: Any) -> float:
        A, b = self._unpack(x)
        W = np.eye(3) + A
        logits = log_p @ W.T + b  # (n,3) @ (3,3).T → (n,3)
        proba = _softmax(logits)
        n = targets.shape[0]
        p_true = proba[np.arange(n), targets]
        p_true = _clip_for_log(p_true, eps=1e-15)
        nll = float(-np.log(p_true).mean())
        # L2 on A
        nll += self.lambda_l2 * float((A * A).sum())
        return nll

    def _grad_numerical(self, x: Any, log_p: Any, targets: Any, eps: float = 1e-5) -> Any:
        # Finite-difference gradient (12 params) — deterministic, no autograd needed
        g = np.zeros_like(x)
        base = self._nll(x, log_p, targets)
        for i in range(x.shape[0]):
            xp = np.array(x, copy=True)
            xp[i] += eps  # type: ignore[index]
            fp = self._nll(xp, log_p, targets)
            xp[i] -= 2 * eps  # type: ignore[index]
            fm = self._nll(xp, log_p, targets)
            g[i] = (fp - fm) / (2 * eps)
            _ = base
        return g

    def fit(
        self,
        raw_probs: Any,
        targets: Any,
    ) -> DirichletCalibrator:
        arr, t = _validate_probas_and_targets(raw_probs, targets)
        if arr.shape[0] == 0:
            raise ValueError("cannot fit dirichlet with n=0")
        clipped = _clip_for_log(arr, eps=CLIP_EPS)
        log_p = np.log(clipped)

        # Try scipy if available
        x0 = self._pack(np.zeros((3, 3)), np.zeros(3))
        best = self._fit_with_scipy(x0, log_p, t)
        if best is not None:
            A, b = self._unpack(best)
            self.W = np.eye(3) + A
            self.b = b
            self._fitted = True
            return self

        # Fallback: simple gradient descent (deterministic)
        best = self._fit_gradient_descent(x0, log_p, t)
        A, b = self._unpack(best)
        self.W = np.eye(3) + A
        self.b = b
        self._fitted = True
        return self

    def _fit_with_scipy(self, x0: Any, log_p: Any, targets: Any) -> Any | None:
        try:
            import scipy.optimize as _opt  # type: ignore[import-untyped]

            def _obj(x: Any) -> float:
                return self._nll(x, log_p, targets)

            # Use L-BFGS-B; bounds not needed (all real)
            res = _opt.minimize(
                _obj,
                x0,
                method="L-BFGS-B",
                options={"maxiter": self.max_iter},
            )
            if res.success and np.isfinite(res.fun):
                return np.asarray(res.x, dtype=np.float64)
        except Exception:
            pass
        return None

    def _fit_gradient_descent(self, x0: Any, log_p: Any, targets: Any) -> Any:
        x = np.array(x0, dtype=np.float64)
        # Simple Adam
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        beta1, beta2 = 0.9, 0.999
        eps_adam = 1e-8
        lr = 0.05
        best_x = np.array(x, copy=True)
        best_nll = self._nll(x, log_p, targets)
        for it in range(1, self.max_iter + 1):
            g = self._grad_numerical(x, log_p, targets)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g * g)
            m_hat = m / (1 - beta1**it)
            v_hat = v / (1 - beta2**it)
            x = x - lr * m_hat / (np.sqrt(v_hat) + eps_adam)
            nll = self._nll(x, log_p, targets)
            if nll < best_nll - 1e-9:
                best_nll = nll
                best_x = np.array(x, copy=True)
            # Early stop if gradient small
            if np.linalg.norm(g) < 1e-6:
                break
            # Decay lr if no improvement for many steps (simple)
            if it % 100 == 0:
                lr *= 0.9
        return best_x

    # ------------------------------------------------------------------
    def transform(self, probs: Any) -> list[MatchProbabilities]:
        arr = _probas_to_array(probs)
        from app.prediction.metrics.classification import validate_multiclass_probabilities

        arr = validate_multiclass_probabilities(arr)
        clipped = _clip_for_log(arr, eps=CLIP_EPS)
        log_p = np.log(clipped)
        W = np.asarray(self.W, dtype=np.float64)
        b = np.asarray(self.b, dtype=np.float64)
        logits = log_p @ W.T + b
        cal = _softmax(logits)
        return _array_to_probas(np.array(cal, copy=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "dirichlet",
            "W": np.asarray(self.W).tolist(),
            "b": np.asarray(self.b).tolist(),
            "lambda_l2": float(self.lambda_l2),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DirichletCalibrator:
        if data.get("kind") != "dirichlet":
            raise ValueError(f"expected kind dirichlet, got {data.get('kind')}")
        obj = cls(lambda_l2=float(data.get("lambda_l2", LAMBDA_L2_DEFAULT)))
        obj.W = np.asarray(data["W"], dtype=np.float64)
        obj.b = np.asarray(data["b"], dtype=np.float64)
        if obj.W.shape != (3, 3) or obj.b.shape != (3,):
            raise ValueError(f"invalid W/b shapes {obj.W.shape} {obj.b.shape}")
        obj._fitted = True
        return obj


__all__ = ["DirichletCalibrator", "LAMBDA_L2_DEFAULT"]
