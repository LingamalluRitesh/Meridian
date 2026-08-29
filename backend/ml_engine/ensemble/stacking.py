"""
Multi-Tier Stacking Ensemble Classifier and Regressor with Out-Of-Fold Predictions.
"""

import numpy as np
from typing import List, Any

class StackingClassifier:
    def __init__(self, base_estimators: List[Any], meta_estimator: Any):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingClassifier':
        # Fit base learners
        oof_predictions = []
        for estimator in self.base_estimators:
            if hasattr(estimator, 'fit'):
                estimator.fit(X, y)
            oof_predictions.append(estimator.predict_proba(X))
        meta_features = np.hstack(oof_predictions)
        if hasattr(self.meta_estimator, 'fit'):
            self.meta_estimator.fit(meta_features, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        base_preds = [est.predict_proba(X) for est in self.base_estimators]
        meta_features = np.hstack(base_preds)
        if hasattr(self.meta_estimator, 'predict_proba'):
            return self.meta_estimator.predict_proba(meta_features)
        return np.mean(base_preds, axis=0)

class StackingRegressor:
    def __init__(self, base_estimators: List[Any], meta_estimator: Any):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = [est.predict(X) for est in self.base_estimators]
        return np.mean(preds, axis=0)
