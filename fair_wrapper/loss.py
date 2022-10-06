import numpy as np

from abc import ABC, abstractmethod
from .tree import RuleSelection, Direction
from .types import Classifier, ClippedClassifier
from typing import TypeVar, Generic, Union
from scipy.stats import entropy

T = TypeVar('T')

class DTLossFunction(ABC, Generic[T]):

    @abstractmethod
    def loss(self, clf: Classifier[T], x: T, y: np.ndarray) -> float:
        pass

    @abstractmethod
    def entropy(self, clf: Classifier[T], x: T, y: np.ndarray) -> float:
        pass

    @abstractmethod
    def tree_value(self, clf: Classifier[T], x: T, y: np.ndarray, blackbox_pred: np.ndarray) -> float:
        pass

    def split_tree_values(self, rule: RuleSelection[np.ndarray],
                          clf: Classifier[np.ndarray], x: np.ndarray,
                          y: np.ndarray, blackbox_pred: np.ndarray) -> Union[float, float]:
        left_mask = np.apply_along_axis( rule.hypothsis, 1, x ) == Direction.LEFT

        left_x = x[left_mask, :]
        right_x = x[~left_mask, :]
        left_y = y[left_mask, :]
        right_y = y[~left_mask, :]

        left_blackbox_pred = blackbox_pred[left_mask, :]
        right_blackbox_pred = blackbox_pred[~left_mask, :]

        left_val = self.tree_value(clf, left_x, left_y, left_blackbox_pred)
        right_val = self.tree_value(clf, right_x, right_y, right_blackbox_pred)

        return left_val, right_val

class DTCrossEntropy(DTLossFunction[np.ndarray]):

    def loss(self, clf: ClippedClassifier[np.ndarray], x: np.ndarray, y: np.ndarray, blackbox_pred: np.ndarray = None) -> float:
        if blackbox_pred is not None:
            y_pred = blackbox_pred
        else:
            y_pred = clf(x)
        return cross_entropy(y, y_pred)

    def entropy(self, clf: ClippedClassifier[np.ndarray], x: np.ndarray, y: np.ndarray, blackbox_pred: np.ndarray) -> float:
        y_range = y * 2 - 1  # Correct y range
        logit = (np.log(blackbox_pred) - np.log(1 - blackbox_pred))
        edge = np.average(y_range * logit) / clf.B

        return binary_entropy((1 + edge) / 2)

    def tree_value(self, clf: ClippedClassifier[np.ndarray], x: np.ndarray, y: np.ndarray, blackbox_pred: np.ndarray) -> float:
        y_range = y * 2 - 1  # Correct y range

        logit = (np.log(blackbox_pred) - np.log(1 - blackbox_pred))
        edge = np.float64(np.average(y_range * logit) / clf.B)

        if edge >= 1 - 1e-9:
            log_val = float('inf')
        elif edge <= -1 + 1e-9:
            log_val = float('-inf')
        else:
            log_val = np.log(1 + edge) - np.log(1 - edge)

        return log_val / clf.B
        
class DTCrossEntropyAggressive(DTLossFunction[np.ndarray]):

    def loss(self, clf: ClippedClassifier[np.ndarray], x: np.ndarray, y: np.ndarray, blackbox_pred: np.ndarray = None) -> float:
        if blackbox_pred is not None:
            y_pred = blackbox_pred
        else:
            y_pred = clf(x)
        return cross_entropy(y, y_pred)

    def entropy(self, clf: ClippedClassifier[np.ndarray], x: np.ndarray, y: np.ndarray, blackbox_pred: np.ndarray) -> float:
        y_range = y * 2 - 1  # Correct y range
        logit = (np.log(blackbox_pred) - np.log(1 - blackbox_pred))

        edge_p = np.average(np.maximum(y_range * logit, 0)) / clf.B
        edge_n = - np.average(np.minimum(y_range * logit, 0)) / clf.B

        return np.log(2) * (1 + (edge_p + edge_n) * (binary_entropy(edge_p / (edge_p + edge_n)) / np.log(2) - 1))

    def tree_value(self, clf: ClippedClassifier[np.ndarray], x: np.ndarray, y: np.ndarray, blackbox_pred: np.ndarray) -> float:
        y_range = y * 2 - 1  # Correct y range

        logit = (np.log(blackbox_pred) - np.log(1 - blackbox_pred))

        edge_p = np.average(np.maximum(y_range * logit, 0)) / clf.B
        edge_n = - np.average(np.minimum(y_range * logit, 0)) / clf.B

        return (np.log(edge_p) - np.log(edge_n)) / clf.B

def cross_entropy(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_prob = np.concatenate([y, 1-y], axis=1)
    cur_ypred = np.clip(y_pred, a_min=1e-3, a_max=1-1e-3)
    y_pred_prob = np.concatenate([cur_ypred, 1-cur_ypred], axis=1)

    return np.mean(entropy(y_prob, axis=1) + entropy(y_prob, y_pred_prob, axis=1))

def binary_entropy(y: float) -> float:
    return entropy([y, 1-y])