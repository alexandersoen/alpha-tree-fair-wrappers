import copy
import warnings
import numpy as np

from pathos.multiprocessing import ProcessPool
from scipy.optimize import minimize_scalar

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Union, List

from .tree import RuleSelection, bool_rule_wrapper
from .types import Classifier, ProtectedInfo
from .loss import DTLossFunction

# Constant
SUPPRESS_MINIMIZE_WARNINGS = True

# Place holder for hypothesis class for alpha tree
T = TypeVar('T')
L = TypeVar('L')

class NoMoreProtectedSplits(Exception):
    pass

class NodeFunction(ABC, Generic[T]):
    loss: DTLossFunction

    def __init__(self, loss: DTLossFunction) -> None:
        """
        Abstract class for determining splits in top-down tree induction.
        """
        super().__init__()
        self.loss = loss

    def rule_selection(self, clf: Classifier[T], x: T,
                       y: np.ndarray, blackbox_pred: np.ndarray) -> Optional[RuleSelection[T]]:
        """
        Determines which rule to select for a split
        """
        orig_ent = self.loss.entropy(clf, x, y, blackbox_pred)
        return self._rule_selection(orig_ent, clf, x, y, blackbox_pred)

    @classmethod
    @abstractmethod
    def _rule_selection(self, orig_ent: float, clf: Classifier[T], x: T,
                        y: np.ndarray, blackbox_pred: np.ndarray) -> Optional[RuleSelection[T]]:
        pass

    def split_tree_values(self, rule: RuleSelection[T], clf: Classifier[T],
                          x: T, y: T, blackbox_pred: T) -> Union[float, float]:
        """
        Calculate the tree values of children nodes from a split
        """
        return self.loss.split_tree_values(rule, clf, x, y, blackbox_pred)

class RootProtectedSplit(NodeFunction[np.ndarray]):
    protected_info: ProtectedInfo
    to_split: List[List[int]]

    def __init__(self, loss: DTLossFunction,
                 protected_info: ProtectedInfo) -> None:
        """
        NodeFunction which makes initial sensitive attribute splits
        """
        super().__init__(loss)
        self.protected_info = protected_info
        self.to_split = copy.deepcopy(self.protected_info.protected_domain)

    def _rule_selection(self, orig_ent: float, clf: Classifier[np.ndarray],
                        x: np.ndarray, y: np.ndarray,
                        blackbox_pred: np.ndarray) -> Optional[RuleSelection[np.ndarray]]:
        # Stop splitting if there is at most 1 left
        if len(self.to_split) < 2:
           raise NoMoreProtectedSplits('No more protected attributes to split.')

        cur_protected = self.to_split.pop(0)

        @bool_rule_wrapper
        def cur_rule(x: np.ndarray) -> bool:
            return np.all(x[self.protected_info.protected_columns] == cur_protected)

        return RuleSelection(cur_rule, 0) #delta_ent)

    def split_tree_values(self, rule: RuleSelection[T], clf: Classifier[T],
                          x: T, y: T, blackbox_pred: T) -> Union[float, float]:
        return (1.0, 1.0)

class UnprotectedProjectionEnumerate(NodeFunction[np.ndarray]):
    protected_info: ProtectedInfo
    min_split: float

    def __init__(self, loss: DTLossFunction, protected_info: ProtectedInfo,
                 min_split: float = 1, min_leaf: Optional[int] = None,
                 max_splits_considered: Optional[int] = None,
                 parallel: bool = True) -> None:
        """
        NodeFunction which makes splits on nonsensitive attributes.
        Splits are all axis aligned.
        If the feature is discrete, then splits are equalities.
        If the feature is continuous, then splits are a threshold inequality.
        """

        super().__init__(loss)
        self.protected_info = protected_info
        self.min_split = min_split
        self.min_leaf = min_leaf
        self.max_splits_considered = max_splits_considered
        self.parallel = parallel

    def _rule_selection(self, orig_ent: float, clf: Classifier[np.ndarray], x:
                        np.ndarray, y: np.ndarray,
                        blackbox_pred: np.ndarray) -> Optional[RuleSelection[np.ndarray]]:

        best_val = float('inf')
        best_rule = None 
        best_col = None
        best_thresh = None
        best_isthresh = False
        best_subset = None

        def col_worker(col_idx):
            cur_col = x[:, col_idx]

            cur_val = None
            cur_thresh = None
            cur_subset = None

            if col_idx in self.protected_info.enumerate_columns:
                cur_isthresh = False

                # Calculate entropy drop
                def delta_entropy(col_vals: List[int]) -> float:
                    cur_x1_idx = np.isin(cur_col, col_vals)
                    cur_x2_idx = np.logical_not(cur_x1_idx)

                    w1 = sum(cur_x1_idx) / len(x)
                    w2 = sum(cur_x2_idx) / len(x)

                    if sum(cur_x1_idx) == 0 or sum(cur_x2_idx) == 0:
                        return float('-inf')
                    elif w1 < self.min_split or w2 < self.min_split:
                        return float('-inf')
                    elif (self.min_leaf is not None and
                        (sum(cur_x1_idx) < self.min_leaf or
                        sum(cur_x2_idx) < self.min_leaf)):
                        return float('-inf')

                    cur_x1 = x[cur_x1_idx, :]
                    cur_x2 = x[cur_x2_idx, :]

                    cur_y1 = y[cur_x1_idx, :]
                    cur_y2 = y[cur_x2_idx, :]

                    cur_blackbox_pred1 = blackbox_pred[cur_x1_idx, :]
                    cur_blackbox_pred2 = blackbox_pred[cur_x2_idx, :]

                    new_ent = w1 * self.loss.entropy(clf, cur_x1, cur_y1, cur_blackbox_pred1) \
                        + w2 * self.loss.entropy(clf, cur_x2, cur_y2, cur_blackbox_pred2)
                    drop = orig_ent - new_ent

                    return drop
                
                col_values = list(np.unique(cur_col))
                if len(col_values) == 1:
                    cur_best_subset = [col_values[0]]
                    cur_best_val = float('-inf')
                else:
                    pos_splits = col_values

                    if self.max_splits_considered is not None and len(pos_splits) > self.max_splits_considered:
                        pos_splits_idx = np.random.choice(
                            range(len(pos_splits)), self.max_splits_considered,
                            replace=False)
                        pos_splits = [pos_splits[i] for i in pos_splits_idx]

                    cur_best_subset = None
                    cur_best_val = float('-inf')
                    for s in pos_splits:
                        v = delta_entropy(s)
                        if v > cur_best_val:
                            cur_best_val = v
                            cur_best_subset = s

                cur_subset = cur_best_subset
                cur_val = -cur_best_val
            else:
                cur_isthresh = True

                # Calculate entropy drop
                def delta_entropy(thresh: float) -> float:
                    cur_x1_idx = cur_col > thresh
                    cur_x2_idx = np.logical_not(cur_x1_idx)

                    if sum(cur_x1_idx) == 0 or sum(cur_x2_idx) == 0:
                        return float('-inf')
                        
                    w1 = sum(cur_x1_idx) / len(x)
                    w2 = sum(cur_x2_idx) / len(x)
                    if w1 < self.min_split or w2 < self.min_split:
                        return float('-inf')
                    elif (self.min_leaf is not None and
                        (sum(cur_x1_idx) < self.min_leaf or
                        sum(cur_x2_idx) < self.min_leaf)):
                        return float('-inf')

                    cur_x1 = x[cur_x1_idx, :]
                    cur_x2 = x[cur_x2_idx, :]

                    cur_y1 = y[cur_x1_idx, :]
                    cur_y2 = y[cur_x2_idx, :]

                    cur_blackbox_pred1 = blackbox_pred[cur_x1_idx]
                    cur_blackbox_pred2 = blackbox_pred[cur_x2_idx]

                    new_ent = w1 * self.loss.entropy(clf, cur_x1, cur_y1, cur_blackbox_pred1) \
                        + w2 * self.loss.entropy(clf, cur_x2, cur_y2, cur_blackbox_pred2)
                    drop = orig_ent - new_ent

                    return drop

                # Minimization of neg-entropy for best drop
                if SUPPRESS_MINIMIZE_WARNINGS:
                    with warnings.catch_warnings():
                        # Ignore double float warnings for minimize function
                        warnings.simplefilter('ignore')
                        cur_res = minimize_scalar(
                            lambda i: -delta_entropy(i),
                            bounds = (np.min(cur_col), np.max(cur_col)),
                            method = 'bounded')
                else:
                    cur_res = minimize_scalar(
                        lambda i: -delta_entropy(i),
                        bounds = (np.min(cur_col), np.max(cur_col)),
                        method = 'bounded')

                cur_thresh = cur_res.x
                cur_val = cur_res.fun

            return cur_val, cur_thresh, cur_subset, cur_isthresh

        # Update + Parallelize
        if self.parallel and len(self.protected_info.unprotected_columns) > 1:
            pool = ProcessPool()
            res = zip(self.protected_info.unprotected_columns,
                      pool.map(col_worker,
                               self.protected_info.unprotected_columns))
        else:
            res = zip(self.protected_info.unprotected_columns,
                      map(col_worker,
                          self.protected_info.unprotected_columns))

        # Get the smallest negative drop (highest drop)
        best = min(res, key=lambda x: x[1][0])
        best_col, (best_val, best_thresh, best_subset, best_isthresh) = best

        if best_val == float('inf'):
            return None

        if best_isthresh:
            @bool_rule_wrapper
            def best_rule(x: np.ndarray) -> bool:
                return x[best_col] > best_thresh
        else:
            @bool_rule_wrapper
            def best_rule(x: np.ndarray) -> bool:
                return np.isin(x[best_col], best_subset)

        return RuleSelection[np.ndarray](best_rule, -best_val, split_index=best_col)