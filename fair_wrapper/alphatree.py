import numpy as np

from typing import Optional
from collections import defaultdict

from .types import Twister, ProtectedInfo, ClippedClassifier, Wrapper
from .loss import DTLossFunction
from .tree import DecisionTree, DecisionNode
from .nodefunction import NoMoreProtectedSplits, NodeFunction, RootProtectedSplit

class NoChange(Exception):
    pass

class AlphaTreeTwister(Twister[np.ndarray]):
    alpha_tree: DecisionTree[np.ndarray, np.ndarray]
    loss: DTLossFunction[np.ndarray]
    protected_info: ProtectedInfo
    hypothsis: NodeFunction[np.ndarray]

    def __init__(self, loss: DTLossFunction[np.ndarray],
                 protected_info: ProtectedInfo,
                 hypothesis: NodeFunction[np.ndarray]) -> None:
        """
        Alpha-tree for CVaR
        """
        self.loss = loss
        self.protected_info = protected_info
        self.hypothsis = hypothesis

        self.alpha_tree = DecisionTree(1) # init

    def root_split(self, x: np.ndarray, y: np.ndarray,
                   clf: ClippedClassifier[np.ndarray]) -> None:
        """
        Creates initial sensitive attribute split
        """
        root_node_splitter = RootProtectedSplit(self.loss, self.protected_info)
        blackbox_pred = clf(x)
        cur_node = self.alpha_tree.root
        cur_x = x
        cur_y = y
        cur_blackbox_pred = blackbox_pred
        while True:
            try:
                grow_tree(cur_node, root_node_splitter, cur_x, cur_y, clf,
                          blackbox_pred=cur_blackbox_pred, skip_ent_check=True)
            except NoMoreProtectedSplits:
                break
            
            # Update node to split
            sample_dict = tree_sample_dict(self.alpha_tree.root, x)
            cur_node = cur_node.right
            cur_index = sample_dict[cur_node]
            cur_x = x[cur_index, :]
            cur_y = y[cur_index, :]
            cur_blackbox_pred = blackbox_pred[cur_index, :]

    def twist(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate twist function alpha (pointwise evaluated).
        """
        return np.apply_along_axis( self.alpha_tree, 1, x ).reshape(-1, 1)

    def grow_alphatree(self, x: np.ndarray, y: np.ndarray,
                       clf: ClippedClassifier[np.ndarray], twisted_clf) -> None:
        # Find worse subgroup
        sg_loss_pairs = []
        for sg in self.protected_info.protected_domain:
            sg_idx = np.all(x[:, self.protected_info.protected_columns] == sg, axis=1)
            sg_idx = sg_idx.reshape(-1)
            
            cur_x = x[sg_idx, :]
            cur_y = y[sg_idx, :]

            cur_loss = self.loss.loss(twisted_clf, cur_x, cur_y)
            sg_loss_pairs.append((sg, cur_loss))

        for sg, _ in sorted(sg_loss_pairs, key=lambda x: x[1], reverse=True):
            sg_idx = np.all(x[:, self.protected_info.protected_columns] == sg, axis=1)
            sg_idx = sg_idx.reshape(-1)
            
            cur_x = x[sg_idx, :]
            cur_y = y[sg_idx, :]

            # Grow tree with subgroup samples
            new_node = grow_tree(self.alpha_tree, self.hypothsis, cur_x, cur_y, clf)
            if new_node is not None:
                return None

        raise NoChange

class AlphaTreeWrapper(Wrapper[np.ndarray, np.ndarray]):
    blackbox: ClippedClassifier[np.ndarray]
    twister: AlphaTreeTwister
    protected_info: ProtectedInfo

    def __init__(self, blackbox: ClippedClassifier[np.ndarray],
                 loss: DTLossFunction[np.ndarray],
                 protected_info: ProtectedInfo,
                 hypothesis: NodeFunction[np.ndarray]) -> None:
        """
        Wrapper function for CVaR alpha-tree.
        """
        super().__init__(
            blackbox, AlphaTreeTwister(loss, protected_info, hypothesis))
        self.protected_info = protected_info

    def init(self, x: np.ndarray, y: np.ndarray) -> None:
        self.twister.root_split(x, y, self.blackbox)

    def step(self, x: np.ndarray, y: np.ndarray) -> None:
        self.twister.grow_alphatree(x, y, self.blackbox,
                                    lambda i: self.predict(i, i))

def tree_sample_dict(dt: DecisionNode, x: np.ndarray):
    """
    Calculate a dictionary from nodes to inputs indices
    """
    # Find where each sample gets sent
    sample_dict = defaultdict( list )
    for i in range( x.shape[0] ):
        cur_dn = dt.find_leaf( x[i, :] )
        sample_dict[cur_dn].append(i)

    return sample_dict

def grow_tree(dt: DecisionNode, hyp: NodeFunction[np.ndarray], x: np.ndarray,
               y: np.ndarray, clf: ClippedClassifier[np.ndarray],
               blackbox_pred: np.ndarray = None,
               skip_ent_check: bool = False) -> Optional[DecisionTree]:
    """
    Grow function for alpha-tree (determining splits etc.)
    """
    sample_dict = tree_sample_dict(dt, x)

    fattest_dn, fattest_indices = max(sample_dict.items(), key=lambda x: len(x[1]))
    cur_x = x[fattest_indices, :]
    cur_y = y[fattest_indices, :]

    if blackbox_pred is None:
        cur_blackbox_pred = clf(cur_x)
    else:
        cur_blackbox_pred = blackbox_pred[fattest_indices, :]

    if fattest_dn.value == 1 and skip_ent_check is False:
        fattest_dn.value = hyp.loss.tree_value(clf, cur_x, cur_y, cur_blackbox_pred)
        return fattest_dn
    else:
        cur_rule = hyp.rule_selection(clf, cur_x, cur_y, cur_blackbox_pred)
        if cur_rule is not None and (cur_rule.local_ent_drop > 0 or skip_ent_check):
            left_alpha, right_alpha = hyp.split_tree_values(cur_rule, clf, cur_x, cur_y, cur_blackbox_pred)
            print('local entropy drop', cur_rule.local_ent_drop)

            fattest_dn.split_node(cur_rule, left_alpha, right_alpha)
            return fattest_dn

    return None
