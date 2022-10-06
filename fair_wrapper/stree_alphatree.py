import numpy as np

from sklearn.tree import DecisionTreeClassifier

from .types import ProtectedInfo, ClippedClassifier, Wrapper
from .loss import DTLossFunction
from .tree import DecisionTree
from .nodefunction import NodeFunction
from fair_wrapper.alphatree import AlphaTreeTwister

class STreeAlphaTreeTwister(AlphaTreeTwister):
    alpha_tree: DecisionTree[np.ndarray, np.ndarray]
    loss: DTLossFunction[np.ndarray]
    protected_info: ProtectedInfo
    hypothsis: NodeFunction[np.ndarray]

    def __init__(self, loss: DTLossFunction[np.ndarray],
                 protected_info: ProtectedInfo,
                 hypothesis: NodeFunction[np.ndarray],
                 root_tree_max_depth: int = 8) -> None:
        """
        Alpha-tree for CVaR with initial proxy tree.
        """
        self.loss = loss
        self.protected_info = protected_info
        self.hypothsis = hypothesis

        self.root_tree = DecisionTreeClassifier(max_depth=root_tree_max_depth)
        self.root_tree_is_trained = False

        # Setup the alpha tree
        self.alpha_tree = DecisionTree(1)

    def generate_rooted_x(self, x: np.ndarray) -> np.ndarray:
        if self.root_tree_is_trained:
            root_input = x[:, self.protected_info.unprotected_columns]
            rooted_protected = self.root_tree.predict(root_input)
            if len(rooted_protected.shape) == 1:
                rooted_protected = rooted_protected.reshape(-1, 1)
            rooted_x = x.copy()
            rooted_x[:, self.protected_info.protected_columns] = rooted_protected
            return rooted_x
        else:
            return x

    def root_split(self, x: np.ndarray, y: np.ndarray, clf: ClippedClassifier[np.ndarray]) -> None:
        root_input = x[:, self.protected_info.unprotected_columns]
        root_output = x[:, self.protected_info.protected_columns]

        self.root_tree.fit(root_input, root_output)
        self.root_tree_is_trained = True
        rooted_x = self.generate_rooted_x(x)
        
        return super().root_split(rooted_x, y, clf)

    def twist(self, x: np.ndarray) -> np.ndarray:
        rooted_x = self.generate_rooted_x(x)
        return np.apply_along_axis( self.alpha_tree, 1, rooted_x ).reshape(-1, 1)

    def grow_alphatree(self, x: np.ndarray, y: np.ndarray, clf: ClippedClassifier[np.ndarray], twisted_clf) -> None:
        rooted_x = self.generate_rooted_x(x)
        return super().grow_alphatree(rooted_x, y, clf, twisted_clf)

class STreeAlphaTreeWrapper(Wrapper[np.ndarray, np.ndarray]):
    blackbox: ClippedClassifier[np.ndarray]
    twister: STreeAlphaTreeTwister
    protected_info: ProtectedInfo

    def __init__(self, blackbox: ClippedClassifier[np.ndarray],
                 loss: DTLossFunction[np.ndarray],
                 protected_info: ProtectedInfo,
                 hypothesis: NodeFunction[np.ndarray]) -> None:
        """
        Wrapper function for CVaR alpha-tree with initial proxy tree.
        """
        super().__init__(
            blackbox, STreeAlphaTreeTwister(loss, protected_info, hypothesis))
        self.protected_info = protected_info

    def init(self, x: np.ndarray, y: np.ndarray) -> None:
        self.twister.root_split(x, y, self.blackbox)

    def step(self, x: np.ndarray, y: np.ndarray) -> None:
        self.twister.grow_alphatree(x, y, self.blackbox,
                                    lambda i: self.predict(i, i))

    def predict(self, xb: np.ndarray, xt: np.ndarray) -> np.ndarray:
        rooted_xb = self.twister.generate_rooted_x(xb)
        return super().predict(rooted_xb, xt)