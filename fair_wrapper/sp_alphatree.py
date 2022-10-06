import numpy as np

from .types import ProtectedInfo, ClippedClassifier, Wrapper
from .loss import DTLossFunction
from .alphatree import AlphaTreeTwister, NoChange, grow_tree
from .nodefunction import NodeFunction

class SPAlphaTreeTwister(AlphaTreeTwister):
    def __init__(self, loss: DTLossFunction[np.ndarray],
                 protected_info: ProtectedInfo,
                 hypothesis: NodeFunction[np.ndarray], reverse=False) -> None:
        """
        Alpha-tree for SP
        """
        super().__init__(loss, protected_info, hypothesis)
        self.reverse = reverse

    def grow_alphatree(self, x: np.ndarray, y: np.ndarray,
                       clf: ClippedClassifier[np.ndarray], twisted_clf) -> None:
        # Find extreme posteriors
        sg_posterior_pairs = []
        for sg in self.protected_info.protected_domain:
            sg_idx = np.all(x[:, self.protected_info.protected_columns] == sg, axis=1)
            sg_idx = sg_idx.reshape(-1)
            
            cur_x = x[sg_idx, :]

            cur_posterior = np.mean(twisted_clf(cur_x))
            sg_posterior_pairs.append((sg, cur_posterior))

        sorted_posteriors = list(sorted(sg_posterior_pairs, key=lambda x: x[1],
                                        reverse=self.reverse))
        largest_posterior_sg = sorted_posteriors[-1][0]
        other_posterior = sorted_posteriors[:-1]

        sg_idx = np.all(x[:, self.protected_info.protected_columns] == largest_posterior_sg, axis=1)
        sg_idx = sg_idx.reshape(-1)
        largest_posterior_x = x[sg_idx, :]
        expected_largest_posterior = np.mean(clf(largest_posterior_x))

        for sg, sg_mean_posterior in other_posterior:
            sg_idx = np.all(x[:, self.protected_info.protected_columns] == sg, axis=1)
            sg_idx = sg_idx.reshape(-1)
            
            cur_x = x[sg_idx, :]
            cur_y = y[sg_idx, :]

            # Change target posterior
            cur_y = np.full(shape=cur_y.shape, fill_value=expected_largest_posterior)

            # Grow tree with subgroup samples
            new_node = grow_tree(self.alpha_tree, self.hypothsis, cur_x, cur_y, clf)
            if new_node is not None:
                return None

        raise NoChange

class SPAlphaTreeWrapper(Wrapper[np.ndarray, np.ndarray]):
    blackbox: ClippedClassifier[np.ndarray]
    twister: AlphaTreeTwister
    protected_info: ProtectedInfo

    def __init__(self, blackbox: ClippedClassifier[np.ndarray],
                 loss: DTLossFunction[np.ndarray],
                 protected_info: ProtectedInfo,
                 hypothesis: NodeFunction[np.ndarray], reverse: bool) -> None:
        """
        Wrapper function for SP alpha-tree.
        """
        super().__init__(
            blackbox, SPAlphaTreeTwister(loss, protected_info, hypothesis, reverse))
        self.protected_info = protected_info

    def init(self, x: np.ndarray, y: np.ndarray) -> None:
        self.twister.root_split(x, y, self.blackbox)

    def step(self, x: np.ndarray, y: np.ndarray) -> None:
        self.twister.grow_alphatree(x, y, self.blackbox,
                                    lambda i: self.predict(i, i))