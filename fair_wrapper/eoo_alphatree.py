from typing import Callable
import numpy as np
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KernelDensity

from .types import ProtectedInfo, ClippedClassifier, Wrapper
from .loss import DTLossFunction
from .alphatree import AlphaTreeTwister, NoChange, grow_tree
from .nodefunction import NodeFunction

class EOOAlphaTreeTwister(AlphaTreeTwister):
    def __init__(self, loss: DTLossFunction[np.ndarray],
                 protected_info: ProtectedInfo,
                 hypothesis: NodeFunction[np.ndarray], epsilon: float,
                 K: float, estimator_func: Callable[[np.ndarray], np.ndarray],
                 use_threshold: bool = True) -> None:
        """
        Alpha-tree for EOO
        Requires an estimation of Bayes posterior
        (either via naive bayes or empirical est.)
        """
        super().__init__(loss, protected_info, hypothesis)
        self.epsilon = epsilon
        self.K = K
        self.estimator_func = estimator_func
        self.density_estimator, self.posterior_estimator = None, None
        self.use_threshold = use_threshold

    def train_estimators(self, x: np.ndarray, y: np.ndarray):
        self.density_estimator, self.posterior_estimator = self.estimator_func(x, y)

    def grow_alphatree(self, x: np.ndarray, y: np.ndarray,
                       clf: ClippedClassifier[np.ndarray],
                       twisted_clf: Callable[[np.ndarray], np.ndarray]) -> None:
        signed_mask = np.all(y == np.array([1]), axis=1).reshape(-1)

        signed_x = x[signed_mask, :]
        signed_y = y[signed_mask, :]
        signed_twisted_pred = twisted_clf(signed_x) #> 0.5

        # Find extreme posteriors
        sg_meanpred_pairs = []
        for sg in self.protected_info.protected_domain:
            sg_idx = np.all(signed_x[:, self.protected_info.protected_columns] == sg, axis=1)
            sg_idx = sg_idx.reshape(-1)

            cur_pos_twisted_pred = signed_twisted_pred[sg_idx, :]

            cur_meanpred = np.mean(cur_pos_twisted_pred)
            sg_meanpred_pairs.append((sg, cur_meanpred))

        sorted_meanpred = list(sorted(sg_meanpred_pairs, key=lambda x: x[1]))
        _, largest_meanpred_value = sorted_meanpred[-1]
        other_meanpred = sorted_meanpred[:-1]

        # Get "pushup" constants
        delta = self.K * self.epsilon / (self.K - 1)
        p_min = largest_meanpred_value + self.epsilon / (self.K - 1)

        # Order support from highest posterior
        sorted_support = sorted(self.posterior_estimator.items(), key=lambda x: x[1], reverse=True)

        # Generate X_{p}
        push_value = 0.5 + delta
        total_support_prob = 0
        using_pushup = True
        in_pushup_interval = []
        for i in range(len(sorted_support)):
            cur_x, pr = sorted_support[i]

            cur_prob = self.density_estimator[cur_x]
            total_support_prob += cur_prob

            # If domain is in condition of push up
            if cur_prob < push_value:
                in_pushup_interval.append(np.array(cur_x))

            # If X_{p} is large enough
            if total_support_prob >= p_min:
                break
        in_pushup_interval = np.array(in_pushup_interval)

        if pr >= 0.5 or len(in_pushup_interval) == 0:
            using_pushup = False

        for sg, _ in other_meanpred:
            sg_idx = np.all(signed_x[:, self.protected_info.protected_columns] == sg, axis=1)
            sg_idx = sg_idx.reshape(-1)
 
            cur_x = signed_x[sg_idx, :]
            cur_y = signed_y[sg_idx, :]

            if using_pushup:
                # Mask to get all rows in cur_x with elements in x_p_support
                pushup_mask = []
                for i in range(cur_x.shape[0]):
                    pushup_mask.append(np.equal(cur_x[i, :], in_pushup_interval).all(axis=1).any())
                in_pushup_mask = np.array(pushup_mask)
                to_change = cur_y[in_pushup_mask, :]
                cur_y[in_pushup_mask, :] = np.full(shape=to_change.shape, fill_value=push_value)

            # Grow tree with subgroup samples
            new_node = grow_tree(self.alpha_tree, self.hypothsis, cur_x, cur_y, clf)
            if new_node is not None:
                return None

        raise NoChange

class EOOAlphaTreeWrapper(Wrapper[np.ndarray, np.ndarray]):
    blackbox: ClippedClassifier[np.ndarray]
    twister: AlphaTreeTwister
    protected_info: ProtectedInfo

    def __init__(self, blackbox: ClippedClassifier[np.ndarray],
                 loss: DTLossFunction[np.ndarray],
                 protected_info: ProtectedInfo,
                 hypothesis: NodeFunction[np.ndarray], epsilon: float,
                 K: float, estimator_func: Callable[[np.ndarray], np.ndarray],
                 use_threshold: bool) -> None:

        """
        Wrapper function for EOO alpha-tree.
        """
        super().__init__(
            blackbox, EOOAlphaTreeTwister(loss, protected_info, hypothesis,
            epsilon, K, estimator_func, use_threshold))
        self.protected_info = protected_info
        self.epsilon = epsilon
        self.K = K

    def init(self, x: np.ndarray, y: np.ndarray) -> None:
        self.twister.root_split(x, y, self.blackbox)
        self.twister.train_estimators(x, y)

    def step(self, x: np.ndarray, y: np.ndarray) -> None:
        self.twister.grow_alphatree(x, y, self.blackbox,
                                    lambda i: self.predict(i, i))

def make_empirical_posterior(x: np.ndarray, y: np.ndarray):
    total_count = defaultdict(int)
    pos_count = defaultdict(int)
    for i in range(x.shape[0]):
        x_tuple = tuple(x[i, :])
        total_count[x_tuple] += 1
        pos_count[x_tuple] += float(y[i])

    return {k: v / len(x) for (k, v) in total_count.items()}, {k: v / total_count[k] for (k, v) in pos_count.items()}

def make_estimators(x: np.ndarray, y: np.ndarray):
    density_est_model = KernelDensity().fit(x)
    posterior_est_model = GaussianNB().fit(x, y.ravel())

    density_est_prob = np.exp(density_est_model.score_samples(x))
    posterior_est_prob = posterior_est_model.predict_proba(x)[:, 1]

    density_dict = {}
    posterior_dict = {}
    for i in range(x.shape[0]):
        x_tuple = tuple(x[i, :])
        density_dict[x_tuple] = density_est_prob[i]
        posterior_dict[x_tuple] = posterior_est_prob[i]

    return density_dict, posterior_dict

def make_empirical_EOOAlphaTreeWrapper(blackbox: ClippedClassifier[np.ndarray],
                                       loss: DTLossFunction[np.ndarray],
                                       protected_info: ProtectedInfo,
                                       hypothesis: NodeFunction[np.ndarray],
                                       epsilon: float, K: float, use_threshold: bool):
    return EOOAlphaTreeWrapper(blackbox, loss, protected_info, hypothesis, epsilon, K,
                               make_empirical_posterior, use_threshold)

def make_sklearn_EOOAlphaTreeWrapper(blackbox: ClippedClassifier[np.ndarray],
                                     loss: DTLossFunction[np.ndarray],
                                     protected_info: ProtectedInfo,
                                     hypothesis: NodeFunction[np.ndarray],
                                     epsilon: float, K: float, use_threshold: bool):
    return EOOAlphaTreeWrapper(blackbox, loss, protected_info, hypothesis, epsilon, K,
                               make_estimators, use_threshold)
