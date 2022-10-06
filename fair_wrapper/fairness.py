import numpy as np

from pathos.multiprocessing import ProcessPool
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

from . import loss as clf_loss
from .types import Wrapper, ProtectedInfo, ClippedClassifier
from .alphatree import AlphaTreeWrapper, tree_sample_dict

### Classifiers ###
class WrappedClassifier(ClippedClassifier[np.ndarray]):
    def __init__(self, wrapper: Wrapper[np.ndarray, np.ndarray], B: float) -> None:
        super().__init__(B)
        self.wrapper = wrapper

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self.wrapper.predict(x, x)

class SKPlainClassifier(ClippedClassifier[np.ndarray]):
    def __init__(self, dt, B: float) -> None:
        super().__init__(B)
        self.dt = dt

    # Override Clipping
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.dt.predict_proba(x)[:, [1]]

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self.dt.predict_proba(x)[:, [1]]

class SKDTClassifier(ClippedClassifier[np.ndarray]):
    def __init__(self, dt, B: float) -> None:
        super().__init__(B)
        self.dt = dt

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self.dt.predict_proba(x)[:, [1]]
### Classifiers ###


### Evaluation ###
def sg_weighted_loss(x: np.ndarray, y: np.ndarray, posterior_pred: np.ndarray, protected_info: ProtectedInfo):
    sg_weight_losses = []
    for sg in protected_info.protected_domain:
        sg_idx = np.all(x[:, protected_info.protected_columns] == sg, axis=1)
        sg_idx = sg_idx.reshape(-1)

        cur_y = y[sg_idx, :]

        cur_posterior = posterior_pred[sg_idx, :]

        cur_loss = clf_loss.cross_entropy(cur_y, cur_posterior)
        cur_weight = sum(sg_idx) / len(y)

        sg_weight_losses.append((cur_weight, cur_loss))

    return sg_weight_losses

def split_count(wrapper: AlphaTreeWrapper, x: np.ndarray):
    tree_sample = tree_sample_dict(wrapper.twister.alpha_tree, x)
    return [len(v) for v in tree_sample.values()]

def cvar(x: np.ndarray, y: np.ndarray, posterior_pred: np.ndarray, protected_info: ProtectedInfo, beta: float = 0.05):
    sg_weight_losses = sg_weighted_loss(x, y, posterior_pred, protected_info)

    # if infinite loss occurs
    for w, l in sg_weight_losses:
        if w > 0 and l == float('inf'):
            return float('inf')

    def cvar_inst(rho):
        return rho + 1/(1-beta) * sum( w * max(0, l - rho) for w, l in sg_weight_losses )

    res = minimize_scalar(cvar_inst, bounds=(0, np.max( [l for _, l in sg_weight_losses] )), method='bounded')
    return res.fun

def ce(x: np.ndarray, y: np.ndarray, posterior_pred: np.ndarray):
    cur_ce = clf_loss.cross_entropy(y, posterior_pred)
    return cur_ce

def twister_alphas(wrapper: AlphaTreeWrapper, x: np.ndarray, y: np.ndarray):
    leaves = wrapper.twister.alpha_tree.leaves
    return [l.value for l in leaves]

def equal_opportunity(x: np.ndarray, y: np.ndarray, posterior_pred: np.ndarray, protected_info: ProtectedInfo, threshold: float = 0.5):
    pos_blackbox_pred = posterior_pred > threshold
    signed_mask = np.all(y == np.array([1]), axis=1).reshape(-1)
    signed_x = x[signed_mask, :]
    signed_blackbox_pred = pos_blackbox_pred[signed_mask, :]

    # Find extreme posteriors
    sg_meanpred_vals = []
    for sg in protected_info.protected_domain:
        sg_idx = np.all(signed_x[:, protected_info.protected_columns] == sg, axis=1)
        sg_idx = sg_idx.reshape(-1)

        cur_pos_blackbox_pred = signed_blackbox_pred[sg_idx, :]

        cur_meanpred = np.mean(cur_pos_blackbox_pred)
        sg_meanpred_vals.append(cur_meanpred)

    eo = np.max(sg_meanpred_vals) - np.min(sg_meanpred_vals)
    return eo

def posterior_equal_opportunity(x: np.ndarray, y: np.ndarray, posterior_pred: np.ndarray, protected_info: ProtectedInfo):
    pos_blackbox_pred = posterior_pred
    signed_mask = np.all(y == np.array([1]), axis=1).reshape(-1)
    signed_x = x[signed_mask, :]
    signed_blackbox_pred = pos_blackbox_pred[signed_mask, :]

    # Find extreme posteriors
    sg_meanpred_vals = []
    for sg in protected_info.protected_domain:
        sg_idx = np.all(signed_x[:, protected_info.protected_columns] == sg, axis=1)
        sg_idx = sg_idx.reshape(-1)

        cur_pos_blackbox_pred = signed_blackbox_pred[sg_idx, :]

        cur_meanpred = np.mean(cur_pos_blackbox_pred)
        sg_meanpred_vals.append(cur_meanpred)

    eo = np.max(sg_meanpred_vals) - np.min(sg_meanpred_vals)
    return eo

def auc(x: np.ndarray, y: np.ndarray, posterior_pred: np.ndarray) -> float:
    auc_val = roc_auc_score(y, posterior_pred)
    return auc_val

def accuracy(x: np.ndarray, y: np.ndarray, posterior_pred: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = posterior_pred > threshold
    return np.average( y_pred == y )

def posterior_accuracy(x: np.ndarray, y: np.ndarray, posterior_pred: np.ndarray):
    return np.mean(np.abs(y - posterior_pred))

def statistical_parity(x: np.ndarray, y: np.ndarray, posterior_pred: np.ndarray, protected_info: ProtectedInfo) -> float:
    posterior_sg_vals = []
    for sg in protected_info.protected_domain:
        sg_idx = np.all(x[:, protected_info.protected_columns] == sg, axis=1)
        sg_idx = sg_idx.reshape(-1)
       
        cur_posterior = posterior_pred[sg_idx, :]

        posterior_sg_vals.append(np.mean(cur_posterior))

    return np.max(posterior_sg_vals) - np.min(posterior_sg_vals)

# Stat functions
def get_stats(wrapper, dataset, i):
    BETAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    THRESHOLDS = np.arange(0.3, 0.71, 0.01)

    x = dataset.x
    y = dataset.y
    posterior_pred = wrapper.predict(x, x)
    p_info = dataset.protected_info
    alphas = twister_alphas(wrapper, x, y)

    eval_workers = (
            [ ('cvar', b, cvar, (x, y, posterior_pred, p_info), {'beta': b}) for b in BETAS ]
            +
            [ ('acc', t, accuracy, (x, y, posterior_pred), {'threshold': t}) for t in THRESHOLDS ]
            +
            [ ('equal_opportunity', t, equal_opportunity, (x, y, posterior_pred, p_info), {'threshold': t}) for t in THRESHOLDS ]
            +
            [
              ('ce', None, ce, (x, y, posterior_pred), {}),
              ('statistical_parity', None, statistical_parity, (x, y, posterior_pred, p_info), {}),
              ('posterior_acc', None, posterior_accuracy, (x, y, posterior_pred), {}),
              ('posterior_equal_opportunity', None, posterior_equal_opportunity, (x, y, posterior_pred, p_info), {}),
              ('sg_weighted', None, sg_weighted_loss, (x, y, posterior_pred, p_info), {}),
              ('auc', None, auc, (x, y, posterior_pred), {}),
              ('data_split', None, split_count, (wrapper, x), {}),
            ])

    def work(vals):
        name, hyper_name, func, args, kwargs = vals

        work_res = func(*args, **kwargs)

        return (name, hyper_name, work_res)

    pool = ProcessPool()
    res = pool.map(work, eval_workers)

    stat_res = {
        'step': i,
        'alphas': alphas,
    }
    for name, hyper_name, output in res:
        if hyper_name is None:
            stat_res[name] = output
        else:
            if name in stat_res:
                stat_res[name].update({hyper_name: output})
            else:
                stat_res[name] = {hyper_name: output}

    return stat_res
### Evaluation ###