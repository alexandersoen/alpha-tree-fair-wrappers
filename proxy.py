#!/usr/bin/env python

import json
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

from fair_wrapper.datasets import acs_age_dataset, cross_validate_dataset, german_credit_age_dataset, bank_age_dataset, split_dataset
from fair_wrapper.fairness import SKDTClassifier, get_stats

from fair_wrapper.loss import DTCrossEntropy, DTCrossEntropyAggressive
from fair_wrapper.alphatree import NoChange
from fair_wrapper.stree_alphatree import STreeAlphaTreeWrapper
from fair_wrapper.nodefunction import UnprotectedProjectionEnumerate

def main() -> None:
    parser = argparse.ArgumentParser( description='Fair Wrappers Example' )

    # Constants
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('-B', '--clip_val', type=float, default=1.0, metavar='B',
                        help='clipping variable for blackbox classifiers (default: 1.0)')
    parser.add_argument('--beta', type=float, default=0.05, metavar='b',
                        help='cvar beta value (default: 0.05)')
    parser.add_argument('--dt-max-depth', type=int, default=8, metavar='D',
                        help='maximum depth for decision tree classifier (default: 8)')

    # Data selecting
    parser.add_argument('--dataset', type=str, default='german', metavar='D',
                        help='selected dataset ["german", "bank"] (default: "german")')
    parser.add_argument('--blackbox', type=str, default='rf', metavar='B',
                        help='selected blackbox ["rf", "dt", "mlp"] (default: "rf")')
    parser.add_argument('--calibrate', action='store_true', default=False,
                        help='Calibrate blackbox (default: False)')

    # Data splitting
    parser.add_argument('--bin-age', action='extend', nargs='+', type=float,
                        help='(sorted) binning values for age, between 0 and inf')
    parser.add_argument('--share-train-data', action='store_true', default=False,
                        help='enable blackbox data to be shared with twister (default: False)')
    parser.add_argument('--cv-folds', type=int, default=5, metavar='T',
                        help='number of CV folds (default: 5)')
    parser.add_argument('--alpha-split', type=float, default=0.5, metavar='T',
                        help='percentage of non-testing data used for training twister, if enabled (default: 0.5)')

    # Wrapper settings
    parser.add_argument('--num-splits', type=int, default=4, metavar='S',
                        help='Number of steps / splits the blackbox wrapper does (default: 4)')
    parser.add_argument('--aggressive-update', action='store_true', default=False,
                        help='use the aggressive alpha-tree weights (default: False)')
    parser.add_argument('--min-split', type=float, default=0.0, metavar='M',
                        help='minimum percentage of examples in leaf splits (default: 0.0)')
    parser.add_argument('--min-leaf', type=int, default=None, metavar='M',
                        help='minimum leaf size in leaf splits (default: None)')

    # Files
    parser.add_argument('--save-folder', type=str, default='.',
                        help='save folder (default: ".")')
    parser.add_argument('--save-file', type=str, default='wrapper_res.json',
                        help='save file (default: "wrapper_res.json")')
    parser.add_argument('--wrapper-folder', type=str, default='',
                        help='wrapper save folder (default: '' [does not save])')
    
    args = parser.parse_args()

    # Display settings #
    print(args)
    print( f'using random seed {args.seed}' )
    np.random.seed( args.seed )

    save_path_train = Path(args.save_folder, f'train_{args.save_file}')
    save_path_test = Path(args.save_folder, f'test_{args.save_file}')
    save_path_alpha = Path(args.save_folder, f'alpha_{args.save_file}')
    save_path_train.parent.mkdir(parents=True, exist_ok=True)
    print(f'saving results to {save_path_train.parent}')

    wrapper_path = None
    if len(args.wrapper_folder):
        wrapper_path = Path(args.wrapper_folder)
        wrapper_path.mkdir(parents=True, exist_ok=True)
        print(f'saving wrapped classifiers to {wrapper_path}')
    # Display settings #

    binning = args.bin_age if args.bin_age else [25]
    binning = [0] + binning + [float('inf')]

    # Get selected dataset
    if args.dataset == 'german':
        dataset = german_credit_age_dataset(binning=binning)
    elif args.dataset == 'bank':
        dataset = bank_age_dataset(binning=binning)
    elif args.dataset == 'acs':
        dataset = acs_age_dataset(binning=binning)
    elif args.dataset == 'acs18':
        dataset = acs_age_dataset(binning=binning, year='2018')
    else:
        raise ValueError('Unknown Dataset String: {}'.format(args.dataset))

    train_stats = []
    alpha_stats = []
    test_stats  = []
    cv_datasets = cross_validate_dataset(dataset, n_splits=args.cv_folds,
                                         random_state=args.seed)
    tqdm_datasets = tqdm(enumerate(cv_datasets), total=args.cv_folds,
                         desc='cross validation fold ', position=0)
    for cv_iter, (dataset_training, dataset_test) in tqdm_datasets:
        # Need to split up the dataset into 3 sets: (1) Training set; (2) Alpha Tree Training set; (3) Testing set
        if args.share_train_data:
            dataset_train = dataset_training
            dataset_alpha = dataset_training
        else:
            dataset_train, dataset_alpha = split_dataset(
                dataset_training, test_size=args.alpha_split, seed=args.seed)

        print('train', dataset_train.x.shape)
        print('alpha', dataset_alpha.x.shape)
        print('test', dataset_test.x.shape)

        p_info = dataset_alpha.protected_info

        # Train base classifier
        if args.blackbox == 'dt':
            model = DecisionTreeClassifier(max_depth=args.dt_max_depth)
        elif args.blackbox == 'random-forest':
            model = RandomForestClassifier(random_state=args.seed, max_depth=4, max_samples=1000, n_estimators=50, min_samples_leaf=30)
        elif args.blackbox == 'rf':
            model = RandomForestClassifier(random_state=args.seed, max_depth=4, max_samples=0.1, n_estimators=50)
        elif args.blackbox == 'naive-bayes':
            model = GaussianNB()
        elif args.blackbox == 'mlp':
            model = MLPClassifier(random_state=args.seed, max_iter=300, early_stopping=True, validation_fraction=0.2)

        if args.calibrate:
            dt = CalibratedClassifierCV(base_estimator=model, cv=5)
        else:
            dt = model

        dt.fit(dataset_train.x, dataset_train.y.ravel())
        if np.max(dt.predict_proba(dataset_train.x)) == 1:
            calculated_clip = np.inf
        else:
            calculated_clip = - np.log(1 / np.max(dt.predict_proba(dataset_train.x)) - 1)
        print('Adjusted Clip', min(args.clip_val, calculated_clip))

        # Wrap the blackbox
        blackbox_dt = SKDTClassifier(dt, min(args.clip_val, calculated_clip))

        # Settings for loss / wrapper
        if args.aggressive_update:
            loss = DTCrossEntropyAggressive()
        else:
            loss = DTCrossEntropy()

        hypothesis_class = UnprotectedProjectionEnumerate(
            loss, p_info, min_split=args.min_split, min_leaf=args.min_leaf,
            max_splits_considered=30)

        # Type of optimization
        wrapper = STreeAlphaTreeWrapper(blackbox_dt, loss, p_info, hypothesis_class)

        # Initialize the wrapper
        wrapper.init(dataset_alpha.x, dataset_alpha.y)
        if wrapper_path is not None:
            wrapper.save(wrapper_path.joinpath(f'{0}_wrapped_classifier.pkl'))

        cur_train_stats = [  ]
        cur_alpha_stats = [  ]
        cur_test_stats  = [  ]

        cur_train_stats.append(get_stats(wrapper, dataset_train, 0))
        if not args.share_train_data:
            cur_alpha_stats.append(get_stats(wrapper, dataset_alpha, 0))
        cur_test_stats.append(get_stats(wrapper, dataset_test, 0))

        # Boost iterate
        for i in tqdm(range(args.num_splits), desc=' >>> boost iter',
                      position=1, leave=False):
            try:
                wrapper.step(dataset_alpha.x, dataset_alpha.y)
            except NoChange:
                break

            if wrapper_path is not None:
                wrapper.save(wrapper_path.joinpath(f'{cv_iter}_{i}_wrapped_classifier.pkl'))

            # Calculate cur stats
            cur_train_stats.append( get_stats(wrapper, dataset_train, i+1) )
            if not args.share_train_data:
                cur_alpha_stats.append( get_stats(wrapper, dataset_test, i+1) )
            cur_test_stats.append( get_stats(wrapper, dataset_test, i+1) )

        # Add stats to fold results
        train_stats.append({'cv_iter': cv_iter, 'results': cur_train_stats})
        if args.share_train_data:
            alpha_stats = train_stats
        else:
            alpha_stats.append({'cv_iter': cv_iter, 'results': cur_alpha_stats})
        test_stats.append({'cv_iter': cv_iter, 'results': cur_test_stats})

        # Save
        with save_path_train.open(mode='w') as f:
            json.dump(train_stats, f, indent=4, separators=(',', ':'))
        with save_path_alpha.open(mode='w') as f:
            json.dump(alpha_stats, f, indent=4, separators=(',', ':'))
        with save_path_test.open(mode='w') as f:
            json.dump(test_stats, f, indent=4, separators=(',', ':'))

if __name__ == '__main__':
    main()
