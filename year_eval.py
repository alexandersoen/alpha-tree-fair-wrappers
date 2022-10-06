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
from collections import defaultdict

from fair_wrapper.datasets import FeaturePreprocessor, acs_age_dataset, cross_validate_dataset, preprocess, split_dataset
from fair_wrapper.fairness import SKDTClassifier, get_stats

from fair_wrapper.loss import DTCrossEntropy, DTCrossEntropyAggressive
from fair_wrapper.alphatree import AlphaTreeWrapper, NoChange
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
    parser.add_argument('--year',  default=2015, type=int, metavar='Y',
                        help='year of training (default: 2015)')
    parser.add_argument('--years',  action='extend', nargs='+', type=int,
                        help='years in order of wrapper generation (default: [2015, 2016, 2017, 2018])')
    parser.add_argument('--cv-folds', type=int, default=5, metavar='T',
                        help='number of CV folds (default: 5)')
    parser.add_argument('--blackbox', type=str, default='rf', metavar='B',
                        help='selected blackbox ["rf", "dt", "mlp"] (default: "rf")')
    parser.add_argument('--calibrate', action='store_true', default=False,
                        help='Calibrate blackbox (default: False)')

    # Data splitting
    parser.add_argument('--bin-age', action='extend', nargs='+', type=float,
                        help='(sorted) binning values for age, between 0 and inf')
    parser.add_argument('--test-split', type=float, default=0.2, metavar='T',
                        help='percentage of original data used for testing (default: 0.2)')
    parser.add_argument('--share-train-data', action='store_true', default=False,
                        help='enable blackbox data to be shared with twister (default: False)')
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

    wrapper_path = None
    if len(args.wrapper_folder):
        wrapper_path = Path(args.wrapper_folder)
        wrapper_path.mkdir(parents=True, exist_ok=True)
        print(f'saving wrapped classifiers to {wrapper_path}')
    # Display settings #

    binning = args.bin_age if args.bin_age else [25]
    binning = [0] + binning + [float('inf')]

    # Get selected dataset
    years = args.years if args.years else [2015, 2016, 2017, 2018]

    i = years.index(args.year)

    # Set up save paths
    save_path_train = Path(args.save_folder, f'train_{years[i]}_{args.save_file}')
    save_path_alpha = Path(args.save_folder, f'alpha_{years[i]}_{args.save_file}')
    save_path_test = Path(args.save_folder, f'test_{years[i]}_{args.save_file}')
    save_path_train.parent.mkdir(parents=True, exist_ok=True)
    print(f'saving results to {save_path_train.parent}')

    cv_datasets = []
    for y in years:
        dataset = acs_age_dataset(binning=binning, year=str(y))
        p_info = dataset.protected_info
        preprocessor = FeaturePreprocessor(p_info, dataset.feature_names)
        preprocessor.fit(dataset.x)
        dataset = preprocess(dataset, preprocessor)
        p_info = dataset.protected_info

        cv_dataset = cross_validate_dataset(
            dataset, n_splits=args.cv_folds, random_state=args.seed)

        cv_datasets.append(cv_dataset)

    train_stats = []
    alpha_stats = []
    test_stats = []
    for cv_iter in tqdm(range(args.cv_folds), desc='cross validation ', total=args.cv_folds, position=1):
        train_datasets = []
        alpha_datasets = []
        test_datasets = []
        for cv_dataset in cv_datasets:
            training_dataset, test_dataset = next(cv_dataset)

            if args.share_train_data:
                train_dataset = training_dataset
                alpha_dataset = training_dataset
            else:
                train_dataset, alpha_dataset = split_dataset(
                    training_dataset, test_size=args.alpha_split, seed=args.seed)

            train_datasets.append(train_dataset)
            alpha_datasets.append(alpha_dataset)
            test_datasets.append(test_dataset)

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
        dt.fit(train_datasets[i].x, train_datasets[i].y.ravel())
        if np.max(dt.predict_proba(train_datasets[i].x)) == 1:
            calculated_clip = np.inf
        else:
            calculated_clip = - np.log(1 / np.max(dt.predict_proba(train_datasets[i].x)) - 1)
        print('Adjusted Clip', min(args.clip_val, calculated_clip))

        # Wrap the blackbox
        blackbox_dt = SKDTClassifier(dt, min(args.clip_val, calculated_clip))

        # Settings for loss / wrapper
        if args.aggressive_update:
            loss = DTCrossEntropyAggressive()
        else:
            loss = DTCrossEntropy()

        cur_blackbox = blackbox_dt

        print('\n ######')

        print(f'Training wrapper for {years[i]}')

        cur_p_info = dataset.protected_info
        cur_train_stats = {
            'year': years[i],
            'stats': defaultdict(list),
        }
        cur_alpha_stats = {
            'year': years[i],
            'stats': defaultdict(list),
        }
        cur_test_stats = {
            'year': years[i],
            'stats': defaultdict(list),
        }

        hypothesis_class = UnprotectedProjectionEnumerate(
            loss, cur_p_info, min_split=args.min_split, min_leaf=args.min_leaf,
            max_splits_considered=30)

        wrapper = AlphaTreeWrapper(cur_blackbox, loss, cur_p_info,
                                hypothesis_class)

        # Stats for init step (only clipping)
        wrapper.init(train_datasets[i].x, train_datasets[i].y)
        for y, d_train, d_alpha, d_test in zip(years, train_datasets, alpha_datasets, test_datasets):
            cur_train_stats['stats'][y].append(get_stats(wrapper, d_train, 0))
            cur_alpha_stats['stats'][y].append(get_stats(wrapper, d_alpha, 0))
            cur_test_stats['stats'][y].append(get_stats(wrapper, d_test, 0))

        # Boosting iterations
        for j in tqdm(range(args.num_splits), desc=' >>> boost iter', position=2, leave=False):
            try:
                wrapper.step(alpha_datasets[i].x, alpha_datasets[i].y)
            except NoChange:
                break

            if wrapper_path is not None:
                wrapper.save(wrapper_path.joinpath(f'{years[i]}_{j}_wrapped_classifier.pkl'))

            # Stats
            for y, d_train, d_alpha, d_test in zip(years, train_datasets, alpha_datasets, test_datasets):
                cur_train_stats['stats'][y].append(get_stats(wrapper, d_train, j+1))
                cur_alpha_stats['stats'][y].append(get_stats(wrapper, d_alpha, j+1))
                cur_test_stats['stats'][y].append(get_stats(wrapper, d_test, j+1))

        # Add stats to fold results
        train_stats.append({'cv_iter': cv_iter, 'results': cur_train_stats})
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
