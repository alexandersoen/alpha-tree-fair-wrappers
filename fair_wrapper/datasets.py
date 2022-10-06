import numpy as np
import pandas as pd

import folktables

from aif360.datasets import GermanDataset, BankDataset
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold

from typing import Iterator, List, Tuple
from dataclasses import dataclass
from .types import ProtectedInfo

@dataclass
class Dataset:
    """
    Generic dataclass for data.
    """
    x: np.ndarray
    y: np.ndarray
    protected_info: ProtectedInfo
    feature_names: List[str]

    def __len__(self):
        return self.x.shape[0]

    def export(self):
        data = np.concatenate((self.x, self.y), axis=1)
        label_names = ['label'] if self.y.shape[1] == 1 \
            else [f'label_{i}' for i in range(self.y.shape[1])]
        
        column_names = list(self.feature_names) + label_names
        label_idx = [column_names.index(l) for l in label_names]

        info_dict = {
            'column_names': column_names,
            'label_idx': label_idx,
            'nonsensitive_feature_idx': self.protected_info.unprotected_columns,
            'sensitive_feature_idx': self.protected_info.protected_columns,
        }
        return data, info_dict

DEFAULT_AGE_BINNING = [0, 25, float('inf')]

def generate_protected_domain(cols):
    # Get unique values
    protected_values = np.unique(
        cols, axis=0)

    # Turn the values (which could be binned values) into strings
    protected_str = np.apply_along_axis(
        lambda x: ''.join(map(lambda i: str(int(i)), x)),
        1, protected_values)

    # Sort for revese binary order (one hots)
    protected_domain = list(protected_values[protected_str.argsort()[::-1]])

    return protected_domain

def aif360_dataset_age_pinfo(df, info, enumerate_columns) -> ProtectedInfo:
    x_features = [ c for c in df.columns if c not in info['label_names'] ]
    protected_columns = [ i for i in range(len(x_features)) if 'age_' in x_features[i] or 'age' == x_features[i] ]
    unprotected_columns = [ i for i in range(len(x_features)) if i not in protected_columns ]
    
    # Generate protected domain (with ordering)
    protected_domain = generate_protected_domain(df.to_numpy()[:, protected_columns])

    return ProtectedInfo(protected_columns, unprotected_columns,
                         protected_domain, enumerate_columns)

def aif360_multisensitive_preprocessing(df, binning=DEFAULT_AGE_BINNING, sens_name='age'):
    # Encoding sensitive attribute binning
    if len(binning) > 3:
        age_binning = pd.cut(df[sens_name], binning, labels=range(len(binning)-1))
        age_hots = np.array(LabelBinarizer().fit_transform(age_binning).tolist())
        del df[sens_name]
        for i in range(age_hots.shape[1]):
            df['{}_{}'.format(sens_name, i)] = age_hots[:, i]
    else:
        df[sens_name] = df[sens_name].apply(lambda x: float(x >= binning[1]))

    return df

def acs_dataset_age_pinfo(features, feature_names, enumerate_columns):
    protected_columns = [ i for i, f in enumerate(feature_names) if 'AGEP_' in f or 'AGEP' == f ]
    unprotected_columns = [ i for i, f in enumerate(feature_names) if i not in protected_columns ]

    protected_domain = generate_protected_domain(features[:, protected_columns])

    return ProtectedInfo(protected_columns, unprotected_columns,
                         protected_domain, enumerate_columns)
    

def acs_multisensitive_preprocessing(features, feature_names, col_num, binning=DEFAULT_AGE_BINNING):
    # Encoding sensitive attribute binning
    old_sens_col = features[:, col_num]
    if len(binning) > 3:
        sens_binning = pd.cut(old_sens_col, binning, labels=range(len(binning)-1))
        sens_hots = np.array(LabelBinarizer().fit_transform(sens_binning).tolist())
        sens_name = feature_names[col_num]

        features = np.delete(features, col_num, 1)
        del feature_names[col_num]

        features = np.concatenate((features, sens_hots), axis=1)
        for i in range(sens_hots.shape[1]):
            feature_names.append('{}_{}'.format(sens_name, i))
    else:
        # Take the middle value make a binary split
        sens_col = np.array(old_sens_col >= binning[1], dtype=float)
        features[:, col_num] = sens_col

    return features, feature_names

def german_credit_age_dataset(binning: List[float] = DEFAULT_AGE_BINNING) -> Dataset:
    #enumerate_columns = list(range(2, 57))
    not_enumerate = ['month', 'credit_amount']

    # Determine the sensitive / protected names
    if len(binning) > 3:
        protected_attributes = ['age_{}'.format(i) for i in range(len(binning)-1)]
    else:
        protected_attributes = ['age']

    # Getting data via AIF360
    dataset_orig = GermanDataset(
        protected_attribute_names=protected_attributes,

        privileged_classes=[],
        features_to_drop=['personal_status', 'sex'], ## Maybe we add these?
        custom_preprocessing=lambda x: aif360_multisensitive_preprocessing(
            x, binning)
    )
    df, info = dataset_orig.convert_to_dataframe()

    enumerate_columns = [
        i for i, n in enumerate(df.columns) if \
            n not in not_enumerate and n not in info['label_names']]

    # Originally 2 = Bad Credit; 1 = Good Credit -> 1, 0 ...
    x = df.loc[:, ~df.columns.isin(info['label_names'])].to_numpy()
    y = df[info['label_names']].to_numpy() - 1

    # Calculate protected information
    p_info = aif360_dataset_age_pinfo(df, info, enumerate_columns)

    feature_names = [n for n in df.columns if n not in info['label_names']]

    return Dataset(x, y, p_info, feature_names)

def bank_age_dataset(binning: List[float] = DEFAULT_AGE_BINNING) -> Dataset:
    not_enumerate = ['duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

    # Determine the sensitive / protected names
    if len(binning) > 3:
        protected_attributes = ['age_{}'.format(i) for i in range(len(binning)-1)]
    else:
        protected_attributes = ['age']

    # Getting data via AIF360
    dataset_orig = BankDataset(
        protected_attribute_names=protected_attributes,

        privileged_classes=[],
        custom_preprocessing=lambda x: aif360_multisensitive_preprocessing(
            x, binning)
    )
    df, info = dataset_orig.convert_to_dataframe()

    enumerate_columns = [
        i for i, n in enumerate(df.columns) if \
            n not in not_enumerate and n not in info['label_names']]

    x = df.loc[:, ~df.columns.isin(info['label_names'])].to_numpy()
    y = df[info['label_names']].to_numpy()

    # Calculate protected information
    p_info = aif360_dataset_age_pinfo(df, info, enumerate_columns)

    feature_names = [n for n in df.columns if n not in info['label_names']]

    return Dataset(x, y, p_info, feature_names)

def acs_age_dataset(binning: List[float] = DEFAULT_AGE_BINNING, year='2015') -> Dataset:
    #enumerate_columns = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    #enumerate_columns = [0, 1, 2, 3, 4, 7, 9]

    not_enumerate = [
        'SCHL',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
    ]

    acs_cols = [
        'AGEP',
        'RAC1P',
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
    ]

    ACSIncomeCustom = folktables.BasicProblem(
        features=acs_cols,
        target='PINCP',
        target_transform=lambda x: x > 50000,
        group='AGEP',
        preprocess=folktables.adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )
    protected_col = acs_cols.index('AGEP')

    data_source = folktables.ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)
    features, labels, _ = ACSIncomeCustom.df_to_numpy(acs_data)    
    labels = np.array(labels, dtype=float)

    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)

    features, feature_names = acs_multisensitive_preprocessing(features, acs_cols, protected_col, binning=binning)

    enumerate_columns = [i for i, n in enumerate(feature_names) \
         if n not in not_enumerate]

    p_info = acs_dataset_age_pinfo(features, feature_names, enumerate_columns)

    return Dataset(features, labels, p_info, feature_names)

def split_dataset(dataset: Dataset, test_size: float,
                  seed: int = 1) -> Tuple[Dataset, Dataset]:
    train_indices, test_indices = train_test_split(range(len(dataset)),
                                                   test_size=test_size,
                                                   random_state=seed)

    # Propagate split indices
    train_x, test_x = dataset.x[train_indices, :], dataset.x[test_indices, :]
    train_y, test_y = dataset.y[train_indices, :], dataset.y[test_indices, :]
    p_info = dataset.protected_info
    feature_names = dataset.feature_names

    train_dataset = Dataset(train_x, train_y, p_info, feature_names)
    test_dataset = Dataset(test_x, test_y, p_info, feature_names)
    return train_dataset, test_dataset

def cross_validate_dataset(dataset: Dataset, n_splits: int, random_state: int) -> Iterator[Tuple[Dataset, Dataset]]:
    # Find the indices
    cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    # Split via indices
    for train_indices, test_indices in cv.split(range(len(dataset))):
        train_data_x = dataset.x[train_indices, :]
        train_data_y = dataset.y[train_indices, :]
        test_data_x = dataset.x[test_indices, :]
        test_data_y = dataset.y[test_indices, :]

        train_dataset = Dataset(train_data_x, train_data_y, dataset.protected_info,
                                dataset.feature_names)
        test_dataset = Dataset(test_data_x, test_data_y, dataset.protected_info,
                               dataset.feature_names)

        yield train_dataset, test_dataset

class FeaturePreprocessor:
    def __init__(self, protected_info: ProtectedInfo, feature_names: List[str]) -> None:

        categorical_cols = protected_info.enumerate_columns

        self.protected_info = protected_info
        self.feature_names = feature_names
        
        self.cat_cols = [i for i in categorical_cols if \
            i not in protected_info.protected_columns]
        self.cts_cols = [i for i in range(len(feature_names)) if \
            i not in protected_info.protected_columns and \
            i not in categorical_cols]

        self.cat_transformers = [(
            self.feature_names[i],
            OneHotEncoder(drop='if_binary', sparse=False),
            [i]) for i in self.cat_cols]

        self.cts_transformers = [(
            self.feature_names[i],
            StandardScaler(),
            [i]) for i in self.cts_cols]

        self.transformer = ColumnTransformer(
            transformers=self.cat_transformers + self.cts_transformers,
            remainder='passthrough')

    def fit(self, x):
        df_x = pd.DataFrame(x, columns=self.feature_names)
        return self.transformer.fit(df_x)

    def transform(self, x):
        return self.transformer.transform(x)

    def get_feature_names_out(self):
        return self.transformer.get_feature_names_out()

def preprocess(dataset: Dataset, preprocessor: FeaturePreprocessor) -> Dataset:
    p_info = dataset.protected_info

    new_x = preprocessor.transform(dataset.x)
    new_feature_names = preprocessor.get_feature_names_out()

    protected_names = [dataset.feature_names[i] for i in p_info.protected_columns]
    new_protected_columns = [i for i in range(len(new_feature_names)) if \
        new_feature_names[i].split('__')[1] in protected_names]

    new_unprotected_columns = [i for i in range(len(new_feature_names)) if \
        i not in new_protected_columns]

    new_enumerated_columns = []
    for i in p_info.enumerate_columns:
        old_name = dataset.feature_names[i]
        for j, new_name in enumerate(new_feature_names):
            if new_name.split('__')[1].rsplit('_', 1)[0] == old_name:
                new_enumerated_columns.append(j)

    new_protected_info = ProtectedInfo(
        new_protected_columns, new_unprotected_columns, p_info.protected_domain,
        new_enumerated_columns)

    return Dataset(new_x, dataset.y, new_protected_info, new_feature_names)