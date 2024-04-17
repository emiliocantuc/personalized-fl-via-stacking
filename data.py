"""
This file admisters datasets used in the experiments.

Each dataset:
    - has a way to fetch itself from the internet (if not already in cache)
    - knows its natural partition column (if applicable)
    - performs basic preprocessing on its data:
        - remove rows with missing values
        - one hot encoding on categorical features
        - remove outliers
        - remove useless columns
    - dataset specific preprocessing
"""

import os, datetime
import pandas as pd
import ucimlrepo, kaggle

DATA_DIR = 'data'
RANDOM_SHUFFLING_STATE = 42

class Dataset:

    def __init__(self) -> None:
        self.fpath = os.path.join(DATA_DIR, f'{self}.csv')
        self.fetch() # Fetchs itself if not already in cache

    @property
    def is_classification(self):
        return self.task == 'classification'
    
    @property
    def is_regression(self):
        return self.task == 'regression'

    def df(self):
        return pd.read_csv(self.fpath)
    
    def preprocess(self): raise NotImplementedError
    
    def fetch(self): raise NotImplementedError
    
    def X_y(self):
        # TODO: do we use this?
        df = self.df()
        X = df.drop(columns = [self.target])
        y = df[self.target]
        if hasattr(self, 'natural_col'):
            X = X.drop(columns = self.natural_col)
        return X, y

    def __repr__(self) -> str:
        return f"{' '.join([i.title() for i in self.__class__.__name__.split('_')])}"
    

class Census(Dataset):

    def __init__(self) -> None:
        super().__init__()

        self.target = 'income'
        self.task = 'classification'

        # Natural partition column
        self.natural_col = 'native-country'

    def fetch(self):
        if not os.path.exists(self.fpath):
            fetch_uci('census', self.preprocess, self.fpath)

    def preprocess(self, df):

        # Drop rows with missing values
        df = df.dropna()

        # One-hot encodings
        one_hot_cols = [
            'education', 'occupation', 'sex', 'race', 'relationship',
            'marital-status', 'workclass', #'native-country'
        ]
        one_hot_cols = [i for i in one_hot_cols if i in df.columns]
        df = pd.get_dummies(df, columns = one_hot_cols)

        # Drop irrelevant columns
        df = df.drop(columns = ['fnlwgt'])

        # Replace y_column for categorical codes
        df['income'] = df['income'].str.contains('>').astype(int)
        # df['income'] = pd.Categorical(df['income']).codes

        # Group countries into larger groups
        replacements = {
            'latin-america':['Mexico','Puerto-Rico','El-Salvador','Cuba','Jamaica','Dominican-Republic','Guatemala','Columbia','Haiti','Nicaragua','Peru','Ecuador','Honduras'],
            'asia':['Philippines','India','China','Vietnam','Japan','Taiwan','Hong','Thailand','Laos','Cambodia'],
            'europe':['Germany','Italy','Poland','Portugal','Greece','France','Hungary','England','Scotland','Ireland'],
            'us':['United-States', 'Outlying-US(Guam-USVI-etc)']
        }

        for value,to_replace in replacements.items():
            df['native-country'] = df['native-country'].replace(to_replace, value)

        # Delete rows that do not appear in replacement keys
        df = df[df['native-country'].isin(replacements.keys())]

        return shuffle_reset_df(df) 

class Covertype(Dataset):
    
    def __init__(self) -> None:
        super().__init__()

        self.target = 'Cover.Type'
        self.task = 'classification'
        self.natural_col = 'Wilderness.Area'

    def fetch(self):
        if not os.path.exists(self.fpath):
            fetch_kaggle('uciml/forest-cover-type-dataset', self.preprocess, self.fpath)
            # fetch_uci('covertype', self.preprocess, self.fpath)

    def preprocess(self, df):

        # Replace '_' in columns with '.'
        df.columns = df.columns.str.replace('_','.')

        # Drop examples from covertypes not 1 or 2
        df = df[df['Cover.Type'].isin([1,2])]

       # Wilderness_Area rename
        wilderness_area_names={
            'Wilderness.Area1': 'Rawah',
            'Wilderness.Area2': 'Neota',
            'Wilderness.Area3': 'Comanche',
            'Wilderness.Area4': 'Poudre',
        }

        df = df.rename(columns = wilderness_area_names)
        df['Wilderness.Area'] = df[wilderness_area_names.values()].idxmax(axis=1)
        df = df.drop(columns = wilderness_area_names.values())

        # Eliminate areas where examples only have 1 class
        gb = df.groupby('Wilderness.Area')['Cover.Type'].nunique()
        to_remove = gb[gb < 2].index.to_list()
        df = df[~df['Wilderness.Area'].isin(to_remove)]

        # Convert columns with 'Soil' in them to bool
        soil_columns = [col for col in df.columns if 'Soil' in col]
        for col in soil_columns:
            df[col] = df[col].astype(bool)

        return shuffle_reset_df(df)
    

class Vehicle_Loan_Default(Dataset):

    def __init__(self) -> None:
        super().__init__()

        self.target = 'loan.default'
        self.task = 'classification'
        self.natural_col = 'branch.id'

    def fetch(self):
        if not os.path.exists(self.fpath):
            fetch_kaggle('mamtadhaker/lt-vehicle-loan-default-prediction', self.preprocess, self.fpath, 'train.csv')

    def preprocess(self, df):

        # Distracting columns
        cols_to_drop=[
            'UniqueID','supplier_id', 'Current_pincode_ID','State_ID','Employee_code_ID',
            'MobileNo_Avl_Flag','PRI.NO.OF.ACCTS','SEC.NO.OF.ACCTS','PRI.CURRENT.BALANCE',
            'PRI.SANCTIONED.AMOUNT','SEC.NO.OF.ACCTS','PRI.NO.OF.ACCTS','PRI.DISBURSED.AMOUNT','PRI.ACTIVE.ACCTS', 
            'PRI.OVERDUE.ACCTS','SEC.CURRENT.BALANCE','SEC.SANCTIONED.AMOUNT', 'SEC.OVERDUE.ACCTS',
            'SEC.DISBURSED.AMOUNT','PRIMARY.INSTAL.AMT','SEC.INSTAL.AMT','disbursed_amount','SEC.ACTIVE.ACCTS'
        ]
        df = df.drop(columns = cols_to_drop)

        # Replace '_' in columns with '.'
        df.columns = df.columns.str.replace('_','.')

        # Convert duration string to number of years
        def duration_in_years(duration_str):
            years = int(duration_str.split(' ')[0].replace('yrs',''))
            months = int(duration_str.split(' ')[1].replace('mon',''))
            return years+(months/12)
        
        df['AVERAGE.ACCT.AGE'] = df['AVERAGE.ACCT.AGE'].apply(duration_in_years)
        df['CREDIT.HISTORY.LENGTH'] = df['CREDIT.HISTORY.LENGTH'].apply(duration_in_years)

        # Calculate an age column
        def birth_year(date_str):
            year = int(date_str.split('-')[-1])
            return year+2000 if year <= 25 else year + 1900
        
        df['Date.of.Birth'] = df['Date.of.Birth'].apply(birth_year)
        df['DisbursalDate'] = df['DisbursalDate'].apply(birth_year)
        df['Age'] = df['DisbursalDate'] - df['Date.of.Birth']
        df=df.drop(columns = ['Date.of.Birth','DisbursalDate'])
        
        # Columns to one-hot-code encode
        one_hot_cols = ['Employment.Type','PERFORM.CNS.SCORE.DESCRIPTION']
        df = pd.get_dummies(df, columns = one_hot_cols)

        # Convert to bool cols with _flag in them to bool
        flag_columns = [col for col in df.columns if 'flag' in col]
        for col in flag_columns:
            df[col] = df[col].astype(bool)


        # Make explicit the number of features each customer is missing
        df['Missing Features'] = (df == 0).astype(int).sum(axis = 1)

        # Only keep branches with more than 5000 examples
        t = df['branch.id'].value_counts()[(df['branch.id'].value_counts() > 5000)].index.to_list()
        df = df.loc[df['branch.id'].isin(t)]
        df['branch.id'] = df['branch.id'].astype('category').cat.codes

        return shuffle_reset_df(df)

# class Vehicle(Dataset):
        
#     def __init__(self) -> None:
#         super().__init__()

#         self.target = 'price'
#         self.task = 'regression'

#         # Natural partition column
#         self.natural_col = 'state'

#     def fetch(self):
#         if not os.path.exists(self.fpath):
#             fetch_kaggle('austinreese/craigslist-carstrucks-data', self.preprocess, self.fpath)

#     def preprocess(self, df):
#         # Distracting columns - exclude 'state' as we will partition on it
#         cols_to_drop=['id', 'model','url', 'region', 'region_url', 'VIN', 'image_url', 'description',
#                     'county', 'lat', 'long', 'posting_date']

#         # Columns to one-hot-code encode
#         one_hot_cols=['manufacturer','condition','fuel','title_status','transmission','drive','size',
#                     'type','paint_color']

#         # Drop distracting columns
#         df=df.drop(columns=cols_to_drop)

#         # Drop rows with NaN values
#         df=df.dropna()

#         # Cylinders cleanup
#         df=df.drop(df[df['cylinders']=='other'].index)
#         df['cylinders']=df['cylinders'].str.split(expand=True)[0].astype('int64')

#         # Make year relative
#         df['year']=datetime.datetime.now().year-df['year']

#         # Delete price and odometer column outliers
#         outlier_cols=['price','odometer']
#         for col in outlier_cols:
#             upper,lower = df[col].quantile(0.99), df[col].quantile(0.1)
#             df = df[(df[col] < upper) & (df[col] > lower)]

#         # TEMP
#         t=df['state'].value_counts()[(df['state'].value_counts()>=2000)].index.to_list()
#         df=df.loc[df['state'].isin(t)]

#         # One-hot-code encodings
#         df=pd.get_dummies(df,columns=one_hot_cols)

#         return shuffle_reset_df(df)


# UTILS

def shuffle_reset_df(df, random_state = RANDOM_SHUFFLING_STATE):
    # Reset index and shuffle
    df = df.reset_index(drop = True)
    df = df.sample(frac = 1, random_state = random_state).reset_index(drop = True) 
    return df

def fetch_uci(name, preprocess_f, fpath):
    # Fetch and preprocess UCI datasets
    print(f'Fetching {name} dataset')
    df = preprocess_f(ucimlrepo.fetch_ucirepo(name)['data']['original'])
    save_dataset(df, fpath)

def fetch_kaggle(id, preprocess_f, fpath, csv_name = None):

    print(f'Fetching {id} dataset')

    # Download and unzip
    files_before = os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else set()
    kaggle.api.dataset_download_files(id, path = DATA_DIR, unzip = True)
    new_files = set(os.listdir(DATA_DIR)) - set(files_before)

    if csv_name is None:
        assert len(new_files) == 1, f'Expected 1 new file, got {len(new_files)}'
    else:
        assert csv_name in new_files, f'Expected {csv_name} in new files, got {new_files}'

        for f in new_files:
            if f != csv_name: os.remove(os.path.join(DATA_DIR, f))

        new_files = [csv_name]

    # Rename file to fpath
    os.rename(os.path.join(DATA_DIR, new_files.pop()), fpath)

    # Preprocess and save
    df = pd.read_csv(fpath)
    df = preprocess_f(df)
    save_dataset(df, fpath)


def save_dataset(df, fpath):
    os.makedirs(DATA_DIR, exist_ok = True)
    df.to_csv(fpath, index = False)
    print(f'Saved dataset to {fpath}')


def check_dataset(dataset_obj):
    for attr in ['fpath','target', 'task', 'fetch', 'preprocess', 'df', 'X_y', 'is_classification', 'is_regression']:
        assert hasattr(dataset_obj, attr), f'{dataset_obj} object must have a {attr} attribute'

    # Columns should not have '_' unless they are binary column
    for col, t in dataset_obj.df().dtypes.items():
        if t != bool:
            assert '_' not in col, f'Column {col} in {dataset_obj} should not have _ in its name'

if __name__ == '__main__':

    import inspect, sys

    for name, dataset_obj in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if 'Dataset' not in name:
            obj = dataset_obj()
            check_dataset(obj)
            print(f'{obj} passed checks. Fetching ...')
            obj.fetch()
    
    print('All datasets fetched and checked')
