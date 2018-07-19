# Simple average of the following:
# Light GBM overfit
# XGB 1.0

import pandas as pd
import numpy as np

DIR = '~/kaggle-home-credit/submissions/'

print('Starting Simple Averaging of LGB and XGB')

lgb = pd.read_csv(DIR + 'lgbm_overfit98.csv')
xgb = pd.read_csv(DIR + 'xgb.csv')

simple_avg = pd.DataFrame({
    
    'SK_ID_CURR': lgb['SK_ID_CURR'],
    'TARGET': 0.2*lgb['TARGET'] + 0.8*xgb['TARGET']
    
})

simple_avg.to_csv(DIR + 'simple_avg.csv', index = False)

print('Dataframe successfully saved')