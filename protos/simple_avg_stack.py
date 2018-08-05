# Simple average of the following:
# Light GBM overfit
# XGB 1.0

import pandas as pd
import numpy as np

DIR = '~/kaggle-home-credit/submissions/'

print('Starting Simple Averaging of LGB and XGB')

lgb = pd.read_csv(DIR + 'lgbm_dynamic.csv')
xgb = pd.read_csv(DIR + 'xgb.csv')
mix = pd.read_csv(DIR + 'WEIGHT_AVERAGE_RANK.csv')
stack = pd.read_csv(DIR + 'stacking_one.csv')

simple_avg = pd.DataFrame({
    
    'SK_ID_CURR': lgb['SK_ID_CURR'],
    #'TARGET': 0.3*lgb['TARGET'] + 0.4*xgb['TARGET'] + 0.3*mix['TARGET']
    
    'TARGET': 0.25*stack['TARGET'] + 0.25*xgb['TARGET'] + 0.25*mix['TARGET'] + 0.25*lgb['TARGET']

})

simple_avg.to_csv(DIR + 'simeple_avg_stack_one.csv', index = False)

print('Dataframe successfully saved')