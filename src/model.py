
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from pycaret.classification import *
import shap

"""
saved_final_et = load_model('Final ET Model 30Jul2020')
saved_final_et_n = load_model('Final ET Normalized Model 30Jul2020')
saved_final_catboost = load_model('Final Catboost Model 30Jul2020')
saved_final_catboost_n = load_model('Final Catboost Normalized Model 30Jul2020')
saved_final_lr = load_model('Final LR Model 30Jul2020')
saved_final_lr_n = load_model('Final LR Normalized Model 30Jul2020')
saved_final_rf = load_model('Final RF Model 30Jul2020')
saved_final_rf_n = load_model('Final RF Normalized Model 30Jul2020')
saved_final_xgboost = load_model('Final Xgboost Model 30Jul2020')
saved_final_xgboost_n = load_model('Final Xgboost Normalized Model 30Jul2020')
"""

def openDataframe(file='./output/unseen_data.csv'):
    return pd.read_csv(file)

def openDataframeNormalized(file='./output/unseen_data_normalized.csv'):
    return pd.read_csv(file)
