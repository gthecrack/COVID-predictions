from argparse import ArgumentParser
from src.model import *
import pandas as pd

parser = ArgumentParser(description='Program that receives a model and an option to use the normalized dataset and prints out the results')
parser.add_argument('-m','--model', type=str, help='Introduce the model to evaluate the dataset with')
parser.add_argument('-n','--normalize', type=str, help='Introduce yes/no if you want to use the normalized dataset')
args = parser.parse_args()
print(args)

def main():
    if args.normalize == 'yes':
        df = openDataframeNormalized()
        print(df)
        if args.model == 'ET':
            saved_final_et_n = load_model('Final ET Normalized Model 30Jul2020')
            new_prediction = predict_model(saved_final_et_n, data=df)
        elif args.model == 'Catboost':
            saved_final_catboost_n = load_model('Final Catboost Normalized Model 30Jul2020')
            new_prediction = predict_model(saved_final_catboost_n, data=df)
        elif args.model == 'LR':
            saved_final_lr_n = load_model('Final LR Normalized Model 30Jul2020')
            new_prediction = predict_model(saved_final_lr_n, data=df)
        elif args.model == 'RF':
            saved_final_rf_n = load_model('Final RF Normalized Model 30Jul2020')
            new_prediction = predict_model(saved_final_rf_n, data=df)
        elif args.model == 'Xgboost':
            saved_final_xgboost_n = load_model('Final Xgboost Normalized Model 30Jul2020')
            new_prediction = predict_model(saved_final_xgboost_n, data=df)
        return print(new_prediction)
    else:
        df = openDataframe()
        print(df)
        df = openDataframeNormalized()
        print(df)
        if args.model == 'ET':
            saved_final_et = load_model('Final ET Model 30Jul2020')
            new_prediction = predict_model(saved_final_et, data=df)
        elif args.model == 'Catboost':
            saved_final_catboost = load_model('Final Catboost Model 30Jul2020')
            new_prediction = predict_model(saved_final_catboost, data=df)
        elif args.model == 'LR':
            saved_final_lr = load_model('Final LR Model 30Jul2020')
            new_prediction = predict_model(saved_final_lr, data=df)
        elif args.model == 'RF':
            saved_final_rf = load_model('Final RF Model 30Jul2020')
            new_prediction = predict_model(saved_final_rf, data=df)
        elif args.model == 'Xgboost':
            saved_final_xgboost = load_model('Final Xgboost Model 30Jul2020')
            new_prediction = predict_model(saved_final_xgboost, data=df)
        return print(new_prediction)


if __name__ == "__main__":
    print(main())