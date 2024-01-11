'''
Created on May 16, 2023

@author: politti
'''

# from utils.utils import get_config, dump_json_to_file
import os
import shutil
import pandas as pd
import numpy as np
import math
import warnings
import pathlib
warnings.filterwarnings("ignore")


def cleanup_no_data_rows(files_list):
    
    for f in files_list:
        idxs = []
        df = pd.read_csv(f)
        cols = list(df.columns)[1:]
        
        for index, row in df.iterrows():
            if(math.isnan(np.amax(row[cols].values)) == True):
                idxs.append(index)
                
        print(idxs)
        
        df.drop(idxs, inplace=True)
        df.to_csv(f)
        print(f'{f} saved.')
                    
        
def cleanup_crops_no_data():
    files_list = [
                    '../data/csv_data_nuts_3/maize_nuts_3.csv',
                    '../data/csv_data_nuts_3/rice_nuts_3.csv',
                    '../data/csv_data_nuts_3/soybean_nuts_3.csv',
                    '../data/csv_data_nuts_3/wheat_nuts_3.csv'
                    ]
    
    cleanup_no_data_rows(files_list)
        
# def standardize_vars_filenames():
#     nut_level = 2
#     in_file = f'../data/nuts{nut_level}_data_map.json'
#     out_file = f'../data/csv_data_nuts_{nut_level}/nuts_{nut_level}_csv_datamap.json'
#     out_dir = f'../data/csv_data_nuts_{nut_level}'
#     files_suffix = f'_nuts_{nut_level}.csv'
#
#     in_dict = get_config(in_file)
#     out_dict = {}
#
#     for k, v in in_dict.items():
#         out_dict[k] = {}
#
#         for var_name, var_file in v.items():
#             fname = var_name + files_suffix
#             fpath = os.path.join(out_dir, fname)
#
#             shutil.copy(var_file, fpath) 
#             out_dict[k][var_name] = fpath
#
#     dump_json_to_file(pydict = out_dict, out_file = out_file, indent = 4)
#     print(f'{out_file} saved')
            
            
def read_tabular_datafile(data_file, worksheet=None):
    
    df = None
    ext = pathlib.Path(data_file).suffix
    try:
        if(ext == '.csv'):
            df = pd.read_csv(data_file)
        elif(ext == '.xlsx' or ext == '.xls'):
            if(worksheet is None):
                raise ValueError('Worksheet name must be provided when reading and excel file.')
            else:
                df = pd.read_excel(data_file, sheet_name=worksheet)
        else:
            print(f'{ext} of file {data_file} not recognized.')
        return df
    except Exception as e:
        print (e)  # TODO replace eventually with logger
        raise 
        
        
def test_read_tabular():
    xl = '/home/politti/git/wbalkan_drought_assess/wbalkan/data/base/test_ws/detrended_hydropower_prod_nuts_2_0.3333333333333333.xlsx'
    worksheet = 'NPP'
    cs = '/home/politti/git/wbalkan_drought_assess/wbalkan/data/base/csv_data_nuts_2/wgi_nuts_2.csv'
    
    df = read_tabular_datafile(data_file=xl, worksheet=worksheet)
    print(df.head())
    
    df = read_tabular_datafile(data_file=cs)
    print(df.head())


if __name__ == '__main__':
    # standardize_vars_filenames()
    # cleanup_crops_no_data()
    test_read_tabular()
    print('Process completed')
    
    
