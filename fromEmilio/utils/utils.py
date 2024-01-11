'''
Created on May 5, 2023

@author: politti
'''

import json
import os
import requests
from loguru import logger
import numpy as np
import pandas as pd
import time
from datetime import date, timedelta
import platform

countries = ['albania', 'bosnia and herzegovina', 'republic of north macedonia', 'montenegro', 'romania', 'serbia', 'kosovo', 'republic of moldova', 'ukraina']
country_codes = ['AL', 'BIH', 'MK', 'MNE', 'RO', 'RS', 'XKO', 'MDA', 'UKR']
country_code_mapper = dict(zip(countries, country_codes))
code_country_mapper = dict(zip(country_codes, countries))



def load_working_dir():
    dirs_config = '../data/indices_working_dirs.json'
    
    if(platform.system().lower() == 'windows'):
        dirs_config = '../data/win_indices_working_dirs.json'
    
    dirs_config = get_config(dirs_config)
    
    choics = []
    for i, k in enumerate(dirs_config.keys()):
        print(f'{i} {k}')
        choics.append(str(i))
        
    kn = input('Enter the number corresponding to the working directory of choice: ')
    if kn not in choics:
        print('Choice not recognized, program will exit.')
        exit(-1)
    
    wk = list(dirs_config.keys())[int(kn)]
    wdir = dirs_config[wk]
    
    if(os.path.isdir(wdir) is False):
        print(f'{wdir} does not exist. Program will exit.')
        exit(-1)
    
    os.chdir(wdir)
    logger.info(f'Moving to directory : {wdir}')
    
    return 0


def check_date_format(df):
    fix_date = False
    dtype = None
    d = df.loc[0][0]
    if(isinstance(d, str)):
        my = d.split('-')
        m = my[0]
        m = int(m)
        if(m>12):
            fix_date = True
            dtype = str
    if(isinstance(d, int)):
        fix_date = True
        dtype = int
    if(isinstance(d, float)):
        fix_date = True
        dtype = float
    return fix_date, dtype
        
def fix_cols(f_in, f_out, date_col='timing', fix_order=True, show=True, inspect = True, start_year = 2021, end_year = 2100):
    
    df = pd.read_csv(f_in)
    
    if(df.columns[0] == 'Unnamed: 0'):
        df.drop(columns = ['Unnamed: 0'], inplace = True)

    if(df.columns[0] == 'monyear'):
        df.rename(columns={'monyear': date_col}, inplace=True)
        
    if(fix_order):
        fix_order, _dtype = check_date_format(df)
        
        if(fix_order):
            dates = []
            years = list(range(start_year, end_year + 1)) * 12
            years = sorted(years)
            i_months = range(1, 13)
            months = []
            for m in i_months:
                ms = str(m)
                if(len(ms)<2):
                    ms = '0' + ms
                months.append(ms)
                
            months = months * (end_year - start_year + 1)
            
            for i, y in enumerate(years):
                m = months[i]
                d = f'{m}-{y}'
                
                dates.append(d)
                
            if(df.columns[0] == 'timing'):
                df['timing'] = dates
            else:
                df.insert(loc=0, column = 'timing', value = dates)
                
    df.to_csv(f_out, index=False)
    
    
    if(show):
        fname = os.path.basename(f_out)
        print('\n', fname)
        print(df[0:4].head(3))
        print()
        if(inspect):
            _go_on = input('Press any key to proceed.')
    
    logger.info(f'{f_out} saved.')
    
    return 0    

# def fix_cols(f_in, f_out, date_col='timing', fix_order=True, show=True, inspect = True, start_year = 2021, end_year = 2100):
#
#     df = pd.read_csv(f_in)
#
#     if(df.columns[0] == 'Unnamed: 0'):
#         df.drop(columns = ['Unnamed: 0'], inplace = True)
#
#     if(fix_order):
#         fix_order, dtype = check_date_format(df)
#
#         if(dtype == int or dtype == float):
#             dates = []
#             years = list(range(start_year, end_year + 1)) * 12
#             years = sorted(years)
#             i_months = range(1, 13)
#             months = []
#             for m in i_months:
#                 ms = str(m)
#                 if(len(ms)<2):
#                     ms = '0' + ms
#                 months.append(ms)
#
#             months = months * (end_year - start_year + 1)
#
#             for i, y in enumerate(years):
#                 m = months[i]
#                 d = f'{m}-{y}'
#
#                 dates.append(d)
#
#
#
#     if(fix_order):
#
#         if(dtype == str):
#             for i, row in df.iterrows():
#                 t = row[0]
#                 t = t.split('-')
#                 m = t[1]
#                 y = t[0]
#                 d = f'{m}-{y}'
#                 df.at[i, 0] = d
#
#         if(dtype == float):
#             df.insert(loc=0, column = 'timing', value = dates)
#
#     cols = list(df.columns)
#     c0 = cols[0]
#     if(c0 != date_col):
#         df.rename(columns={c0: date_col}, inplace=True)
#
#     df.to_csv(f_out, index=False)
#
#
#     if(show):
#         fname = os.path.basename(f_out)
#         print('\n', fname)
#         print(df[0:4].head(3))
#         print()
#         if(inspect):
#             _go_on = input('Press any key to proceed.')
#
#     logger.info(f'{f_out} saved.')
#
#     return 0


def dump_json_to_file(pydict, out_file, indent=4):
    try:
        with open(out_file, 'w') as fp:
            json.dump(pydict, fp, indent=indent)
        return 0
    except:
        raise

    
def get_config(json_file):

    with open(json_file) as json_config:
        config = json_config.read()
      
    config = json.loads(config)     
    return config


def download_from_url(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        logger.info("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
        return file_path
    else:  # HTTP status code 4XX/5XX
        logger.error("Download failed: status code {}\n{}".format(r.status_code, r.text))


class InvalidCodeOrCountry(Exception):
    'Country name or code not recognized'
    pass


def get_country_or_code(country_or_code):
    ret = ''
    if(country_or_code.upper() in country_codes):
        ret = code_country_mapper[country_or_code.upper()]  # return country name
    elif(country_or_code.lower() in countries):
        ret = country_code_mapper[country_or_code.lower()]  # return country code
    else:
        logger.error(country_or_code + ' is not a valid country code or country name')
        raise InvalidCodeOrCountry
    return ret

        
def set_country_code(df, country_col='country', code_col='code'):
    df[code_col] = ''
    countries = list(df[country_col].unique())
    
    for c_name in countries:
        print(c_name)
        c_code = get_country_or_code(c_name)
        df.loc[df[country_col] == c_name, code_col] = c_code
    return df

    
def join_shape_and_table(gdf, df, gdf_code_col='CNTR_CODE', df_country_col='country'):
    df = set_country_code(df, country_col=df_country_col)
    gdf_inlan_join = gdf.set_index(gdf_code_col).join(df.set_index(df_country_col)).reset_index()
    return gdf_inlan_join


def remove_duplicate_columns(df):
    df = df = df.T.drop_duplicates().T
    return df


def ensure_states(df, all_states):
    
    cols = list(df.columns)
    for s in all_states:
        if(s not in cols):
            df[s] = np.nan
            
    return df

