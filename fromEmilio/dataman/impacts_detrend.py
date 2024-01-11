'''
Created on May 22, 2023

@author: politti
'''

import pandas as pd
import json
import os
import numpy as np 
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess  # the smoothening algorithm
import warnings
from loguru import logger
warnings.filterwarnings("ignore")
from datetime import datetime
import scipy.interpolate
import random
import matplotlib.pyplot as plt


def smooth(x, y, xgrid, frac):
    # https://james-brennan.github.io/posts/lowess_conf/
    samples = np.random.choice(len(x), 50, replace=True)
    y_s = y[samples]
    x_s = x[samples]
    y_sm = sm_lowess(y_s, x_s, frac=frac, return_sorted=False, missing='drop')
    # regularly sample it onto the grid
    y_grid = scipy.interpolate.interp1d(x_s, y_sm, fill_value='extrapolate')(xgrid)
    return y_grid


def explore_detrend(args):
    
    data_config = args['data_config']
    impacts_dir = args['impacts_dir']
    
    lowes = args['lowes']
    lowes_labels = args['lowes_labels']
    
    bins = args['bins']
        
    with open(data_config) as json_config:
        config = json_config.read()
    config = json.loads(config) 
    
    if(os.path.isdir(impacts_dir) == False):
        os.mkdir(impacts_dir)
    impacts = config['impacts']    
    
    bins_labels = []
    for bin_val in bins:
        v = bin_val * 100
        lab = 'higher'
        if(bin_val <= 0):
            lab = 'lower'
        lab = f'{bin_val}%_{lab}'
        bins_labels.append(lab)
        
    bins_dict = dict(zip(bins, bins_labels))
    
    for var_name, var_file in impacts.items():
        logger.info(f'Processing {var_name}')
        
        df = pd.read_csv(var_file)
        df.fillna(0, inplace=True)
        rename_col = df.columns[0]
        df.rename(columns={rename_col: "date"}, inplace=True)
        nuts_cols = list(df.columns)[1:]
        date_values = df['date']

        date_values = df['date']
        x = np.arange(0, len(date_values), 1, dtype=int)

        stop = False
        while(stop == False):
            
            ncol = random.randint(0, len(nuts_cols))
            col = nuts_cols[ncol]
            logger.info(f'Testing loess for {col}...')
            
            npp = df[col]
            
            if(np.count_nonzero(npp) > 5):
                
                for f, frac in enumerate(lowes):
                    lowes_label = lowes_labels[f]
                
                    y = npp.values
                    mask = np.logical_not(np.isnan(y))
                    x2 = x[mask]
                    y2 = y[mask]
                    
                    xgrid = np.linspace(x2.min(), x2.max())
                    k = 100
                    smooths = np.stack([smooth(x2, y2, xgrid, frac) for k in range(k)]).T
                    mean = np.mean(smooths, axis=1)
                    stderr = scipy.stats.sem(smooths, axis=1)
                    stderr = np.nanstd(smooths, axis=1, ddof=0)
                    tot_stderr = np.sum(stderr)
                    
                    plt.subplot(len(lowes), 1, f + 1)
                    
                    plt.fill_between(xgrid, mean - 1.96 * stderr, mean + 1.96 * stderr, alpha=0.25)
                    plt.plot(xgrid, mean, color='tomato')
                    plt.plot(x, y, 'k.')
                    plt.title(f'{var_name} frac: {lowes_label} stderr: {round(tot_stderr, 2)} ')
                    
                plt.show()
                
            else:
                logger.warning(f'{col} for varible {var_name} has less than 5 data points and will not be processed')

            user_in = input('Type a to see another nuts interpolation or any other key to continue to another variable \n')
            print()
               
            if(user_in.strip().lower() != 'a'):
                stop = True
    

def explore_detrend_batch(args, n_test=100):
    
    data_config = args['data_config']
    impacts_dir = args['impacts_dir']
    
    lowes = args['lowes']
    lowes_labels = args['lowes_labels']
    
    bins = args['bins']
        
    with open(data_config) as json_config:
        config = json_config.read()
    config = json.loads(config) 
    
    if(os.path.isdir(impacts_dir) == False):
        os.mkdir(impacts_dir)
    impacts = config['impacts']    
    
    bins_labels = []
    for bin_val in bins:
        v = bin_val * 100
        lab = 'higher'
        if(bin_val <= 0):
            lab = 'lower'
        lab = f'{bin_val}%_{lab}'
        bins_labels.append(lab)
    
    test_summary = {'variable':[], 'fraction':[], 'mean standard error': [], 'sd standard error': []}
    
    for var_name, var_file in impacts.items():
        logger.info(f'Processing {var_name}')
        
        df = pd.read_csv(var_file)
        df.fillna(0, inplace=True)
        rename_col = df.columns[0]
        df.rename(columns={rename_col: "date"}, inplace=True)
        nuts_cols = list(df.columns)[1:]
        date_values = df['date']
        
        errors = []

        date_values = df['date']
        x = np.arange(0, len(date_values), 1, dtype=int)
        
        for f, frac in enumerate(lowes):
            lowes_label = lowes_labels[f]

            for _n in range(0, n_test):
                
                ncol = random.randint(0, len(nuts_cols) - 1)
                col = nuts_cols[ncol]
                logger.info(f'Testing loess for nut {col}...')
                
                npp = df[col]
                
                if(np.count_nonzero(npp) > 5):
                    
                    y = npp.values
                    mask = np.logical_not(np.isnan(y))
                    x2 = x[mask]
                    y2 = y[mask]
                    
                    xgrid = np.linspace(x2.min(), x2.max())
                    k = 100
                    smooths = np.stack([smooth(x2, y2, xgrid, frac) for k in range(k)]).T
                    mean = np.mean(smooths, axis=1)
                    stderr = scipy.stats.sem(smooths, axis=1)
                    stderr = np.nanstd(smooths, axis=1, ddof=0)
                    tot_stderr = np.sum(stderr)
                    
                    errors.append(tot_stderr)
        
            avg_err = round(np.mean(errors), 1)
            esd = round(np.std(errors), 1)
            test_summary['variable'].append(var_name)
            test_summary['fraction'].append(lowes_label)
            test_summary['mean standard error'].append(avg_err)
            test_summary['sd standard error'].append(esd)

    df_summary = pd.DataFrame(test_summary)
    print(df_summary.head(200))
    
    df_summary.to_csv('impacts_detrend_diagnostics.csv', index=False)


def detrend(args):
    
    start_time = datetime.utcnow()
    
    nuts_level = args['nuts_level']
    
    data_config = args['data_config']
    impacts_dir = args['impacts_dir']
    exposure_dir = args['exposure_dir']
    
    xcl_file_template_name = args['xcl_file_template_name']
    verbose = args['verbose']  
    
    lowes = args['lowes']
    lowes_labels = args['lowes_labels']
    
    bins = args['bins']
    bins_eval = args['bins_eval']
        
    with open(data_config) as json_config:
        config = json_config.read()
    config = json.loads(config) 
    
    if(os.path.isdir(impacts_dir) == False):
        os.mkdir(impacts_dir)
    impacts = config['impacts'] 
    
    if(os.path.isdir(exposure_dir) == False):
        os.mkdir(exposure_dir)
    exposures = config['exposure']    
    
    dicts = {}
    dicts.update(impacts)
    dicts.update(exposures)

    bins_labels = []
    for bin_val in bins:
        v = bin_val * 100
        lab = 'higher'
        if(bin_val <= 0):
            lab = 'lower'
        lab = f'{bin_val}%_{lab}'
        bins_labels.append(lab)
        
    bins_dict = dict(zip(bins, bins_labels))
    
    anomaly_summary = {'var':[],
                       'fraction':[],
                       # 'mean_trend':[],
                       'bin':[],
                       'bin_class_sum':[]
        }
        
    for var_name, var_file in dicts.items():
        
        out_dir = None
        if(var_name in impacts.keys()):
            out_dir = impacts_dir
        if(var_name in exposures.keys()):
            out_dir = exposure_dir
        
        logger.info(f'Processing {var_name}')
        
        df = pd.read_csv(var_file)
        df.fillna(0, inplace=True)
        rename_col = df.columns[0]
        df.rename(columns={rename_col: "date"}, inplace=True)
        nuts_cols = list(df.columns)[1:]
        date_values = df['date']
        
        for f, frac in enumerate(lowes):
            logger.info(f'Running lowes analysis for fraction {round(frac,2)}')
            date_values = df['date']
            x = np.arange(0, len(date_values), 1, dtype=int)
            lowes_label = lowes_labels[f]
            
            xcl_file_name = xcl_file_template_name.format(var_name, nuts_level, lowes_label)
            xcl_file_name = os.path.join(out_dir, xcl_file_name)
            writer = pd.ExcelWriter(xcl_file_name, engine='xlsxwriter')
            
            dfdt = pd.DataFrame(0, index=date_values, columns=nuts_cols)
            dfdty = pd.DataFrame(0, index=date_values, columns=nuts_cols)
            dfdtt = pd.DataFrame(0, index=date_values, columns=nuts_cols)
            
            df_bins_dict = {}
            for bin_value in bins:
                df_bin = pd.DataFrame(0, index=date_values, columns=nuts_cols)
                df_bins_dict[bin_value] = df_bin
            
            for col in nuts_cols:
                npp = df[col]
                
                if(np.count_nonzero(npp) > 5):
                    if(verbose):
                        logger.info(f'Detrending {col} NUT for {round(frac,2)}')
                    
                    y = npp.values
                    mask = np.logical_not(np.isnan(y))
                    x2 = x[mask]
                    y2 = y[mask]
                    trend = y.copy()
    
                    # expected value
                    trd = sm_lowess(y2, x2, frac=frac, missing='drop', return_sorted=False).astype('float')               
                    trend[mask] = trd
                            
                    # anomalies comapred to expected value
                    z = (y - trend) / (trend)
                    
                    # histo_dict = {'obs':y, 'pred': trend, 'anomalies': z}
                    # histo = multivars_histo(vars_dict = histo_dict)
                    # histo.show()
                    
                    # zero yield for nan
                    dfdty[col] = y
                    dfdty[col][np.isnan(y)] = np.nan
                    dfdtt[col] = trend
                    dfdtt[col][np.isnan(y)] = np.nan
                    dfdt[col] = z
                    dfdt[col][np.isnan(y)] = np.nan
                    
                    # bins = [0, -0.02, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3, -0.33]
                    # z: detrended_NPP
                    for bin_val in bins:
                        df_bin = df_bins_dict[bin_val]
                        df_bin[col][z < bin_val] = 1  # TODO replace this with eval if the operator can be other than < 
                        df_bin[col][np.isnan(y)] = np.nan
                    
                    z_mean_det = round(np.nanmean(z.astype('float')), 3)
                    logger.info(f"{col} calculated, mean trend: {round(np.nanmean(z.astype('float')), 3)}")
                
                else:
                    logger.warning(f'{col} does not have enough data for detrending')
                
            dfdty.to_excel(writer, sheet_name=f"NPP")
            dfdtt.to_excel(writer, sheet_name=f"trend_NPP")
            dfdt.to_excel(writer, sheet_name=f"detrended_NPP")     
            
            for bin_val, bin_label in bins_dict.items():
                df_bin = df_bins_dict[bin_val]
                sheet_name = f'{bin_label}'
                df_bin.to_excel(writer, sheet_name=sheet_name)
                
                low_sum = df_bin[nuts_cols].sum(axis=0).sum()
                frac_lab = round(frac, 2)
                logger.info(f'{var_name} anomaly test sum for fraction: {frac_lab}, bin: {bin_label} is {low_sum}')
                
                anomaly_summary['var'].append(var_name)
                anomaly_summary['fraction'].append(frac_lab)
                # anomaly_summary['mean_trend'].append(z_mean_det)
                anomaly_summary['bin'].append(bin_label)
                anomaly_summary['bin_class_sum'].append(low_sum)
                                
                print()
                
            writer.close()
    
        logger.info('--------------------------------------------------------------------------------------------\n')    
    
    df_summary = pd.DataFrame(anomaly_summary)
    df_summary.to_csv(os.path.join(impacts_dir, 'impacts_detrend_stats.csv'), index=False)
    
    logger.info(f'Start time: {start_time.time()}')    
    end_time = datetime.utcnow()
    logger.info(f'End time:  {end_time.time()}')  
    # delta = start_time - end_time
    # n = delta.seconds
    # time_format = time.strftime("%H:%M:%S", time.gmtime(n))
    # logger.info(f'Total elapsed time: {time_format}')
    
    return 0


if __name__ == '__main__':
    
    nuts_level = 2
    # data_config = '../csv_data_nuts_3/nuts_3_csv_datamap.json'
    # impacts_dir = '../csv_data_nuts_3/impacts' 
    data_config = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/csv_data_nuts_{nuts_level}/nuts_{nuts_level}_csv_datamap.json'
    
    # edit and use this one when only one/few vars need to be detrended
    data_config = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/csv_data_nuts_{nuts_level}/nuts_{nuts_level}_single_csv.json'
    
    impacts_dir = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/csv_data_nuts_{nuts_level}/impacts'
    exposure_dir = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/csv_data_nuts_{nuts_level}/exposure' 
    
    # data_config = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/test_ws/nuts_{nuts_level}_csv_datamap_test.json'
    # impacts_dir = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/test_ws/impacts'
    
    xcl_file_template_name = 'detrended_{}_nuts_{}_{}.xlsx'
    verbose = True   
    
    lowes = [1 / 3, 1 / 4, 1 / 2, 2 / 3, 3 / 4]
    lowes_labels = ['1-3', '1-4', '1-2', '2-3', '3-4']
    
    bins = [0, -0.02, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3, -0.33]
    bins_eval = ['z < 0', 'z < -0.02', 'z < -0.05', 'z < -0.1', 'z < -0.15', 'z < -0.2', 'z < -0.25', 'z < -0.3', 'z < -0.33']
    
    args = {'nuts_level': nuts_level,
            'data_config': data_config,
            'impacts_dir': impacts_dir,
            'exposure_dir': exposure_dir,
            'xcl_file_template_name': xcl_file_template_name,
            'verbose': verbose,
            'lowes': lowes,
            'lowes_labels': lowes_labels,
            'bins': bins,
            'bins_eval': bins_eval
            }

    detrend(args)
    # explore_detrend(args)
    # explore_detrend_batch(args, n_test=100)
    
    print ('Process completed.')

