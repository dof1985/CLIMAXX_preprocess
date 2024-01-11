'''
Created on May 31, 2023

@author: politti
'''
#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import zipfile
import matplotlib.pyplot as plt
import gzip
import csv
import json

# In[2]:

base_dir = '/data'
indir = "input"
middir = r"metafiles"
ixdir = r"indicesfiles2"
outdir = r"output"

# In[3]:

NUTSfile = pd.read_excel(r"C:\Users\mws311\surfdrive\Documents\Projects\EDORA\Data-driven-analysis\NUTS_rasters_for_hazard_analysis\NUTS3_names2.xlsx")
nutsregions = NUTSfile['NAME_LATN'].values

# In[4]:


# functions
def moving_sum(a, b):
    
    cummuldata = np.cumsum(a, dtype=float)                 
    cummuldata[b:] = cummuldata[b:] - cummuldata[:-b]         
    cummuldata[:11] = np.nan  # first year ignore                                           
    
    return cummuldata


def moving_mean(a, b):

    cummuldata = np.cumsum(a, dtype=float)                 
    cummuldata[b:] = (cummuldata[b:] - cummuldata[:-b]) / float(b)        
    cummuldata[:11] = np.nan                                           
    
    return cummuldata

    
def write_to_excel(df, sheet):
    
    """This function writes df to excel"""

    col = 0
    for i, name in enumerate(df.columns):
        sheet.write(0, 1 + i, name)
    for i, date in enumerate(df.index):
        sheet.write(1 + i, 0, date)
    df = np.nan_to_num(df, copy=True, nan=0)
    for col, data in enumerate(df.T):
        sheet.write_column(1, col + 1, data)

    return


def calculate_Zval(refseries, fullseries, Index):

    # find fitting distribution for reference sereis                          
    if Index == 'SEI' or Index == 'NEI':  # Suggestions by Stagge et al. (2015)
        dist_names = ['genextreme', 'genlogistic', 'pearson3']  # 'fisk' 
    elif Index == 'SETI' or Index == 'NETI':  # Suggestions by Stagge et al. (2015)
        dist_names = ['genextreme', 'genlogistic', 'pearson3']  # 'fis               
    elif Index == 'SSMI' or Index == 'NSMI':  # Suggestions in Ryu et al. (2005)
        dist_names = ['genextreme', 'norm', 'beta']  # ,'pearson3'
    elif Index == 'SPEI' or Index == 'NPEI':  # Suggestions in Ryu et al. (2005)
        dist_names = ['genextreme', 'norm', 'beta', 'pearson3']  
    elif Index == 'STI' or Index == 'NTI': 
        dist_names = ['genextreme', 'norm']  # ,'beta','pearson3'
    elif Index == 'SPI' or Index == 'NPI':  # Suggestions by Stagge et al. (2015)
        dist_names = ['logistic', 'weibull_min', 'gumbel_r']  # 'gamma',
    elif Index == 'SQI'  or Index == 'NQI':  # Suggestions by Vincent_Serrano et al. (2012) Modarres 2007 https://agupubs-onlinelibrary-wiley-com.vu-nl.idm.oclc.org/doi/full/10.1002/2016WR019276
        dist_names = ['genlogistic', 'lognorm', 'weibull_min' ]  # 'weibull_min', 'kappa3''fisk','gumbel','genpareto','weibull_min', 

    # find fit for each optional distribution on reference data
    dist_results = []
    params = {}
    for dist_name in dist_names:  # Find distribution parameters       
        dist = getattr(stats, dist_name)
        param = dist.fit(np.sort(refseries))
        params[dist_name] = param

        # Assess fitting of different distributions on reference data
        D, p = stats.kstest(np.sort(refseries), dist_name, args=param)  # Applying the Kolmogorov-Smirnov test
        dist_results.append((dist_name, p))          

    # find best fitting statistical distribution on reference data
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))  # Select the best fitted distribution
    dist = getattr(stats, str(best_dist))                                  
    rv = dist(*params[best_dist])
    # print(best_dist)
    
    # fit historic time series over best distribution    
    if best_dist == 'gamma': 
        nyears_zero = len(fullseries) - np.count_nonzero(fullseries)
        p_zero = nyears_zero / len(fullseries)
        p_zero_mean = (fullseries + 1) / (2 * (len(fullseries) + 1))          
        ppd = (fullseries.copy() * 0) + p_zero_mean
        if len(fullseries[np.nonzero(fullseries)]) > 0:
            ppd[np.nonzero(fullseries)] = p_zero + ((1 - p_zero) * rv.cdf(fullseries[np.nonzero(fullseries)]))                
    else:
        ppd = rv.cdf(fullseries)

    # print(ppd)
    Zval = stats.norm.ppf(ppd)                              
    Zval[Zval > 4] = 4
    Zval[Zval < -4] = -4 
    
    return Zval

# In[5]:


# functions 
def calculate_index(data, data_his, Ind):
        
    # amount of full years in data
    years = int(len(data) / 12)
    years_his = int(len(data_his) / 12)
    
    indices = np.zeros(len(data)) * np.nan
    for m in list(range(12)): 
              
        valuesofmonth = np.zeros(years) * np.nan                    
        for yr in list(range(years)): 
            valuesofmonth[yr] = data[(12 * yr) + m]                                                                       
        valuesofmonth_his = np.zeros(years_his)                      
        for yr in list(range(years_his)): 
            valuesofmonth_his[yr] = data_his[(12 * yr) + m]    
        
        # find gapts
        Z2 = valuesofmonth.copy() * np.nan
        yearswithnovalue = np.isnan(valuesofmonth) 
        
        # remove gaps
        valuesofmonth = valuesofmonth[np.logical_not(yearswithnovalue)]      
        valuesofmonth_his = valuesofmonth_his[np.isfinite(valuesofmonth_his)]

        if len(valuesofmonth_his) > 5:
            # extract reference time series and calculate Z values         
            Z = calculate_Zval(valuesofmonth_his, valuesofmonth, Ind)            

            # rescale Z
            Z2[np.logical_not(yearswithnovalue)] = Z

            # Reconstruct time series  
            for yr in list(range(years)):
                indices[(12 * yr) + m] = Z2[yr]
    
    return indices


def calculate_share(data, data_his, Ind):
         
    years = int(len(data) / 12)
    years_his = int(len(data_his) / 12)
    
    index = np.zeros(len(data)) * np.nan
    for m in list(range(12)): 
              
        valuesofmonth = np.zeros(years) * np.nan                    
        for yr in list(range(years)): 
            valuesofmonth[yr] = data[(12 * yr) + m]                                                                       
        valuesofmonth_his = np.zeros(years_his)                      
        for yr in list(range(years_his)): 
            valuesofmonth_his[yr] = data_his[(12 * yr) + m]                        
      
        D2 = valuesofmonth.copy() * np.nan
        yearswithnovalue = np.isnan(valuesofmonth) 
        
        valuesofmonth = valuesofmonth[np.logical_not(yearswithnovalue)]      
        valuesofmonth_his = valuesofmonth_his[np.isfinite(valuesofmonth_his)]

        if len(valuesofmonth_his) > 5:
            # calculate median (expected) value and deviations
            medianofmonth = np.median(valuesofmonth_his)
            D = 100 * valuesofmonth / medianofmonth
        
            D2[np.logical_not(yearswithnovalue)] = D
            
            for yr in list(range(years)):
                index[(12 * yr) + m] = D2[yr]
 
    return index

# In[ ]:


# create metafiles
for filename in os.listdir(indir):
    print(filename)
    
    meta_df = []
    meta_df = pd.DataFrame(columns=np.concatenate((np.array(['indicator', 'scenario', 'type', 'RCM', 'timing', 'year', 'month', 'accumulation']), nutsregions)))
        
    hydroindicator = filename.split('_')[0]
            
    data = pd.read_excel(os.path.join(indir, filename))
    # print(data.head())
    if hydroindicator == "evapotranspiration":
        data[data.columns[1:]] = data[data.columns[1:]] * -1
        
    data.rename(columns={'Unnamed: 0':'timing'}, inplace=True)
    data[['month', 'year']] = data.timing.str.split("-", expand=True)
    data['RCM'] = "ERA5"
    data['type'] = "HIS"
    data['scenario'] = "historical"
    data['indicator'] = hydroindicator   
    data['accumulation'] = 1     
            
    meta_df = pd.concat([meta_df, data], axis=0)
  
    accumulated3 = data.copy()
    accumulated3['accumulation'] = 3
    accumulated6 = data.copy()
    accumulated6['accumulation'] = 6   
    accumulated12 = data.copy()
    accumulated12['accumulation'] = 12  
    accumulated24 = data.copy()
    accumulated24['accumulation'] = 24 
            
    # calculate moving sum or average
    if (hydroindicator == 'Temperature') or (hydroindicator == 'SoilMoisture'): 
                for nuts in data.columns[1:-7]: 
                    print(nuts)
                    accumulated3[nuts] = moving_mean(accumulated3[nuts].values, 3)
                    accumulated6[nuts] = moving_mean(accumulated6[nuts].values, 6)
                    accumulated12[nuts] = moving_mean(accumulated12[nuts].values, 12)
                    accumulated24[nuts] = moving_mean(accumulated12[nuts].values, 24)
                    
    else: 
                for nuts in data.columns[1:-7]:
                    print(nuts)
                    accumulated3[nuts] = moving_sum(accumulated3[nuts].values, 3)
                    accumulated6[nuts] = moving_sum(accumulated6[nuts].values, 6)
                    accumulated12[nuts] = moving_mean(accumulated12[nuts].values, 12)
                    accumulated24[nuts] = moving_mean(accumulated12[nuts].values, 24)
                   
    # merge and save
    meta_df = pd.concat([meta_df, accumulated3], axis=0)
    meta_df = pd.concat([meta_df, accumulated6], axis=0)
    meta_df = pd.concat([meta_df, accumulated12], axis=0)
    meta_df = pd.concat([meta_df, accumulated24], axis=0)
    
    meta_df.to_csv(os.path.join(middir, hydroindicator + "_meta_hydro_file.csv.gz"), index=False, compression="gzip")
    meta_df = []

# In[ ]:

print(pd.read_csv(os.path.join(middir, "Precipitation" + '_meta_hydro_file.csv.gz')).columns.tolist()
)

# In[6]:

RCM = "ERA5"

# In[ ]:

# Calculate PR indices
precipitation_df = pd.read_csv(os.path.join(middir, "Precipitation" + '_meta_hydro_file.csv.gz'))  # outdir, _meta
# print(precipitation_df.head())

# Calculate indices
spi_df = precipitation_df.copy() 
spi_df['indicator'] = 'SPI'
npi_df = precipitation_df.copy() 
npi_df['indicator'] = 'SPI'
precipitation_df = []

for acc in np.unique(spi_df['accumulation']):
        print(acc)
        for nuts in spi_df.columns[8:-1]:
            print(nuts)
            # SPI
            spidata = spi_df.loc[(spi_df['accumulation'] == acc) & (spi_df['RCM'] == RCM)][nuts].values
            spi_his = spi_df.loc[(spi_df['scenario'] == 'historical') & (spi_df['accumulation'] == acc) & (spi_df['RCM'] == RCM)][nuts].values
            
            spi_df.loc[(spi_df['accumulation'] == acc) & (spi_df['RCM'] == RCM)][nuts] = calculate_index(spidata, spi_his, 'SPI')
            npi_df.loc[(npi_df['accumulation'] == acc) & (npi_df['RCM'] == RCM)][nuts] = calculate_share(spidata, spi_his, 'NPI')
                                                                         
spi_df.to_csv(os.path.join(ixdir, "SPI" + "_meta_hydro_file.csv.gz"), index=False, compression="gzip")
spi_df = []
npi_df.to_csv(os.path.join(ixdir, "NPI" + "_meta_hydro_file.csv.gz"), index=False, compression="gzip")
npi_df = []

# In[ ]:

# Calculate PET indices
potentialevaporation_df = pd.read_csv(os.path.join(middir, "evapotranspiration" + '_meta_hydro_file.csv.gz'))  # outdir, _meta
print(potentialevaporation_df.head())

sei_df = potentialevaporation_df.copy() 
sei_df['indicator'] = 'SEI'
nei_df = potentialevaporation_df.copy() 
nei_df['indicator'] = 'NEI'
potentialevaporation_df = []

for acc in np.unique(sei_df['accumulation']):
        print(acc)
        for nuts in sei_df.columns[8:-1]:
            print(nuts)                     

            seidata = sei_df.loc[(sei_df['accumulation'] == acc) & (sei_df['RCM'] == RCM)][nuts].values
            sei_his = sei_df.loc[(sei_df['scenario'] == 'historical') & (sei_df['accumulation'] == acc) & (sei_df['RCM'] == RCM)][nuts].values
            sei_df.loc[(sei_df['accumulation'] == acc) & (sei_df['RCM'] == RCM)][nuts] = calculate_index(seidata, sei_his, 'SEI')
            nei_df.loc[(nei_df['accumulation'] == acc) & (nei_df['RCM'] == RCM)][nuts] = calculate_share(seidata, sei_his, 'NEI')
                        
sei_df.to_csv(os.path.join(ixdir, "SEI" + "_meta_hydro_file.csv.gz"), index=False, compression="gzip")
sei_df = []
nei_df.to_csv(os.path.join(ixdir, "NEI" + "_meta_hydro_file.csv.gz"), index=False, compression="gzip")
nei_df = []

# In[ ]:

# Calculate PR-PET indices
precipitation_df = pd.read_csv(os.path.join(middir, "Precipitation" + '_meta_hydro_file.csv.gz'))  # outdir, _meta
potentialevaporation_df = pd.read_csv(os.path.join(middir, "Evapotranspiration" + '_meta_hydro_file.csv.gz'))  # outdir, _meta

# Calculate indices

spei_df = precipitation_df.copy() 
spei_df['indicator'] = 'SPEI'
seti_df = precipitation_df.copy() 
seti_df['indicator'] = 'SEI'
npei_df = precipitation_df.copy() 
npei_df['indicator'] = 'NPEI'
neti_df = precipitation_df.copy() 
neti_df['indicator'] = 'NEI'
potentialevaporation_df = []
precipitation_df = []

for acc in np.unique(spei_df['accumulation']):
        print(acc)
        for nuts in spei_df.columns[8:-1]:
            print(nuts)

            spidata = spei_df.loc[(spei_df['accumulation'] == acc) & (spei_df['RCM'] == RCM)][nuts].values
            seidata = spei_df.loc[(spei_df['accumulation'] == acc) & (spei_df['RCM'] == RCM)][nuts].values

            spei_df.loc[(spei_df['accumulation'] == acc) & (spei_df['RCM'] == RCM)][nuts] = calculate_index(spidata - seidata, spi_his - sei_his, 'SPEI')
            seti_df.loc[(seti_df['accumulation'] == acc) & (seti_df['RCM'] == RCM)][nuts] = calculate_index(spidata / seidata, spi_his / sei_his, 'SETI')

            npei_df.loc[(npei_df['accumulation'] == acc) & (npei_df['RCM'] == RCM)][nuts] = calculate_share(spidata - seidata, spi_his - sei_his, 'NPEI')
            neti_df.loc[(neti_df['accumulation'] == acc) & (neti_df['RCM'] == RCM)][nuts] = calculate_share(spidata / seidata, spi_his / sei_his, 'NETI')

spei_df.to_csv(os.path.join(ixdir, "SPEI" + "_meta_hydro_file.csv.gz"), index=False, compression="gzip")
seti_df.to_csv(os.path.join(ixdir, "SETI" + "_meta_hydro_file.csv.gz"), index=False, compression="gzip")
spei_df = []
seti_df = []
npei_df.to_csv(os.path.join(ixdir, "NPEI" + "_meta_hydro_file.csv.gz"), index=False, compression="gzip")
neti_df.to_csv(os.path.join(ixdir, "NETI" + "_meta_hydro_file.csv.gz"), index=False, compression="gzip")
npei_df = []
neti_df = []

# In[ ]:

# Calculate SM indices
soilmoisture_df = pd.read_csv(os.path.join(middir, "SoilMoisture" + '_meta_hydro_file.csv.gz'))  # outdir, _meta

# Calculate indices
ssmi_df = soilmoisture_df.copy() 
ssmi_df['indicator'] = 'SSMI'
nsmi_df = soilmoisture_df.copy() 
nsmi_df['indicator'] = 'NSMI'
soilmoisture_df = []

for acc in np.unique(ssmi_df['accumulation']):
        print(acc)
        for nuts in ssmi_df.columns[8:-1]:
            print(nuts)
            ssmidata = ssmi_df.loc[(ssmi_df['accumulation'] == acc) & (ssmi_df['RCM'] == RCM)][nuts].values
            ssmi_his = ssmi_df.loc[(ssmi_df['scenario'] == 'historical') & (ssmi_df['accumulation'] == acc) & (ssmi_df['RCM'] == RCM)][nuts].values
            ssmi_df.loc[(ssmi_df['accumulation'] == acc) & (ssmi_df['RCM'] == RCM)][nuts] = calculate_index(ssmidata, ssmi_his, 'SSMI')
            nsmi_df.loc[(nsmi_df['accumulation'] == acc) & (nsmi_df['RCM'] == RCM)][nuts] = calculate_share(ssmidata, ssmi_his, 'NSMI')

ssmi_df.to_csv(os.path.join(ixdir, "SSMI" + "_meta_hydro_file.csv.gz"), index=False, compression="gzip")
ssmi_df = []
nsmi_df.to_csv(os.path.join(ixdir, "NSMI" + "_meta_hydro_file.csv.gz"), index=False, compression="gzip")
nsmi_df = []

# In[ ]:

# Calculate Q indices
streamflow_df = pd.read_csv(os.path.join(middir, "Streamflow" + '_meta_hydro_file.csv.gz'))  # outdir, _meta

# Calculate indices
sqi_df = streamflow_df.copy() 
sqi_df['indicator'] = 'SQI'
nqi_df = streamflow_df.copy() 
nqi_df['indicator'] = 'NQI'
streamflow_df = []

for acc in np.unique(sqi_df['accumulation']):
        print(acc)
        for nuts in sqi_df.columns[8:-1]:
            print(nuts)

            # SQI
            sqidata = sqi_df.loc[(sqi_df['accumulation'] == acc) & (sqi_df['RCM'] == RCM)][nuts].values
            sqi_his = sqi_df.loc[(sqi_df['scenario'] == 'historical') & (sqi_df['accumulation'] == acc) & (sqi_df['RCM'] == RCM)][nuts].values

            sqi_df.loc[(sqi_df['accumulation'] == acc) & (sqi_df['RCM'] == RCM)][nuts] = calculate_index(sqidata, sqi_his, 'SQI')
            nqi_df.loc[(nqi_df['accumulation'] == acc) & (nqi_df['RCM'] == RCM)][nuts] = calculate_index(sqidata, sqi_his, 'NQI')
                                                     
sqi_df.to_csv(os.path.join(ixdir, "SQI" + "_meta_hydro_file.csv.gz"), index=False, compression="gzip")
sqi_df = []
nqi_df.to_csv(os.path.join(ixdir, "NQI" + "_meta_hydro_file.csv.gz"), index=False, compression="gzip")
nqi_df = []

# In[ ]:

# Rescale to yearly
for inputfile in os.listdir(ixdir):
    print(inputfile)
    indicator = inputfile.split('_')[0]

    inputdata = pd.read_csv(os.path.join(ixdir, inputfile))
    
    for SCE in np.unique(inputdata['scenario']): 
        print(SCE)
        for WL in np.unique(inputdata['type']):
            print(WL)
            for ACC in np.unique(inputdata['accumulation']):
                print("accumulation " + str(ACC))                   
            
                # get data
                alldata = inputdata.loc[(inputdata['RCM'] == RCM) & (inputdata['scenario'] == SCE) & (inputdata['type'] == WL) & (inputdata['accumulation'] == ACC)]  
                    
                if len(alldata) > 0:
                    years = list(np.unique(alldata['year']))

                    outputdataset = pd.DataFrame(columns=['NUTS', 'indicator'] + years)
                    NUTSregions = np.array(alldata.columns[9:])
                    droughtdrivers = [indicator + str(ACC) + s for s in ['_jan', '_feb', '_mar', '_apr', '_may', '_jun', '_jul', '_aug', '_sep', '_oct', '_nov', '_dec', '_mean', '_min', '_duration']]              
                    outputdataset['indicator'] = np.tile(droughtdrivers, len(NUTSregions))
                    outputdataset['NUTS'] = np.sort(np.tile(NUTSregions, len(droughtdrivers)))  
                        
                    # calculate annual aggregates
                    if inputfile[0] == "S":
                        thr = 0  # for standardized deficits
                    elif inputfile[0] == "N":
                        thr = 100  # for normalised ones
                                
                    for NUTS in NUTSregions:
                        
                        print(NUTS)

                        for y, year in enumerate(years):
                            
                            # print(year)

                            # for yearly analysis
                            data = alldata.loc[alldata['year'] == year][NUTS].values         
                            # print(data)

                            if (inputfile[:3] == "NEI") or (inputfile[:3] == "SEI"):
                                dur = (data >= thr).sum()  # 
                            else: 
                                dur = (data <= thr).sum()  # for t and evt   

                            # calculate yearly indices
                            outputdataset.loc[(outputdataset['NUTS'] == NUTS) & (outputdataset['indicator'] == indicator + str(ACC) + '_mean'), [year] ] = np.nanmean(data)
                            outputdataset.loc[(outputdataset['NUTS'] == NUTS) & (outputdataset['indicator'] == indicator + str(ACC) + 'duration'), [year]] = dur
                            outputdataset.loc[(outputdataset['NUTS'] == NUTS) & (outputdataset['indicator'] == indicator + str(ACC) + '_min'), [year] ] = np.nanmin(data)

                            # save individual months
                            outputdataset.loc[(outputdataset['NUTS'] == NUTS) & (outputdataset['indicator'] == indicator + str(ACC) + '_jan'), [year] ] = data[0]  # .loc[data['year']== 1]
                            outputdataset.loc[(outputdataset['NUTS'] == NUTS) & (outputdataset['indicator'] == indicator + str(ACC) + '_feb'), [year] ] = data[1]  # .loc[data['year']== 2]
                            outputdataset.loc[(outputdataset['NUTS'] == NUTS) & (outputdataset['indicator'] == indicator + str(ACC) + '_mar'), [year] ] = data[2]  # .loc[data['year']== 3]
                            outputdataset.loc[(outputdataset['NUTS'] == NUTS) & (outputdataset['indicator'] == indicator + str(ACC) + '_apr'), [year] ] = data[3]  # .loc[data['year']== 4]
                            outputdataset.loc[(outputdataset['NUTS'] == NUTS) & (outputdataset['indicator'] == indicator + str(ACC) + '_may'), [year] ] = data[4]  # .loc[data['year']== 5]
                            outputdataset.loc[(outputdataset['NUTS'] == NUTS) & (outputdataset['indicator'] == indicator + str(ACC) + '_jun'), [year] ] = data[5]  # .loc[data['year']== 6]
                            outputdataset.loc[(outputdataset['NUTS'] == NUTS) & (outputdataset['indicator'] == indicator + str(ACC) + '_jul'), [year] ] = data[6]  # .loc[data['year']== 7]
                            outputdataset.loc[(outputdataset['NUTS'] == NUTS) & (outputdataset['indicator'] == indicator + str(ACC) + '_aug'), [year] ] = data[7]  # .loc[data['year']== 8]
                            outputdataset.loc[(outputdataset['NUTS'] == NUTS) & (outputdataset['indicator'] == indicator + str(ACC) + '_sep'), [year] ] = data[8]  # .loc[data['year']== 9]
                            outputdataset.loc[(outputdataset['NUTS'] == NUTS) & (outputdataset['indicator'] == indicator + str(ACC) + '_oct'), [year] ] = data[9]  # .loc[data['year']== 10]
                            outputdataset.loc[(outputdataset['NUTS'] == NUTS) & (outputdataset['indicator'] == indicator + str(ACC) + '_nov'), [year] ] = data[10]  # .loc[data['year']== 11]
                            outputdataset.loc[(outputdataset['NUTS'] == NUTS) & (outputdataset['indicator'] == indicator + str(ACC) + '_dec'), [year] ] = data[11]  # .loc[data['year']== 12]

                print(outputdataset.head())
                outputdataset.to_csv(os.path.join(outdir, RCM + "_" + SCE + "_" + WL + "_" + indicator + str(ACC) + "_yearly_hazard_file.csv"), index=False)
                outputdataset = []

# In[ ]:

inputfile

# In[ ]:

