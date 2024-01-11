'''
Created on May 5, 2023

@author: politti
'''
import os
from glob import glob
from loguru import logger
import rasterio as rio
import numpy as np
from utils.spatial import avg_rasters
from utils.utils import get_config, dump_json_to_file


def get_tifs_map(out_file='soil_moist_idxs.json', overwrite=False):
    
    tifs_map = None
    if(os.path.exists(out_file) == False):
        overwrite = True
    
    if(overwrite):

        tifs = glob('*.tif')
        logger.info(f'Found {len(tifs)} files.')
        
        tifs_map = {}
        for tif in tifs:
            date = tif.replace('.tif', '').split('idx')[1].split('_')
            year = date[1]
            month = date[2]
            day = date[3]
            
            print(f'Date: {day}-{month}-{year}')
            
            if(year not in tifs_map.keys()):
                tifs_map[year] = {}
            
            tifs_of_year = tifs_map[year]
            
            if(month not in tifs_of_year.keys()):
                tifs_of_year[month] = []
                
            tifs_month = tifs_of_year[month]
            tifs_month.append(tif)
        
        dump_json_to_file(tifs_map, out_file)
    
    else:
        tifs_map = get_config(out_file)
        
    return tifs_map


def make_monthly_mean(out_dir='month_means', out_file='monthly_mean_map.json', overwrite=False):
    avgs_map = None
    if(os.path.exists(out_file) == False):
        overwrite = True
    
    if(overwrite):
        
        if(os.path.isdir(os.path.join(os.getcwd(), out_dir)) == False):
            os.mkdir(out_dir)
    
        tifs_map = get_tifs_map(out_file='soil_moist_idxs.json', overwrite=False)
        avgs_map = {}
        
        years = sorted(list(tifs_map.keys()))
        for year in years:
            avgs_map[year] = {}
            month_map = tifs_map[year]
            months = sorted(list(month_map.keys()))
            for month in months:
                tifs = month_map[month]
                # print(f'{year} {month} {tifs}')
                out_raster = f'{year}_{month}_smoist_idx_wbalkans.tif'
                out_raster = os.path.join(os.getcwd(), out_dir, out_raster)
                avg_rasters(raster_files=tifs, out_raster=out_raster)
                avgs_map[year][month] = os.path.join(os.getcwd(), out_dir, out_raster)
                
        dump_json_to_file(avgs_map, out_file) 
    
    else:
        logger.info(f'{out_file} exist and will not be ovewritten.')
        avgs_map = get_config(out_file)
            
    return avgs_map
            

def year_mean(month_mean_map, out_dir='year_means', overwrite=False, out_file='yearly_mean_map.json'): 
    
    avgs_map = None
    if(os.path.exists(out_file) == False):
        overwrite = True
    
    if(overwrite):
        avgs_map = {}
        if(os.path.isdir(os.path.join(os.getcwd(), out_dir)) == False):
            os.mkdir(out_dir)
            
        years = sorted(list(month_mean_map.keys()))
        
        for year in years:
            months_map = month_mean_map[year]
            
            raster_files = []
            for _month, raster_file in months_map.items():
                raster_files.append(raster_file)        

            out_raster = f'{year}_mean_smoist_idx_wbalkans.tif'
            out_raster = os.path.join(os.getcwd(), out_dir, out_raster)
            avg_rasters(raster_files=raster_files, out_raster=out_raster)
            avgs_map[year] = os.path.join(os.getcwd(), out_dir, out_raster)             
            dump_json_to_file(avgs_map, out_file)    
            
    else:
        logger.info(f'{out_file} exist and will not be ovewritten.')
        avgs_map = get_config(out_file)
        
    return avgs_map
      

if __name__ == '__main__':
    ws = '../data/soil_moisture_index_clip'
    os.chdir(ws)
    mean_map = make_monthly_mean(out_dir='month_means')
    mean_year_map = year_mean(month_mean_map=mean_map, out_dir='year_means', overwrite=False, out_file='yearly_mean_map.json')
    
    print('Process completed')
    
    
