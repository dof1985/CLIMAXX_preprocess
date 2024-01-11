'''
Created on May 8, 2023

@author: politti
'''

from utils.spatial import read_netcdf_single_t_step_var, clip_raster_xarray
import os
from glob import glob
from loguru import logger
from utils.utils import dump_json_to_file, get_config


def extract_netcdfs(down_wat_dir, tot_wat_rasters_dir, wat_var, down_map_file):
    
    cur_dir = os.getcwd()
    
    if(os.path.isdir(tot_wat_rasters_dir) == False):
        os.mkdir(tot_wat_rasters_dir)

    os.chdir(down_wat_dir)
    netcdfs = glob('*.nc4')  
    os.chdir(cur_dir)  
    
    logger.info(f'Found {len(netcdfs)}...')
    downs_dict = {}
    
    for ncf in netcdfs:
        logger.info(f'Processing {ncf}...')
        ncf = os.path.join(down_wat_dir, ncf)

        filename, date_k = read_netcdf_single_t_step_var(nc_file=ncf, var_name=wat_var, out_dir=tot_wat_rasters_dir, time_var='time', lat_var='lat', lon_var='lon', crs_string='epsg:4326')
        downs_dict[date_k] = filename
    
    dump_json_to_file(downs_dict, down_map_file)
    
    return downs_dict


def main():
    
    down_wat_dir = '../data/total_water_storage/downloads'
    tot_wat_rasters_dir = '../data/total_water_storage/rasters'
    tot_wat_rasters_clip_dir = '../data/total_water_storage/clipped'
    # f1 = 'GRD-3_2018152-2018181_GRFO_UTCSR_BA01_0601_LND_v04.nc4'
    wat_var = 'lwe_thickness'
    crop_extent = '../data/shapes/west_balkans_ukr_md_level_3.shp'
    
    down_map_file = '../data/total_water_storage/water_store_downloads_date_rasters.json'
    clip_map_file = '../data/total_water_storage/water_store_clip_date_rasters.json'
        
    downs_dict = None
    
    if(os.path.isfile(down_map_file) == False):
        downs_dict = extract_netcdfs(down_wat_dir=down_wat_dir,
                                     tot_wat_rasters_dir=tot_wat_rasters_dir,
                                     wat_var=wat_var,
                                     down_map_file=down_map_file)
    
    else:
        downs_dict = get_config(down_map_file)
        
    clip_dict = {}
    
    if(os.path.isdir(tot_wat_rasters_clip_dir) == False):
        os.mkdir(tot_wat_rasters_clip_dir)
    
    for date_k, rf in downs_dict.items():
        
        logger.info(f'Clipping {rf}')
        out_raster = os.path.basename(rf)
        out_raster = out_raster.replace('.tif', '_clip.tif')
        out_raster = os.path.join(tot_wat_rasters_clip_dir, out_raster)
        
        crop_extent = clip_raster_xarray(raster_path=rf, crop_extent=crop_extent, out_raster=out_raster)
        
        clip_dict[date_k] = out_raster
        
    dump_json_to_file(clip_dict, clip_map_file)
    
        
if __name__ == '__main__':
    main()
    
    print('Process completed')
    
    
