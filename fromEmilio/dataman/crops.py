'''
Created on May 15, 2023

@author: politti
'''
import os
from utils.spatial import get_netcdf_var
from utils.utils import dump_json_to_file
from loguru import logger
from dataman.pev_t2m_tp import clip_vars


def etxtract_vars(nc_files_map, out_dir):
    
    vars_dict = {}
    for var, nc_file in nc_files_map.items():
        logger.info(f'Processing {var}')
        
        var_dir = os.path.join(out_dir, var)
        if(os.path.isdir(var_dir) == False):
            os.mkdir(var_dir)
            
        var_outputs, nc_file = get_netcdf_var(nc_file=nc_file,
                               var_name=var,
                               time_var='time',
                               lat_var='lat',
                               lon_var='lon',
                               crs_string='epsg:4326',
                               out_dir=var_dir,
                               multiplier=1,
                               from_date=None,
                               to_date=None)
        
        out_file = os.path.join(out_dir, f'{var}_monthly_means_raster_files.json')
        dump_json_to_file(pydict=var_outputs, out_file=out_file)
        vars_dict[var] = var_outputs
        
    return vars_dict


def extra_clip_crops(nc_files_map, out_rasters_dir, out_dir_clips, crop_extent):
    
    vars_rasters_dict = etxtract_vars(nc_files_map=nc_files_map, out_dir=out_rasters_dir)
    vars_clip_dict = clip_vars(vars_dict=vars_rasters_dict, crop_extent=crop_extent, out_dir=out_dir_clips)
    
    return vars_clip_dict


if __name__ == '__main__':
    
    out_rasters_dir = '../data/crops/rasters'
    nc_downloads = '../data/crops/downloads'
    out_dir_clips = '../data/crops/clip_rasters'
    nc_files = ['wheat_yieldsTS.nc', 'maize_yieldsTS.nc', 'soybean_yieldsTS.nc', 'rice_yieldsTS.nc']
    nc_files = [(os.path.join(nc_downloads, ncf)) for ncf in nc_files]
    nc_vars = ['wheat_yield', 'maize_yield', 'soybean_yield', 'rice_yield']
    crop_extent = '../data/shapes/west_balkans_ukr_md_shape.shp'
    nc_files_map = dict(zip(nc_vars, nc_files))
    
    vars_rasters_dict = etxtract_vars(nc_vars=nc_files_map, out_dir=out_rasters_dir)
    
    vars_clip_dict = clip_vars(vars_dict=vars_rasters_dict, crop_extent=crop_extent, out_dir=out_dir_clips)
    
    print('Process completed...')
    
    
