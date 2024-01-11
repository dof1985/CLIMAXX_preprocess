'''
Created on May 8, 2023

@author: politti
'''

from utils.spatial import get_monthly_netcdf_var, clip_raster_xarray
from utils.utils import dump_json_to_file
from loguru import logger
import os


def etxtract_vars(nc_vars, out_dir, nc_file):
    
    vars_dict = {}
    for var in nc_vars:
        logger.info(f'Processing {var}')
        
        var_dir = os.path.join(out_dir, var)
        if(os.path.isdir(var_dir) == False):
            os.mkdir(var_dir)
            
        var_outputs, nc_file = get_monthly_netcdf_var(nc_file=nc_file,
                               var_name=var,
                               time_var='time',
                               lat_var='latitude',
                               lon_var='longitude',
                               crs_string='epsg:4326',
                               out_dir=var_dir,
                               multiplier=1,
                               from_date=None,
                               to_date=None)
        
        out_file = os.path.join(out_dir, f'{var}_monthly_means_files.json')
        dump_json_to_file(pydict=var_outputs, out_file=out_file)
        vars_dict[var] = var_outputs
        
    return vars_dict
   

def clip_vars(vars_dict, crop_extent, out_dir):
    
    vars_clip_dict = {}
    for var, var_dict in vars_dict.items():
        logger.info(f'Clipping raster of variable: {var}')
        
        var_clip_map = {}
        out_var_dir = os.path.join(out_dir, f'{var}_clip')
        
        if(os.path.isdir(out_var_dir) == False):
            os.mkdir(out_var_dir)
        
        for date_k, tif in var_dict.items():
            raster_name = os.path.basename(tif)
            out_raster = os.path.join(out_var_dir, raster_name.replace('.tif', '_clip.tif'))
            
            clip_raster_xarray(raster_path=tif, crop_extent=crop_extent, out_raster=out_raster)
            
            var_clip_map[date_k] = out_raster
            
        out_file = os.path.join(out_dir, f'{var}_monthly_means_files_clip.json')
        dump_json_to_file(pydict=var_clip_map, out_file=out_file)        
        
        vars_clip_dict[var] = var_clip_map
        
    return vars_clip_dict


if __name__ == '__main__':
    out_dir = '../data/ERA5'
    nc_file = '../data/ERA5/data.nc'
    nc_vars = ['pev', 't2m', 'tp']
    vars_dict = etxtract_vars(nc_vars=nc_vars, out_dir=out_dir, nc_file=nc_file)
    
    crop_extent = '../data/shapes/west_balkans_ukr_md_shape.shp'
    vars_clip_dict = clip_vars(vars_dict, crop_extent, out_dir)
    
    print ('Process completed')
    
    
