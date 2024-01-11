'''
Created on May 9, 2023

@author: politti
'''

from utils.spatial import get_monthly_netcdf_var, clip_raster_xarray
from utils.utils import dump_json_to_file, get_config
from loguru import logger
import os
from glob import glob


def etxtract_fpar_var(nc_var, out_dir, nc_file):
                
    var_outputs, _nc_file = get_monthly_netcdf_var(nc_file=nc_file,
                                                   var_name=nc_var,
                                                   time_var='time',
                                                   lat_var='lat',
                                                   lon_var='lon',
                                                   crs_string='epsg:4326',
                                                   out_dir=out_dir,
                                                   multiplier=1,
                                                   from_date=None,
                                                   to_date=None
                                                )
    return var_outputs


def clip_var(var_dict, crop_extent, out_dir, out_json):
    
    var_clip_dict = {}
       
    for date_k, tif in var_dict.items():
        logger.info(f'Clipping {os.path.basename(tif)}...')
        out_raster = os.path.basename(tif).replace('.tif', '_clip.tif')
        out_raster = os.path.join(out_dir, out_raster)
        clip_raster_xarray(raster_path=tif, crop_extent=crop_extent, out_raster=out_raster)
        var_clip_dict[date_k] = out_raster
    
    dump_json_to_file(pydict=var_clip_dict, out_file=out_json)        
    logger.info(f'{os.path.basename(out_json)} saved.')
        
    return var_clip_dict


if __name__ == '__main__':
    
    rasters_out_dir = '../data/wetland_fapan/rasters'
    clipped_rasters_out_dir = '../data/wetland_fapan/clipped_rasters'
    ncs_dir = '../data/wetland_fapan/downloads'
    nc_var = 'fapan'
    crop_extent = '../data/shapes/west_balkans_ukr_md_shape.shp'
    rasters_dict_file = f'../data/wetland_fapan/{nc_var}_rasters.json'
    clipped_rasters_dict_file = f'../data/wetland_fapan/{nc_var}_clipped_rasters.json'
    
    if(os.path.isfile(rasters_dict_file) == False):
        cur_dir = os.getcwd()
        os.chdir(ncs_dir)
        
        ncs = glob('*.nc')
        ncs = [os.path.join(ncs_dir, nc) for nc in ncs]
        logger.info(f'Found {len(ncs)} files...')
        
        os.chdir(cur_dir)
        
        if(os.path.isdir(rasters_out_dir) == False):
            os.mkdir(rasters_out_dir)
        
        rasters_dict = {}
        for nc in ncs:
            logger.info(f'Converting {nc} to raster...')
            nc_dict = etxtract_fpar_var(nc_var=nc_var, out_dir=rasters_out_dir, nc_file=nc)
            rasters_dict.update(nc_dict)
            
        dump_json_to_file(pydict=rasters_dict, out_file=rasters_dict_file)
        logger.info(f'{rasters_dict_file} saved.')
    
    if(os.path.isfile(clipped_rasters_dict_file) == False):
        
        if(os.path.isdir(clipped_rasters_out_dir) == False):
            os.mkdir(clipped_rasters_out_dir)    
        
        raster_dict = get_config(rasters_dict_file)
    
        vars_clip_dict = clip_var(var_dict=raster_dict,
                                   crop_extent=crop_extent,
                                   out_dir=clipped_rasters_out_dir,
                                   out_json=clipped_rasters_dict_file)
    
    print('Process completed.')
    
    
