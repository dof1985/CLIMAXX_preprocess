'''
Created on May 15, 2023

@author: politti
'''

import os
from utils.spatial import get_netcdf_var, clip_raster_xarray
from utils.utils import dump_json_to_file
from loguru import logger
from dataman.pev_t2m_tp import clip_vars


def npp_export_clip_nutmean(npp, crop_extent):
    
    if(npp['run_npp'] == True):
        logger.info(f'Processing NPP')
        
        var_name = npp['var_name']
        nc_file = npp['nc_file']
        rasters_dir = npp['rasters_dir']
        clip_rasters_dir = npp['clip_rasters_dir']
        nuts_mean_dir = npp['nuts_mean_dir']
            
        vars_rasters_dict, nc_file = get_netcdf_var(nc_file=nc_file,
                               var_name=var_name,
                               time_var='time',
                               lat_var='lat',
                               lon_var='lon',
                               crs_string='epsg:4326',
                               out_dir=rasters_dir,
                               multiplier=1,
                               from_date=None,
                               to_date=None,
                               resample_step=None)
        
        out_file = os.path.join(rasters_dir, f'{var_name}_monthly_means_raster_files.json')
        dump_json_to_file(pydict=vars_rasters_dict, out_file=out_file)        
        
        npp_clip_dict = {}
        
        for date_k, tif in vars_rasters_dict.items():
            raster_name = os.path.basename(tif)
            out_raster = os.path.join(clip_rasters_dir, raster_name.replace('.tif', '_clip.tif'))
            
            clip_raster_xarray(raster_path=tif, crop_extent=crop_extent, out_raster=out_raster)
            
            npp_clip_dict[date_k] = out_raster
            
        out_file = os.path.join(clip_rasters_dir, f'npp_monthly_means_files_clip.json')
        dump_json_to_file(pydict=npp_clip_dict, out_file=out_file)        
        
        return npp_clip_dict


if __name__ == '__main__':
    
    npp = {}
    npp['run_npp'] = True
    npp['var_name'] = 'NPP'
    npp['nc_file'] = '..data/npp/download/EXP_NPPWetland03.nc'
    npp['rasters_dir'] = '../data/npp/rasters'
    npp['clip_rasters_dir'] = '../data/npp/clip_rasters'
    npp['nuts_mean_dir'] = '../data/npp/nuts_mean'
    crop_extent = '../data/shapes/west_balkans_ukr_md_shape.shp'
    
    print('process completed')
