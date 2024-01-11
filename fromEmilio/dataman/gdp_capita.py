'''
Created on May 11, 2023

@author: politti
'''

from utils.spatial import get_netcdf_var, clip_raster_xarray, raster_stats_by_feature, transpose_dataframe
from utils.utils import dump_json_to_file
from loguru import logger
import os


def gdp_per_capita(nc_file, var, var_dir, clip_dir, gdp_shape, crop_extent, nuts_shape, aggregate_level, drop_cols):
    
    rasters_map, _nc_file = get_netcdf_var(nc_file=nc_file,
                               var_name=var,
                               time_var='time',
                               lat_var='latitude',
                               lon_var='longitude',
                               crs_string='epsg:4326',
                               out_dir=var_dir,
                               multiplier=1,
                               from_date=None,
                               to_date=None,
                               resample_step=None)
        
    rasters_clip_map = {} 
    for date_k, tif in rasters_map.items():
        
        raster_name = os.path.basename(tif)
        out_raster = os.path.join(clip_dir, raster_name.replace('.tif', '_clip.tif'))
        
        clip_raster_xarray(raster_path=tif, crop_extent=crop_extent, out_raster=out_raster)
        
        rasters_clip_map[date_k] = out_raster
        
    out_file = os.path.join(clip_dir, f'{var}_monthly_means_files_clip.json')
    dump_json_to_file(pydict=rasters_clip_map, out_file=out_file)     

    for date_k, tif in rasters_clip_map.items():
        
        logger.info(f'Extracting GDP for nuts of date: {date_k}')
        nuts_shape = raster_stats_by_feature(raster_path=tif, geo_df=nuts_shape)
        nuts_shape.rename(columns={'mean': date_k}, inplace=True)
        nuts_shape.drop(columns=['min', 'max', 'count'], inplace=True)
    
    nuts_shape.to_file(gdp_shape)
    logger.info(f'{gdp_shape} saved.')    
    
    means_csv = gdp_shape.replace('.shp', '.csv')
    df = transpose_dataframe(gdf=nuts_shape, agg_field=aggregate_level, drop_cols=drop_cols)
    df.to_csv(means_csv, index=True)
    logger.info(f'{means_csv} saved.')
    
    return 0


if __name__ == '__main__':
    
    nc_file = '/home/politti/git/wbalkan_drought_assess/wbalkan/data/gdp/downloads/GDP_PPP_1990_2015_5arcmin_v2.nc'
    var = 'GDP_PPP'
    var_dir = '../data/gdp/rasters'
    clip_dir = '../data/gdp/rasters_clip'
    means_shape = '../data/gdp/nuts_means/gdp_ppp_nuts3.shp'
    
    crop_extent = '../data/shapes/west_balkans_ukr_md_shape.shp'
    nuts_shape = '../data/shapes/west_balkans_ukr_md_level_3.shp'
    aggregate_level = 'NUTS_ID'
    drop_cols = ['CNTR_CODE', 'LEVL_CODE', 'NUTS2_ID', 'Shape_Leng', 'Shape_Area', 'geometry']
    
    gdp_per_capita(nc_file,
                   var,
                   var_dir,
                   clip_dir,
                   means_shape,
                   crop_extent,
                    nuts_shape,
                    aggregate_level,
                    drop_cols
                )
    
    print('Process completed.')
    
    
