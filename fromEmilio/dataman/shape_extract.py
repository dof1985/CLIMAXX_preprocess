'''
Created on Jul 21, 2023

@author: politti
'''

from loguru import logger
from utils.spatial import raster_stats_by_feature, transpose_dataframe
from utils.utils import get_config, dump_json_to_file, remove_duplicate_columns, ensure_states
import os
import time
from copy import deepcopy
import geopandas as gpd


def shape_extraction(raster_path, sp_shape, col_name, out_file, agg_field, drop_cols):
    
    sp_shape = raster_stats_by_feature(raster_path=raster_path, geo_df=sp_shape)
    sp_shape.rename(columns={'mean': col_name}, inplace=True)
    sp_shape.drop(columns=['min', 'max', 'count'], inplace=True)
    
    sp_shape.to_file(out_file)
    logger.info(f'{out_file} saved.')
    
    df = transpose_dataframe(gdf=sp_shape, agg_field=agg_field, drop_cols=drop_cols)
    means_csv = out_file.replace('.shp', '.csv')
    df.to_csv(means_csv, index=True)
    logger.info(f'{means_csv} saved.')
    
    return df


def get_drop_cols(cols, keep_cols=[]):
    drop_cols = []
    for c in cols:
        if(c not in keep_cols):
            keep_cols.append(c)
            
    return drop_cols


def main(config_path):
    config = get_config(config_path)
    
    out_dir = config['out_dir']
    def_date = config['default_date']
    agg_field = config['aggregate_field']
    shape = config['shape']
    save_shapes = config['save_shapes']
    
    if(os.path.isdir(out_dir) is False):
        os.mkdir(out_dir)
        
    try:
        var_list = config['vars']
        for var in var_list:
            if(var['run'] is True):
                var_name = var['name']
                logger.info(f'Processing {var_name}')
                f = var['file']
                ext = os.path.basename(f).split('.')[1]
                
                var_files_dict = {}
                if(ext == 'json'):
                    var_files_dict = get_config(f)
                elif(ext == 'tif'):
                    var_files_dict = {def_date: f}
                elif(ext == 'shp'):
                    pass
                else:
                    logger.warning(f'Extension {ext} of file {f} from variable {var_name} not recognized.')
                
                if(len(var_files_dict.keys()) == 0):
                    logger.warning(f'No file found for variable {var_name}.')
                else:
                    
                    table_name = f'{var_name}_{agg_field}_means'
                    means_shape = os.path.join(out_dir, table_name + '.shp')
                    means_csv = means_shape.replace('.shp', '.csv')
                    
                    dfs = []
                    nuts_shape = deepcopy(shape)
                    cols = gpd.read_file(shape)
                    drop_cols = get_drop_cols(cols, keep_cols=[agg_field])
                    for date_k, var_file in var_files_dict.items():
                        
                        logger.info(f'{var_name} {date_k}')
                        nuts_shape = raster_stats_by_feature(raster_path=var_file, geo_df=nuts_shape)
                        nuts_shape.rename(columns={'mean': date_k}, inplace=True)
                        nuts_shape.drop(columns=['min', 'max', 'count'], inplace=True)
                        # print(nuts_shape.columns) 
                    
                    if(save_shapes):
                        nuts_shape.to_file(means_shape)
                        logger.info(f'{means_shape} saved.')
                        
                    df = transpose_dataframe(gdf=nuts_shape, agg_field=agg_field, drop_cols=drop_cols)
                    print(df.columns)                   
                    
                    df.to_csv(means_csv, index=True)
                    logger.info(f'{means_csv} saved.')
        
    except Exception as e:
        logger.error(e)


if __name__ == '__main__':
    
    def_config = '../data/base/romania/basins_spatial_config.json'
    config_file = input('Type the file path of the json config file or press enter to accept the default. ')
    
    if(len(config_file) == 0):
        config_file = def_config
    if(os.path.isfile(config_file) is False):
        logger.error(f'{config_file} does not exists. Application will be terminated in 7 seconds.')
        time.sleep(7)
    else:
        main(config_file)
    
    print('Process completed')

