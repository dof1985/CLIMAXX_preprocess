'''
Created on Aug 3, 2023

@author: politti
'''
import os
import pandas as pd
from loguru import logger
import geopandas as gpd
from glob import glob
import json
from rasterstats import zonal_stats
from datetime import datetime
import xarray as xr
import rasterio
import numpy as np
from shapely.geometry import mapping
import rioxarray as rxr


def dump_json_to_file(pydict, out_file, indent=4):
    try:
        with open(out_file, 'w') as fp:
            json.dump(pydict, fp, indent=indent)
        return 0
    except:
        raise

    
def get_config(json_file):

    with open(json_file) as json_config:
        config = json_config.read()
      
    config = json.loads(config)     
    return config



def clip_raster_xarray(raster_path, crop_extent, out_raster):
    '''
        Clips a raster by the extent of a geodataframe and saves the newly created raster to a file.
        Return the geodataframe.
    '''
    
    if(isinstance(crop_extent, str)):
        crop_extent = gpd.read_file(crop_extent)
    
    src = rxr.open_rasterio(raster_path, masked=True).squeeze()
        
    src_clipped = src.rio.clip(crop_extent.geometry.apply(mapping), crop_extent.crs)
    src_clipped.rio.to_raster(out_raster)
    
    logger.info(f'{out_raster} saved.')
    
    return crop_extent

def get_netcdf_var(nc_file, var_name, time_var='time', lat_var='lat', lon_var='lon', \
                               crs_string='epsg:4326', out_dir='../data', multiplier=1, from_date=None, to_date=None, resample_step=None, time_format='%Y-%m-%d %H:%M:%S'):
    
    '''
    Reads all time steps of a variable in a netcdf file and creates a raster for each time step.
    Returns a dictionary whose keys are the dates and values are the file paths of the rasters.
    '''
    outputs = {}
    ds = nc_file
    if(isinstance(nc_file, str)):
        ds = xr.open_dataset(nc_file, decode_times=True)
    crs = rasterio.crs.CRS({"init": crs_string})
    
    ds_m = ds
    if(resample_step is not None): 
        ds_m = ds.resample(time=resample_step).mean() 
    times = ds_m[time_var].values
    
    if(from_date is None):
        from_date = times[0]
    if(to_date is None):
        to_date = times[-1]   
        
    if(isinstance(from_date, str)):
        from_date = np.datetime64(from_date)  # datetime.strptime(from_date.strip(), '%Y-%m-%d')
    if(isinstance(to_date, str)):
        to_date = np.datetime64(to_date)  # datetime.strptime(to_date.strip(), '%Y-%m-%d')
    
    # Get the latitude and longitude values
    lats = ds_m[lat_var].values
    lons = ds_m[lon_var].values
    
    # Loop over each time step and convert it to a raster file
    for t in range(ds_m.time.size):
        # Get the data for this time step
        data = ds_m[var_name][t,:,:].values
        data = data * multiplier
        t_step = times[t]
        logger.info(f'Reading {t_step} of {var_name} variable.')   
        
        if(t_step >= from_date and t_step <= to_date):
            if(isinstance(t_step, np.float32)):
                t_step = str(int(t_step))
                t_step = f'{t_step}-01-01 01:00:00'
            # datetime_step = datetime.strptime(str(t_step).split('T')[0], '%Y-%m-%d')
            if('T' in str(t_step)):
                t_step = str(t_step).split('.')[0].replace('T', ' ')
            
            datetime_step = datetime.strptime(str(t_step), time_format)
            month = str(datetime_step.month)
            
            if(len(month) == 1):
                month = '0' + month
            filename = f'{datetime_step.year}-{month}_{var_name}.tif'
            filename = os.path.join(out_dir, filename)
            outputs[f'{datetime_step.year}-{month}'] = filename
                    
            # Create a raster file using rasterio
            with rasterio.open(
                filename,
                'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                crs=crs,
                transform=rasterio.transform.from_bounds(lons.min(), lats.min(), lons.max(), lats.max(), data.shape[1], data.shape[0])
            ) as dst:
                dst.write(data, 1)
            
    return outputs, ds


def raster_stats_by_feature(raster_path, geo_df):
    '''
    Compute summary statistics of a raster for each feature of a geodataframe. The statistics are
    added to the geodataframe.
    '''
    if(isinstance(geo_df, gpd.GeoDataFrame) == False):
        geo_df = gpd.read_file(geo_df)
        
    with rasterio.open(raster_path) as src:
        affine = src.transform
        array = src.read(1)
        df_zonal_stats = pd.DataFrame(zonal_stats(geo_df,
                                                  array,
                                                  affine=affine,
                                                  # stats=['min', 'max', 'median', 'majority', 'sum']
                                                  )
        )
    
    # adding statistics back to original GeoDataFrame
    gdf2 = pd.concat([geo_df, df_zonal_stats], axis=1)     

    return gdf2


def transpose_dataframe(gdf, agg_field='NUTS_ID', drop_cols=None):
    df = pd.DataFrame(gdf)
    
    if(drop_cols is not None):
        cols = list(df.columns)
        d_cols = [col for col in drop_cols if col in cols]
        
        if(agg_field in d_cols):
            d_cols.remove(agg_field)
            
        df.drop(columns=d_cols, inplace=True)
        
    df.rename(columns={agg_field: 'Date'}, inplace=True)
    
    df = df.T
    new_header = df.iloc[0] 
    df = df[1:]
    df.columns = new_header 
    
    return df


def var_nuts_mean(var_dict, var_name, nuts_shape,  out_file, agg_field, drop_cols=['CNTR_CODE', 'LEVL_CODE', 'NUTS2_ID', 'Shape_Leng', 'Shape_Area', 'geometry']):
    
    logger.info(f'Computing  {agg_field} polygons mean for {var_name}')

    means_shape = out_file.replace('.csv', '.shp')

    
    for date_k, rf in var_dict.items():
        logger.info(f'{var_name} {date_k}')
        nuts_shape = raster_stats_by_feature(raster_path=rf, geo_df=nuts_shape)
        # date_k = os.path.basename(rf).split('_')[0]
        nuts_shape.rename(columns={'mean': date_k}, inplace=True)
        nuts_shape.drop(columns=['min', 'max', 'count'], inplace=True)
    
    nuts_shape.to_file(means_shape)
    logger.info(f'{means_shape} saved.')
    
    df = transpose_dataframe(gdf=nuts_shape, agg_field=agg_field, drop_cols=drop_cols)
    df.to_csv(out_file, index=True)
    logger.info(f'{out_file} saved.')
    
    return df


def clip_var(var_dict, crop_extent, out_dir, out_json):
    
    var_clip_dict = {}
       
    for date_k, tif in var_dict.items(): 
        logger.info(f'Clipping {os.path.basename(tif)}...')
        out_raster = os.path.basename(tif).replace('.tif', '_clip.tif')
        out_raster = os.path.join(out_dir, out_raster)
        clip_raster_xarray(raster_path=tif, crop_extent=crop_extent, out_raster=out_raster)
        var_clip_dict[date_k] = out_raster
    
    if(out_json is not None):
        dump_json_to_file(pydict=var_clip_dict, out_file=out_json)        
        logger.info(f'{os.path.basename(out_json)} saved.')
        
    return var_clip_dict


def get_var_name(vars_dict, file_name):
    var_name = None
    for k, v in vars_dict.items():
        if k in file_name:
            var_name = v
            break
    return var_name


def cleanup(files_dict):
    for _k, v in files_dict.items():
        os.remove(v)
        
    return 0


def main(config_file):
    config = get_config(config_file)
    nuts_shape = config['nuts_shape'] 
    nc_directory = config['nc_directory']
    vars_dict = config['vars_dict']
    agg_field = config['aggregation_field']
    drop_cols = config['shapefile_drop_cols']
    
    os.chdir(nc_directory)
    ncs = glob('**/*.nc')
    if(len(ncs)==0):
        ncs = glob('*.nc')
        
    logger.info(f'Found {len(ncs)} files.')
    
    if(len(ncs)>0):
        shape = gpd.read_file(nuts_shape)
        out_dir = './data/temp'
        
        if(os.path.isdir(out_dir) == False):
            os.makedirs(out_dir, exist_ok=True)
        
        for nc in ncs:
            nc_var = get_var_name(vars_dict, nc)
            if(nc_var is not None):
                logger.info(f'Processing {nc}...')
                
                
                #converst each time step of the netcdf into a raster named after the variable and the date
                var_outputs, _nc_ds = get_netcdf_var(nc_file =nc,
                                                       var_name = nc_var,
                                                       time_var='time',
                                                        lat_var='lat',
                                                        lon_var='lon',
                                                        crs_string='epsg:4326',
                                                        out_dir = out_dir,
                                                        multiplier=1,
                                                        from_date=None,
                                                        to_date=None
                                                    )
                
                #clip all rasters by the study area shapefile extent
                var_clip_dict = clip_var(var_dict = var_outputs, 
                                         crop_extent = shape, 
                                         out_dir = out_dir, 
                                         out_json = None)
                
                out_file = nc.replace('.nc', '.csv')
                out_file = os.path.basename(out_file)
                out_file = os.path.join(nc_directory, out_file)
                _df = var_nuts_mean(var_dict = var_clip_dict, 
                                    var_name = nc_var, 
                                    nuts_shape = shape, 
                                    out_file = out_file, 
                                    agg_field = agg_field, 
                                    drop_cols = drop_cols
                                )
                logger.info(f'Cleaning up raster intermediate files for variable {nc_var}')
                cleanup(var_outputs)
                cleanup(var_clip_dict)
        os.removedirs(out_dir)
        

if __name__ == '__main__':
    
    jsons = glob('../data/base/*.json')
    
    if(len(jsons) > 0):
        print(f'Found {len(jsons)} json files:')
        js_dict = {}
        for i, js in enumerate(jsons):
            print(i, js)
            js_dict[str(i)] = js
    
    inp = input('Enter the path to the config file or the number corresponding to the above json: ')
    
    if(inp in js_dict.keys()):
        config_file = js_dict[inp]
    
    elif(os.path.isfile(inp)):
        config_file = inp
    elif(len(inp) != 0):
        print('Choice not recognized. Program will exit.')
        exit(-1)
    
    main(config_file)    
    
    
    main(config_file)
    
    print('Process completed')









