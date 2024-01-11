'''
Created on May 5, 2023

@author: politti
'''

# import pyproj
import os
from datetime import datetime
import xarray as xr
import rasterio
import rasterio as rio
import numpy as np
from loguru import logger
from shapely.geometry import mapping
import rioxarray as rxr
import geopandas as gpd
from rasterstats import zonal_stats
import pandas as pd


def transpose_dataframe(gdf, agg_field='NUTS_ID', drop_cols=['CNTR_CODE', 'LEVL_CODE', 'NUTS2_ID', 'Shape_Leng', 'Shape_Area', 'geometry']):
    df = pd.DataFrame(gdf)
    
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


def read_raster(raster_file):
    with rio.open(raster_file) as src:
        return (src.read(1))

    
def avg_rasters(raster_files, out_raster):
    '''
    Computes the average value of a list of rasters covering the same extent.
    '''
    array_list = [read_raster(x) for x in raster_files]
    array_mean = np.mean(array_list, axis=0)
    
    with rio.open(raster_files[0]) as src:
        meta = src.meta
        
    with rio.open(out_raster, 'w', **meta) as dst:
        dst.write(array_mean.astype(rio.float32), 1)
        
    logger.info(f'{out_raster} saved.')
        
    return 0


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


def get_monthly_netcdf_var(nc_file, var_name, time_var='time', lat_var='lat', lon_var='lon', \
                               crs_string='epsg:4326', out_dir='../data', multiplier=1, from_date=None, to_date=None):
    
    '''
    Reads all the time steps of a netcdf variable and calculates the monthly average, then saves
    a raster for each year-month
    Returns a dictionary with year-month as keys and the raster path as value and the netcdf dataset
    '''
    outputs, ds = get_netcdf_var(nc_file,
                                 var_name,
                                 time_var,
                                 lat_var,
                                 lon_var,
                                 crs_string,
                                 out_dir,
                                 multiplier,
                                 from_date,
                                 to_date,
                                 resample_step='1MS')
            
    return outputs, ds


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


def read_netcdf_single_t_step_var(nc_file, var_name, out_dir, time_var='time', lat_var='lat', lon_var='lon', crs_string='epsg:4326'):
    
    ds = xr.open_dataset(nc_file, decode_times=True)
    crs = rasterio.crs.CRS({"init": crs_string})
        
    time = ds[time_var].values[0]
    date_items = str(time).split('-')
    year = date_items[0]
    month = date_items[1]
    date_k = f'{year}-{month}'
    
    # Get the latitude and longitude values
    lats = ds[lat_var].values
    lons = ds[lon_var].values
    
    data = ds[var_name][0,:,:].values
    logger.info(f'Reading {time} of {var_name} variable.')   
        
    filename = f'{year}-{month}_{var_name}.tif'
    filename = os.path.join(out_dir, filename)
    
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
            
    return filename, date_k


def nuts_mean_raster(raster_path, crop_extent, raster_clip_path, nuts_shape, out_shp, var_name):
    
    _crop_extent = clip_raster_xarray(raster_path=raster_path,
                                     crop_extent=crop_extent,
                                     out_raster=raster_clip_path
                                    )

    gdf = raster_stats_by_feature(raster_path=raster_clip_path, geo_df=nuts_shape)

    gdf.rename(columns={'mean': var_name}, inplace=True)
    gdf.drop(columns=['min', 'max', 'count'], inplace=True)
    
    gdf.to_file(out_shp)
    logger.info(f'{out_shp} saved.')
    
    return gdf

