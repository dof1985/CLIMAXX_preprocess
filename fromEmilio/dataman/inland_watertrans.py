'''
Created on May 11, 2023

@author: politti
'''
import pandas as pd
import geopandas as gpd
from utils.utils import get_country_or_code
import numpy as np
from loguru import logger


def inland_water_transport_nuts_mean(down_data, out_file, nuts_shape):
    
    logger.info('Extracting inland water transportation...')

    gdf = gpd.read_file(nuts_shape)
    df = pd.read_csv(down_data)
    
    years = list(df.columns)[1:]
    for y in years:
        gdf[y] = 0
            
    countries = list(df['country'].unique())
    
    for c in countries:
        c_code = get_country_or_code(c)
        for y in years:
            mask = gdf['CNTR_CODE'] == c_code
            val = df[df['country'] == c][y].values[0]
            gdf[y] = np.where(mask, val, gdf[y])
            
    out_shape = out_file.replace('.csv', '.shp')
    gdf.to_file(out_shape)
    # print(gdf.columns)
    
    logger.info('Inland water transportation extracted.')
    return gdf


if __name__ == '__main__':
    down_data = '../data/inland_water_transport/download/inland_water_transport.csv'
    out_file = '../data/inland_water_transport/nuts_means/inland_water_transport_nuts3_means.csv'
    nuts_shape = '../data/shapes/west_balkans_ukr_md_level_3.shp'
    
    agg_field = 'NUTS_ID'
    drop_cols = ['CNTR_CODE', 'LEVL_CODE', 'NUTS2_ID', 'Shape_Leng', 'Shape_Area', 'geometry']
    
    inland_water_transport_nuts_mean(down_data, out_file, nuts_shape, agg_field=agg_field, drop_cols=drop_cols)
    
    print('Process completed')
    
    
