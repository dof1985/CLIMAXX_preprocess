'''
Created on May 11, 2023

@author: politti
'''

from utils.spatial import clip_raster_xarray, raster_stats_by_feature
from loguru import logger


def nuts_mean_aridity(arid_raster_path, crop_extent, arid_raster_clip_path, nuts_shape, aridity_shp):
    
    _crop_extent = clip_raster_xarray(raster_path=arid_raster_path,
                                     crop_extent=crop_extent,
                                     out_raster=arid_raster_clip_path
                                    )

    gdf = raster_stats_by_feature(raster_path=arid_raster_clip_path, geo_df=nuts_shape)

    gdf.rename(columns={'mean': 'aridity'}, inplace=True)
    gdf.drop(columns=['min', 'max', 'count'], inplace=True)
    
    # gdf.to_file(aridity_shp)
    logger.info(f'{aridity_shp} saved.')
    
    # tdf = transpose_dataframe(gdf_arid, agg_field = aggregate_level, drop_cols = drop_cols) 
    # tdf.to_csv(aridity_csv)
    # logger.info('Aidity transposed and saved.')   
    
    return gdf


if __name__ == '__main__':
    
    arid_raster_path = '../data/aridity/downloads/aridity.tif'
    crop_extent = '../data/shapes/west_balkans_ukr_md_shape.shp'
    arid_raster_clip_path = '../data/aridity/clipped_rasters/aridity_clip.tif'
    nuts_shape = '../data/shapes/west_balkans_ukr_md_level_3.shp'
    aridity_csv = '../data/aridity/nuts_means/aridity_nuts3_means.csv'
    
    nuts_mean_aridity(arid_raster_path, crop_extent, arid_raster_clip_path, nuts_shape, aridity_csv)
    
    print('Process completed')
