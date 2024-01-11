'''
Created on May 8, 2023

@author: politti
'''

from utils.spatial import raster_stats_by_feature, nuts_mean_raster, transpose_dataframe
from utils.utils import get_config, dump_json_to_file, remove_duplicate_columns, ensure_states
import os
import pandas as pd
from loguru import logger
import sqlite3
from dataman.inland_watertrans import inland_water_transport_nuts_mean
from dataman.aridity import nuts_mean_aridity
import geopandas as gpd
import numpy as np
from dataman.gdp_capita import gdp_per_capita
from dataman.crops import extra_clip_crops
from dataman.npp import npp_export_clip_nutmean
from glob import glob


def check_csvs_headers(nut_level=2):
    ws = f'../data/base/csv_data_nuts_{nut_level}'
    os.chdir(ws)
    csvs = glob('*.csv')
    for c in csvs:
        print(c)
        cols = list(pd.read_csv(c).columns)
        print(cols)
        print()
    return 0


def var_nuts_mean(var_dict_file, var_name, nuts_shape, out_dir, agg_field, con, drop_cols=['CNTR_CODE', 'LEVL_CODE', 'NUTS2_ID', 'Shape_Leng', 'Shape_Area', 'geometry']):
    
    logger.info(f'Computing  {agg_field} polygons mean for {var_name}')
    if(isinstance(var_dict_file, dict) == False):
        var_dict = get_config(var_dict_file)
    else:
        var_dict = var_dict_file
    table_name = f'{var_name}_{agg_field}_means'
    means_shape = os.path.join(out_dir, table_name + '.shp')
    means_csv = means_shape.replace('.shp', '.csv')
    
    for date_k, rf in var_dict.items():
        logger.info(f'{var_name} {date_k}')
        nuts_shape = raster_stats_by_feature(raster_path=rf, geo_df=nuts_shape)
        # date_k = os.path.basename(rf).split('_')[0]
        nuts_shape.rename(columns={'mean': date_k}, inplace=True)
        nuts_shape.drop(columns=['min', 'max', 'count'], inplace=True)
    
    nuts_shape.to_file(means_shape)
    logger.info(f'{means_shape} saved.')
    
    df = transpose_dataframe(gdf=nuts_shape, agg_field=agg_field, drop_cols=drop_cols)
    df.to_csv(means_csv, index=True)
    logger.info(f'{means_csv} saved.')
    
    # df.to_sql(name = 't_'+table_name, con=con, if_exists = 'replace', index = True)
    
    return df
        
    
def get_nuts_mean_config():
    nut_level = 2
    config_file = f'../data/nuts_{nut_level}_means_config.json'
    config = {}
    config['nut_level'] = nut_level
    # config['nuts_shape'] =  '../data/shapes/west_balkans_ukr_md_level_3.shp'
    config['nuts_shape'] = f'../data/shapes/west_balkans_ukr_md_level_{nut_level}.shp'
    config['aggregate_level'] = f'NUTS{nut_level}_ID'
    # config['drop_cols'] = ['CNTR_CODE', 'LEVL_CODE', 'NUTS2_ID','Shape_Leng', 'Shape_Area', 'geometry']
    # config['drop_cols'] = ['CNTR_CODE', 'LEVL_CODE', 'NUTS3_ID','Shape_Leng', 'Shape_Area', 'geometry']
    # config['drop_cols'] = ['CNTR_CODE', 'geometry']
    # config['hazards_dir'] = '../data/vars_shapes/hazards'
    config['wbalkan_uk_md_db'] = '../data/wbalkan_uk_md.db'
    config['crop_extent'] = '../data/shapes/west_balkans_ukr_md_shape.shp'
    
    hazards = {}
    pev = {}
    pev['var_name'] = 'pev'
    pev['var_mean_files'] = '../data/ERA5/pev_monthly_means_files.json'
    pev['run_mean'] = False
    pev['nuts_mean_dir'] = f'../data/ERA5/pev_nuts{nut_level}_means'

    t2m = {}
    t2m['var_name'] = 't2m'
    t2m['var_mean_files'] = '../data/ERA5/t2m_monthly_means_files.json'
    t2m['run_mean'] = False
    t2m['nuts_mean_dir'] = f'../data/ERA5/t2m_nuts{nut_level}_means'
    
    tp = {}
    tp['var_name'] = 'tp'
    tp['var_mean_files'] = '../data/ERA5/tp_monthly_means_files.json'   
    tp['run_mean'] = False
    tp['nuts_mean_dir'] = f'../data/ERA5/tp_nuts{nut_level}_means'
    
    wet_fpan = {}
    wet_fpan['var_name'] = 'wet_fpan'
    wet_fpan['var_mean_files'] = '../data/wetland_fapan/fapan_clipped_rasters.json'   
    wet_fpan['run_mean'] = False
    wet_fpan['nuts_mean_dir'] = f'../data/wetland_fapan/nuts{nut_level}_means'
    
    forest_lai = {}  # done
    forest_lai['var_name'] = 'forest_lai'
    forest_lai['var_mean_files'] = '../data/forest_lai/Lai_500m_clipped_rasters.json'   
    forest_lai['run_mean'] = False
    forest_lai['nuts_mean_dir'] = f'../data/forest_lai/nuts{nut_level}_means'
    
    soil_moist = {}
    soil_moist['var_name'] = 'soil_moist'
    soil_moist['var_mean_files'] = '../data/soil_moisture_index/soil_moist_index_month_means.json'   
    soil_moist['run_mean'] = False
    soil_moist['nuts_mean_dir'] = f'../data/soil_moisture_index/nuts{nut_level}_means_month'
    
    hazards['pev'] = pev
    hazards['t2m'] = t2m
    hazards['tp'] = tp
    hazards['wet_fpan'] = wet_fpan
    hazards['forest_lai'] = forest_lai
    hazards['soil_moist'] = soil_moist
    
    config['hazards'] = hazards
    
    vulnerabilities = {}
    forest = {}
    forest['run_mean'] = False
    forest['raster_files'] = {'2001': '../data/EU_TreeMap/DominantSpecies.tif'}
    
    vulnerabilities['forest'] = forest
    config['vulnerabilities'] = vulnerabilities
    
    water_transport = {}
    water_transport['down_data'] = '../data/inland_water_transport/download/inland_water_transport.csv'
    water_transport['out_file'] = f'../data/inland_water_transport/nuts_means/inland_water_transport_nuts{nut_level}_means.csv'
    water_transport['run_mean'] = False
    
    config['water_transport'] = water_transport
    
    # aridity
    aridity = {}
    aridity['arid_raster_path'] = '../data/aridity/downloads/aridity.tif'
    aridity['arid_raster_clip_path'] = '../data/aridity/clipped_rasters/aridity_clip.tif'
    aridity['aridity_csv'] = f'../data/aridity/nuts_means/aridity_nuts{nut_level}_means.csv'
    aridity['run_mean'] = False
    
    config['aridity'] = aridity
    
    # ecoregions
    ecoregions = {}
    config['ecoregions'] = ecoregions
    ecoregions['run_eco'] = False
    ecoregions['ecoregions_nuts'] = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/ecoregions/nuts_means/wes_balkan_ukr_md_level{nut_level}_ecoregions.shp'
    ecoregions['field'] = 'ECO_CODE'
    
    # tree species
    tree_spp = {}
    config['tree_spp'] = tree_spp
    tree_spp['run_tree_spp'] = False
    tree_spp['tree_spp_nuts'] = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/EU_TreeMap/nuts_means/tree_sp_nuts{nut_level}.shp'
    
    # agroecological zones
    aez = {}
    config['aez'] = aez
    aez['run_aez'] = False
    aez['aez_nuts'] = f'../data/agro_eco_zones/nuts_means/agroeco_zones_nuts{nut_level}.shp'
    aez['field'] = 'aez_code'
    
    # hydropower
    hydropower = {}
    config['hydropower'] = hydropower
    hydropower['run_hydropower'] = False
    hydropower['hydro_data'] = '../data/hydropower/downloads/IEA_hydro.csv'
    hydropower['hydro_shape'] = f'../data/hydropower/nuts_means/hydropower_nuts{nut_level}.shp'
    
    # gdp
    # https://datadryad.org/stash/dataset/doi:10.5061%2Fdryad.dk1j0 
    gdp = {}
    config['gdp'] = gdp
    gdp['run_gdp'] = False
    gdp['nc_file'] = '/home/politti/git/wbalkan_drought_assess/wbalkan/data/gdp/downloads/GDP_PPP_1990_2015_5arcmin_v2.nc'
    gdp['var'] = 'GDP_PPP'
    gdp['var_dir'] = '../data/gdp/rasters'
    gdp['clip_dir'] = '../data/gdp/rasters_clip'
    gdp['gdp_shape'] = f'../data/gdp/nuts_means/gdp_ppp_nuts{nut_level}.shp'
    
    # crops
    crops = {}
    config['crops'] = crops
    crops['run_crops'] = False
    crops['out_rasters_dir'] = '../data/crops/rasters'
    crops['nc_downloads'] = '../data/crops/downloads'
    crops['out_dir_clips'] = '../data/crops/clip_rasters'
    nc_files = ['wheat_yieldsTS.nc', 'maize_yieldsTS.nc', 'soybean_yieldsTS.nc', 'rice_yieldsTS.nc']
    nc_files = [(os.path.join(crops['nc_downloads'], ncf)) for ncf in nc_files]
    crops['nc_vars'] = ['wheat_yield', 'maize_yield', 'soybean_yield', 'rice_yield']
    crops['nc_files_map'] = dict(zip(crops['nc_vars'], nc_files))
    crops['nuts_mean_dir'] = f'../data/crops/nuts_means/'
    
    # npp
    npp = {}
    config['npp'] = npp
    npp['run_npp'] = False
    npp['var_name'] = 'NPP'
    npp['nc_file'] = '/home/politti/git/wbalkan_drought_assess/wbalkan/data/npp/download/EXP_NPPWetland03.nc'
    npp['rasters_dir'] = '../data/npp/rasters'
    npp['clip_rasters_dir'] = '../data/npp/clip_rasters'
    npp['nuts_mean_dir'] = '../data/npp/nuts_means'
    npp['npp_shape'] = f'../data/npp/nuts_means/npp_nuts{nut_level}.shp'
    
    # npp_forest
    npp_forest = {}
    config['npp_forest'] = npp_forest
    npp_forest['run_npp'] = True
    npp_forest['var_name'] = 'NPPForest'
    npp_forest['nc_file'] = '/home/politti/git/wbalkan_drought_assess/wbalkan/data/npp_forest/download/EXP_NPPOverForest.nc'
    npp_forest['rasters_dir'] = '../data/npp_forest/rasters'
    npp_forest['clip_rasters_dir'] = '../data/npp_forest/clip_rasters'
    npp_forest['nuts_mean_dir'] = '../data/npp_forest/nuts_means'
    npp_forest['npp_shape'] = f'../data/npp_forest/nuts_means/npp_nuts{nut_level}.shp'
    
    # forest
    forest = {}
    config['forest'] = forest
    forest['forest_run'] = False
    forest['nuts_means_shape'] = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/forest_map/nuts_means/forest_majority_code_wbalkans_ukr_nut_{nut_level}.shp'
    forest['nuts_means_csv'] = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/forest_map/nuts_means/forest_majority_code_wbalkans_ukr_nuts{nut_level}.csv'
    forest['for_field'] = 'forest_maj'

    # share irrigation
    shir = {}
    config['shir'] = shir
    shir['run_shir'] = False
    shir['shir_nuts_mean_shp'] = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/share_irrigation/nuts_mean/shareIrr_west_balkans_ukr_md_nuts_mean_{nut_level}.shp'
    shir['shir_nuts_mean_csv'] = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/share_irrigation/nuts_mean/shareIrr_west_balkans_ukr_md_nuts_mean_{nut_level}.csv'
    shir['shi_field'] = 'shareIrr_m'
    
    # share irrigation maize
    shir_maize = {}
    config['shir_maize'] = shir_maize       
    shir_maize['run_shir'] = True
    shir_maize['shir_nuts_mean_shp'] = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/share_irrigation/nuts_mean/MaizShareIrr_west_balkans_ukr_md_nuts_mean_{nut_level}.shp'
    shir_maize['shir_nuts_mean_csv'] = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/share_irrigation/nuts_mean/MaizeShareIrr_west_balkans_ukr_md_nuts_mean_{nut_level}.csv'
    shir_maize['shi_field'] = 'irrMaiz_me'   
    
    # share irrigation wheat
    shir_wheat = {}
    config['shir_wheat'] = shir_wheat
    shir_wheat['run_shir'] = True
    shir_wheat['shir_nuts_mean_shp'] = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/share_irrigation/nuts_mean/WheatShareIrr_west_balkans_ukr_md_nuts_mean_{nut_level}.shp'
    shir_wheat['shir_nuts_mean_csv'] = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/share_irrigation/nuts_mean/WheatShareIrr_west_balkans_ukr_md_nuts_mean_{nut_level}.csv'
    shir_wheat['shi_field'] = 'irrWheat_m' 
    
    # soil compaction
    soic = {}
    config['soic'] = soic
    soic['run_soic'] = False
    soic['soic_nuts_mean_shp'] = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/soil_compact/nuts_mean/soil_compact_west_balkans_ukr_md_nuts_mean_{nut_level}.shp'
    soic['soic_nuts_mean_csv'] = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/soil_compact/nuts_mean/soil_compact_west_balkans_ukr_md_nuts_mean_{nut_level}.csv'
    soic['soic_field'] = 'soil_comp_'
    
    # wgi
    wgi = {}
    config['wgi'] = wgi
    wgi['run_wgi'] = False
    wgi['wgi_nuts_mean_shp'] = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/wgi/nuts_means/wgi_west_balkans_ukr_md_level{nut_level}.shp'
    wgi['wgi_nuts_mean_csv'] = f'/home/politti/git/wbalkan_drought_assess/wbalkan/data/wgi/nuts_means/wgi_west_balkans_ukr_md_level_{nut_level}.csv'
    
    # soil water capacity
    swc = {}
    config['swc'] = swc
    swc['run_swc'] = False
    swc_files = {'0cm': f'../data/soil_water_capacity/nuts_means/water_cap_0cm_west_balkans_ukr_md_nut{nut_level}_{nut_level}.shp',
                 '10cm': f'../data/soil_water_capacity/nuts_means/water_cap_10cm_west_balkans_ukr_md_nut{nut_level}_{nut_level}.shp',
                 '100cm': f'../data/soil_water_capacity/nuts_means/water_cap_100cm_west_balkans_ukr_md_nut{nut_level}_{nut_level}.shp'
        }
    swc['swc_files'] = swc_files
    
    print(config)
    dump_json_to_file(config, config_file)
    return config


def get_drop_cols(shp_file, nutid_field, agg_field=None, keep_fields=[]):
    df = gpd.read_file(shp_file)
    cols = list(df.columns)
    cols.remove(nutid_field)
    if(agg_field is not None):
        cols.remove(agg_field)
    if(len(keep_fields) > 0):
        for f in keep_fields:
            cols.remove(f)
    return cols
    
    
def main():
    config_file = f'../data/base/nations_means_config.json'
    config = get_config(config_file)  # get_nuts_mean_config()
    nuts_shape = config['nuts_shape'] 
    aggregate_level = config['aggregate_level'] 
    # drop_cols = config['drop_cols'] 
    wbalkan_uk_md_db = config['wbalkan_uk_md_db']
    nut_level = config['nut_level']
    
    if('all_states' in config.keys()):
        all_states = config['all_states']
    
    hazards = config['hazards']
    conn = sqlite3.connect(wbalkan_uk_md_db)
    
    haz_drop_cols = get_drop_cols(shp_file=nuts_shape, nutid_field=aggregate_level, agg_field=None)
    for hvar, var_dict in hazards.items():
        
        var_name = var_dict['var_name']
        var_mean_files = var_dict['var_mean_files']
        run_mean = var_dict['run_mean']
        nuts_mean_dir = var_dict['nuts_mean_dir']
        
        if(os.path.isdir(nuts_mean_dir) == False):
            os.mkdir(nuts_mean_dir)
        
        if(run_mean):
            logger.info(f'Processing {hvar}')
            var_nuts_mean(var_dict_file=var_mean_files,
                          var_name=var_name,
                          nuts_shape=nuts_shape,
                          out_dir=nuts_mean_dir,
                          agg_field=aggregate_level,
                          con=conn,
                          drop_cols=haz_drop_cols
                        )
            
    # water transport
    water_transport = config['water_transport']
    wt_run_mean = water_transport['run_mean']
    if(wt_run_mean):
        
        down_data = water_transport['down_data']
        out_file = water_transport['out_file']
    
        wt_gdf = inland_water_transport_nuts_mean(down_data=down_data,
                                             out_file=out_file,
                                             nuts_shape=nuts_shape
                                            )
        wat_drop_cols = get_drop_cols(shp_file=nuts_shape, nutid_field=aggregate_level, agg_field=None)
        tdf = transpose_dataframe(wt_gdf, agg_field=aggregate_level, drop_cols=wat_drop_cols) 
        if(aggregate_level == 'CNTR_CODE'):
            tdf = remove_duplicate_columns(tdf)
            tdf = ensure_states(tdf, all_states)
        tdf.to_csv(out_file)
        logger.info('Inland water transportation transposed and saved.')    
        
    # aridity
    aridity = config['aridity']
    run_aridity = aridity['run_mean']
    if(run_aridity):
    
        arid_raster_path = aridity['arid_raster_path']
        arid_raster_clip_path = aridity['arid_raster_clip_path']
        aridity_csv = aridity['aridity_csv']
        crop_extent = config['crop_extent']
        aridity_shp = aridity_csv.replace('.csv', '.shp')
    
        gdf_arid = nuts_mean_aridity(arid_raster_path=arid_raster_path,
                          crop_extent=crop_extent,
                          arid_raster_clip_path=arid_raster_clip_path,
                          nuts_shape=nuts_shape,
                          aridity_shp=aridity_shp
                          )
        arid_drops = get_drop_cols(shp_file=nuts_shape, nutid_field=aggregate_level, agg_field=None)
        tdf = transpose_dataframe(gdf_arid, agg_field=aggregate_level, drop_cols=arid_drops) 
        tdf.to_csv(aridity_csv)
        logger.info('Aidity transposed and saved.')    
        
    # ecoregions
    ecoregions = config['ecoregions']
    if(ecoregions['run_eco'] == True):
        ecoregions_nuts = ecoregions['ecoregions_nuts']
        ecor_field = ecoregions['field']
        
        gdf_eco = gpd.read_file(ecoregions_nuts)
        eco_csv = ecoregions_nuts.replace('.shp', '.csv')
        
        ecor_drop_cols = list(gdf_eco.columns)
        ecor_drop_cols.remove(ecor_field)
        ecor_drop_cols.remove(aggregate_level)
            
        tdf = transpose_dataframe(gdf_eco, agg_field=aggregate_level, drop_cols=ecor_drop_cols) 
        tdf.to_csv(eco_csv)
        logger.info('Ecorigion transposed and saved.') 
        
    # tree species
    tree_spp = config['tree_spp']
    if(tree_spp['run_tree_spp'] == True):
        tree_spp_nuts = tree_spp['tree_spp_nuts'] 
        
        tree_spp_csv = tree_spp_nuts.replace('.shp', '.csv')
        tsp_gdf = gpd.read_file(tree_spp_nuts)
        treesp_drops = get_drop_cols(shp_file=nuts_shape, nutid_field=aggregate_level, agg_field=None)
        tdf = transpose_dataframe(tsp_gdf, agg_field=aggregate_level, drop_cols=treesp_drops) 
        tdf.to_csv(tree_spp_csv)
        logger.info('Tree species transposed and saved.')  
        
    # agroecological zones
    aez = config['aez']

    if(aez['run_aez'] == True):
        aez_nuts = aez['aez_nuts']
        aez_field = aez['field']
        
        aez_csv = aez_nuts.replace('.shp', '.csv')
        aez_gdf = gpd.read_file(aez_nuts)
        aez_drop_cols = list(aez_gdf.columns)
        aez_drop_cols.remove(aez_field)
        aez_drop_cols.remove(aggregate_level)
        tdf = transpose_dataframe(aez_gdf, agg_field=aggregate_level, drop_cols=aez_drop_cols) 
        tdf.to_csv(aez_csv)
        logger.info('Agroecological zones transposed and saved.')  
    
    # hydropower
    hydropower = config['hydropower']   
    if(hydropower['run_hydropower'] == True):
        hydro_data = hydropower['hydro_data']
        hydro_shape = hydropower['hydro_shape']
        
        hdf = pd.read_csv(hydro_data)
        gdf = gpd.read_file(nuts_shape)
        
        countries = list(hdf.columns)[1:]
        years = list(hdf['year'].unique())
        print(countries)
        print(gdf['CNTR_CODE'].unique())
        
        for y in years:
            gdf[str(y)] = 0
            for c in countries:
                val = hdf[hdf['year'] == y][c].values[0]
                mask = gdf['CNTR_CODE'] == c
                gdf[str(y)] = np.where(mask, val, gdf[str(y)])
        
        gdf.to_file(hydro_shape)
        
        hydro_csv = hydro_shape.replace('.shp', '.csv')
        
        hydro_drops = get_drop_cols(shp_file=nuts_shape, nutid_field=aggregate_level, agg_field=None)
        tdf = transpose_dataframe(gdf, agg_field=aggregate_level, drop_cols=hydro_drops) 
        if(aggregate_level == 'CNTR_CODE'):
            tdf = remove_duplicate_columns(tdf)
            tdf = ensure_states(tdf, all_states)
        tdf.to_csv(hydro_csv)
        
        logger.info('Hydropower data transposed and saved.')  
    
    # gdp
    gdp = config['gdp']
    if(gdp['run_gdp'] == True):
        
        nc_file = gdp['nc_file']
        var = gdp['var']
        var_dir = gdp['var_dir']
        clip_dir = gdp['clip_dir']
        gdp_shape = gdp['gdp_shape']
        crop_extent = config['crop_extent']
        
        gdp_drops = get_drop_cols(shp_file=nuts_shape, nutid_field=aggregate_level, agg_field=None)
        gdp_per_capita(nc_file=nc_file,
                       var=var,
                       var_dir=var_dir,
                       clip_dir=clip_dir,
                       gdp_shape=gdp_shape,
                       crop_extent=crop_extent,
                       nuts_shape=nuts_shape,
                       aggregate_level=aggregate_level,
                       drop_cols=gdp_drops
                    )
        
    # crops
    crops = config['crops']
    if(crops['run_crops'] == True):
        out_rasters_dir = crops['out_rasters_dir']
        out_dir_clips = crops['out_dir_clips']
        nc_vars = crops['nc_vars']
        nc_files_map = crops['nc_files_map']
        crop_extent = config['crop_extent']
        nuts_mean_dir = crops['nuts_mean_dir']
        
        vars_clip_dict = extra_clip_crops(nc_files_map=nc_files_map,
                                          out_rasters_dir=out_rasters_dir,
                                          out_dir_clips=out_dir_clips,
                                          crop_extent=crop_extent)
        
        vars_clip_dict = {'soybean_crop': '../data/crops/clip_rasters/soybean_yield_monthly_means_files_clip.json',
                           'maize_crop': '../data/crops/clip_rasters/maize_yield_monthly_means_files_clip.json',
                           'wheat_crop': '../data/crops/clip_rasters/wheat_yield_monthly_means_files_clip.json',
                           'rice_crop': '../data/crops/clip_rasters/rice_yield_monthly_means_files_clip.json'}
        
        crop_drops = get_drop_cols(shp_file=nuts_shape, nutid_field=aggregate_level, agg_field=None)
        for crop, crop_clip_dict in vars_clip_dict.items():
            logger.info(f'Making nuts mean for {crop}...')
                
            nuts_mean_dir_crop = os.path.join(nuts_mean_dir, 'crop')
            if(os.path.isdir(nuts_mean_dir_crop) == False):
                os.mkdir(nuts_mean_dir_crop)
        
            var_nuts_mean(var_dict_file=crop_clip_dict,
                          var_name=crop,
                          nuts_shape=nuts_shape,
                          out_dir=nuts_mean_dir_crop,
                          agg_field=aggregate_level,
                          con=conn,
                          drop_cols=crop_drops
                        )
            
    # npp
    npp = config['npp']
    if(npp['run_npp'] == True):

        crop_extent = '../data/shapes/west_balkans_ukr_md_shape.shp'
        nuts_shape = config['nuts_shape']    
        nuts_mean_dir = npp['nuts_mean_dir']
        
        npp_clip_dict = npp_export_clip_nutmean(npp=npp, crop_extent=crop_extent)
        npp_drops = get_drop_cols(shp_file=nuts_shape, nutid_field=aggregate_level, agg_field=None)
        
        var_nuts_mean(var_dict_file=npp_clip_dict,
                          var_name='npp',
                          nuts_shape=nuts_shape,
                          out_dir=nuts_mean_dir,
                          agg_field=aggregate_level,
                          con=conn,
                          drop_cols=npp_drops
                        )
        
    # npp_forest
    npp_forest = config['npp_forest']
    if(npp_forest['run_npp'] == True):

        crop_extent = '../data/shapes/west_balkans_ukr_md_shape.shp'
        nuts_shape = config['nuts_shape']    
        nuts_mean_dir = npp['nuts_mean_dir']
        
        npp_clip_dict = npp_export_clip_nutmean(npp=npp_forest, crop_extent=crop_extent)
        npp_drops = get_drop_cols(shp_file=nuts_shape, nutid_field=aggregate_level, agg_field=None)
        
        var_nuts_mean(var_dict_file=npp_clip_dict,
                          var_name='NPPForest',
                          nuts_shape=nuts_shape,
                          out_dir=nuts_mean_dir,
                          agg_field=aggregate_level,
                          con=conn,
                          drop_cols=npp_drops
                        )
        
    # forest
    forest = config['forest']
    if(forest['forest_run'] == True):
        #    Forests codes
        # 1    Evergreen Needleleaf Forests (Tree cover > 60%)
        # 2    Evergreen Broadleaf Forests (Tree cover > 60%)
        # 3    Deciduous Needleleaf Forests (Tree cover > 60%)
        # 4    Deciduous Broadleaf Forests (Tree cover > 60%)
        # 5    Mixed Forests (Tree cover > 60%)
        
        forest_nuts_means_shape = forest['nuts_means_shape']
        nuts_means_csv = forest['nuts_means_csv']
        for_field = forest['for_field']
 
        forest_gdf = gpd.read_file(forest_nuts_means_shape)
        
        for_drop_cols = list(forest_gdf.columns)
        for_drop_cols.remove(for_field)
        for_drop_cols.remove(aggregate_level)        
        
        tdf = transpose_dataframe(forest_gdf, agg_field=aggregate_level, drop_cols=for_drop_cols) 
        tdf.to_csv(nuts_means_csv)
        logger.info('Forest transposed and saved.') 
        
    # share irrigation
    shir = config['shir']
    if(shir['run_shir'] == True):
        
        shir_nuts_mean_shp = shir['shir_nuts_mean_shp']
        shir_nuts_mean_csv = shir['shir_nuts_mean_csv']  
        shi_field = shir['shi_field']
        
        shir_gdf = gpd.read_file(shir_nuts_mean_shp)
        
        sci_drop_cols = list(shir_gdf.columns)
        sci_drop_cols.remove(shi_field)
        sci_drop_cols.remove(aggregate_level) 
        
        shir_tdf = transpose_dataframe(shir_gdf, agg_field=aggregate_level, drop_cols=sci_drop_cols) 
        shir_tdf.to_csv(shir_nuts_mean_csv)
        logger.info('Share irrigation transposed and saved.') 
        
    # share irrigation
    shir_maize = config['shir_maize']
    if(shir_maize['run_shir'] == True):
        
        shir_nuts_mean_shp = shir_maize['shir_nuts_mean_shp']
        shir_nuts_mean_csv = shir_maize['shir_nuts_mean_csv']  
        shi_field = shir_maize['shi_field']
        
        shir_gdf = gpd.read_file(shir_nuts_mean_shp)
        
        sci_drop_cols = list(shir_gdf.columns)
        sci_drop_cols.remove(shi_field)
        sci_drop_cols.remove(aggregate_level) 
        
        shir_tdf = transpose_dataframe(shir_gdf, agg_field=aggregate_level, drop_cols=sci_drop_cols) 
        shir_tdf.to_csv(shir_nuts_mean_csv)
        logger.info('Share maize irrigation transposed and saved.')   
        
    # share irrigation
    shir_wheat = config['shir_wheat']
    if(shir_wheat['run_shir'] == True):
        
        shir_nuts_mean_shp = shir_wheat['shir_nuts_mean_shp']
        shir_nuts_mean_csv = shir_wheat['shir_nuts_mean_csv']  
        shi_field = shir_wheat['shi_field']
        
        shir_gdf = gpd.read_file(shir_nuts_mean_shp)
        
        sci_drop_cols = list(shir_gdf.columns)
        sci_drop_cols.remove(shi_field)
        sci_drop_cols.remove(aggregate_level) 
        
        shir_tdf = transpose_dataframe(shir_gdf, agg_field=aggregate_level, drop_cols=sci_drop_cols) 
        shir_tdf.to_csv(shir_nuts_mean_csv)
        logger.info('Share wheat irrigation transposed and saved.')      
    
    # soil compaction
    soic = config['soic']
    if(soic['run_soic'] == True):
        
        soic_nuts_mean_shp = soic['soic_nuts_mean_shp'] 
        soic_nuts_mean_csv = soic['soic_nuts_mean_csv']
        soic_field = soic['soic_field']
        
        soic_gdf = gpd.read_file(soic_nuts_mean_shp)

        soic_drop_cols = list(soic_gdf.columns)
        soic_drop_cols.remove(soic_field)
        soic_drop_cols.remove(aggregate_level)         
        
        soic_tdf = transpose_dataframe(soic_gdf, agg_field=aggregate_level, drop_cols=soic_drop_cols) 
        soic_tdf.to_csv(soic_nuts_mean_csv)
        logger.info('Soil compaction transposed and saved.')   
    
    # wgi
    wgi = config['wgi']
    if(wgi['run_wgi'] == True):
        
        wgi_nuts_mean_shp = wgi['wgi_nuts_mean_shp'] 
        wgi_nuts_mean_csv = wgi['wgi_nuts_mean_csv']
        
        wgi_gdf = gpd.read_file(wgi_nuts_mean_shp)
        wgi_drops = get_drop_cols(shp_file=nuts_shape, nutid_field=aggregate_level, agg_field=None)
        wgi_tdf = transpose_dataframe(wgi_gdf, agg_field=aggregate_level, drop_cols=wgi_drops) 
        wgi_tdf.to_csv(wgi_nuts_mean_csv)
        logger.info('WGI transposed and saved.')    
        
    # SWC    
    swc = config['swc']
    if(swc['run_swc'] == True):
        swc_files = swc['swc_files']
        swc_drops = get_drop_cols(shp_file=nuts_shape, nutid_field=aggregate_level, agg_field=None)
        
        for k, v in swc_files.items():
            
            logger.info(f'Transposing {k}')
            
            swc_nuts_mean_csv = v.replace(f'nut{nut_level}_{nut_level}.shp', f'nut_{nut_level}.csv')
            swc_gdf = gpd.read_file(v)
            swc_tdf = transpose_dataframe(swc_gdf, agg_field=aggregate_level, drop_cols=swc_drops) 
            swc_tdf.to_csv(swc_nuts_mean_csv)
    
    logger.info('Soil water contents transposed and saved.')              
        
    # for k in config.keys():
    #     print(f'"{k}" = ')      
    #     if(k == 'hazards'):
    #         h = config[k]
    #
    #         for hk in h.keys():
    #             print(f'"{hk}" = ')    
    #
    #     if(k == 'vulnerabilities'):
    #         v = config[k]
    #
    #         for vk in v.keys():
    #             print(f'"{vk}" = ') 
    

if __name__ == '__main__':
    main()
    
    print('Process completed')
