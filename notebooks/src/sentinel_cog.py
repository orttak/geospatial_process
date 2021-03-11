import folium
import os, shutil
from glob import glob
#%matplotlib inline  
import json
import satsearch
import geopandas as gpd
import numpy as np
import pandas as pd
import warnings
from shapely.geometry import  Polygon
import rioxarray
import satstac

warnings.filterwarnings("ignore")

band_list=['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A',
           'B09', 'B11', 'B12', 'AOT', 'WVP', 'SCL', 'info', 'metadata',
           'visual', 'overview', 'thumbnail']
           
def find_sentinel_tile(target_area,sentinel_tiles_path=''):
    if sentinel_tiles_path:
        #sentinel_tiles_path=data\raw\boundries\sentinel_tr_tiles.shp
        tr_sentinel_tiles=gpd.read_file(sentinel_tiles_path)
        #intersect target area with landsat path-row 
        tiles_intersection = tr_sentinel_tiles[tr_sentinel_tiles.intersects(target_area.geometry[0])]
        tiles_intersection.reset_index(drop=True,inplace=True)
        # Get the center of the map
        xy = np.asarray(target_area.centroid[0].xy).squeeze()
        center = list(xy[::-1])
        zoom = 7
        # Create the most basic OSM folium map
        m = folium.Map(location=center, zoom_start=zoom)
        # Add the bounds GeoDataFrame in red
        m.add_child(folium.GeoJson(target_area.__geo_interface__, name='Area of Study', 
                                style_function=lambda x: {'color': 'red', 'alpha': 0}))
        
        # Iterate through each Polygon of paths and rows intersecting the area
        for i, row in tiles_intersection.iterrows():
            # Create a string for the name containing the path and row of this Polygon
            name = f'ID:{row[0]}' 
            # Create the folium geometry of this Polygon 
            g = folium.GeoJson(row.geometry.__geo_interface__, name=name)
            # Add a folium Popup object with the name string
            g.add_child(folium.Popup(name))
            # Add the object to the map
            g.add_to(m)
        
        return tiles_intersection, m
    else:
        xy = np.asarray(target_area.centroid[0].xy).squeeze()
        center = list(xy[::-1])
        zoom = 7
        # Create the most basic OSM folium map
        m = folium.Map(location=center, zoom_start=zoom)
        # Add the bounds GeoDataFrame in red
        m.add_child(folium.GeoJson(target_area.__geo_interface__, name='Area of Study', 
                                style_function=lambda x: {'color': 'red', 'alpha': 0}))
        return target_area, m

def find_stac_result(target_aoi,date,max_cloud=10):
    URL='https://earth-search.aws.element84.com/v0'
    results = satsearch.Search.search(url=URL,
                                collections=['sentinel-s2-l2a-cogs'],
                                datetime=date,
                                bbox=target_aoi,
                                query={'eo:cloud_cover': {'lt':max_cloud}}, )
    return results

def create_tiles_list(stac_result):
    items = stac_result.items()
    items_json=items.geojson()
    items_json=json.dumps(items_json)
    df=gpd.read_file(items_json)
    df['tile']=df.apply(lambda row: str(row['sentinel:utm_zone'])+row['sentinel:latitude_band']+row['sentinel:grid_square'], axis=1)
    tiles_list = sorted(df['tile'].unique().tolist())
    return tiles_list


def find_sentinel_item(stac_result,tiles_list=[],min_coverage=95,max_cloud=2):
    if tiles_list:
        tile_result_list=[]
        items = stac_result.items()
        for tile in tiles_list:
            #find best image for target tile 
            tile_number=int(tile[0:2])
            lat_band=tile[2]
            grid_sq=tile[3:]
            #create temporary items 
            tmp_items=satstac.ItemCollection(items)
            tmp_items.filter('sentinel:utm_zone',[tile_number,])
            tmp_items.filter('sentinel:latitude_band',[lat_band])
            tmp_items.filter('sentinel:grid_square', [grid_sq])
            try:
                coverage=sorted(tmp_items.properties('sentinel:data_coverage'))
                coverage=[c for c in coverage if c>min_coverage]
                tmp_items.filter('sentinel:data_coverage',coverage)
                if 100 in tmp_items.properties('sentinel:data_coverage'):
                    #get the index of value that data_coverage==100
                    dc_index=[i for i, x in enumerate(tmp_items) if tmp_items[i].properties['sentinel:data_coverage']==100]
                    #get images cloud info
                    filtered_list=[tmp_items[x].properties['eo:cloud_cover'] for i, x in enumerate(dc_index) if tmp_items[x].properties['eo:cloud_cover']<max_cloud]
                    #get first cloud cover after sorting
                    try:
                        # if data exist we use filter method
                        selected_item=sorted(filtered_list)[0]
                        tmp_items.filter('sentinel:data_coverage',[100])
                    except:
                        selected_item=sorted(tmp_items.properties('eo:cloud_cover'))[0]
        
                    tmp_items.filter('eo:cloud_cover', [selected_item])
                    tile_result_list.append(tmp_items[0])

                else:
                    selected_item=sorted(tmp_items.properties('eo:cloud_cover'))[0]
                    #select best image
                    tmp_items.filter('eo:cloud_cover', [selected_item])
                    #get newest image
                    tile_result_list.append(tmp_items[0])
                          
            except:
                # threshold ekle -2 yerine
                coverage=sorted(tmp_items.properties('sentinel:data_coverage'))[-2:]
                tmp_items.filter('sentinel:data_coverage',coverage)
                selected_item=sorted(tmp_items.properties('eo:cloud_cover'))[0]
                #select best image
                tmp_items.filter('eo:cloud_cover', [selected_item])
                #get newest image
                tile_result_list.append(tmp_items[0])
        
        return tile_result_list
    
    else:
        # code define best image for each tile from stac result
        # function return list of tile's information
        tiles_list=create_tiles_list(stac_result)
        #we collect each tile's result
        tile_result_list=[]
        items = stac_result.items()
        
        for tile in tiles_list:
            tile_number=int(tile[0:2])
            lat_band=tile[2]
            grid_sq=tile[3:]
            tmp_items=satstac.ItemCollection(items)
            tmp_items.filter('sentinel:utm_zone',[tile_number,])
            tmp_items.filter('sentinel:latitude_band',[lat_band])
            tmp_items.filter('sentinel:grid_square', [grid_sq])
            try:
                coverage=sorted(tmp_items.properties('sentinel:data_coverage'))
                coverage=[c for c in coverage if c>min_coverage]
                tmp_items.filter('sentinel:data_coverage',coverage)
                if 100 in tmp_items.properties('sentinel:data_coverage'):
                    #get the index of value that data_coverage==100
                    dc_index=[i for i, x in enumerate(tmp_items) if items[i].properties['sentinel:data_coverage']==100]
                    #get images cloud info
                    filtered_list=[tmp_items[x].properties['eo:cloud_cover'] for i, x in enumerate(dc_index) if tmp_items[x].properties['eo:cloud_cover']<max_cloud]
                    #get first cloud cover after sorting
                    try:
                        # if data exist we use filter method
                        selected_item=sorted(filtered_list)[0]
                        tmp_items.filter('sentinel:data_coverage',[100])

                    except:
                        selected_item=sorted(tmp_items.properties('eo:cloud_cover'))[0]
        
                    tmp_items.filter('eo:cloud_cover', [selected_item])
                    tile_result_list.append(tmp_items[0])
                    
                else:
                    selected_item=sorted(tmp_items.properties('eo:cloud_cover'))[0]
                    #select best image
                    tmp_items.filter('eo:cloud_cover', [selected_item])
                    #get newest image
                    tile_result_list.append(tmp_items[0])
                          
            except:
                coverage=sorted(items.properties('sentinel:data_coverage'))[-2:]
                tmp_items.filter('sentinel:data_coverage',coverage)
                selected_item=sorted(tmp_items.properties('eo:cloud_cover'))[0]
                #select best image
                tmp_items.filter('eo:cloud_cover', [selected_item])
                #get newest image
                tile_result_list.append(tmp_items[0])
        
        return tile_result_list

def show_items_list(sentinel_items_list,band_list=band_list):
    # this function return item's band as a list
    # you can use result of find_sentinel_item function
    ######
    #if band list empty show all
    ####
    result_list=[]
    for item in sentinel_items_list:        
        #create tile name for output
        tile=str(item.properties['sentinel:utm_zone'])+item.properties['sentinel:latitude_band']+item.properties['sentinel:grid_square']
        tile_dict={'tile_name':tile}
        bands_dict={}
        for b in band_list:
            band_url=item.assets[b]['href']
            bands_dict[b]=band_url
            img_name=item.properties['sentinel:product_id']
            imgs_dict={'image_name':img_name,'bands':bands_dict}
            tile_dict[f'tile_images']=imgs_dict        

        result_list.append(tile_dict)
    
    return result_list


def show_result_df(result=None,items_list=[]):
    #this function return stac result as a pandas dataframe
    if not items_list:
        # if you want to see all result from main stac result(find_stac_result function)
        # you can use this method
        items = result.items()
        items_json=items.geojson()
        items_json=json.dumps(items_json)
        df=gpd.read_file(items_json)
        df['datetime']=pd.to_datetime(df['datetime'], infer_datetime_format=True)
        df['datetime']=pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d')
        return df
    else:
        # list comprehension result from find_sentinel_item method,
        # you can show that as a dataframe

        #create empty df
        df=gpd.GeoDataFrame()
        for item in items_list:
            #get item properties as a json
            items_json=item.properties
            tmp=gpd.GeoDataFrame(items_json)
            geo_dict = {'geometry': [Polygon(item.geometry['coordinates'][0])]}
            gdf = gpd.GeoDataFrame(geo_dict, crs="EPSG:4326")
            #import geo info
            tmp['geometry']=gdf['geometry']
            df=df.append(tmp)
            #change datatime columt datatype
        df['datetime']=pd.to_datetime(df['datetime'], infer_datetime_format=True)
        df['datetime']=pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d')
        df.reset_index(inplace=True)
        return df
#***********************************
#bnun dataframe ile dusunelim
def drop_tile(df,drop_list):
    # All tile numbers in intersect dataframe
    # You can select your tile which you want to drop from list
    #tiles_list = sorted(df['name'].unique().tolist())
    tmp=df.copy()
    for t in drop_list:
        tmp.drop(tmp[tmp.name==t].index, inplace=True)
    tmp.reset_index(inplace=True)
    df=tmp.copy()
    return df
#######################################
def show_result_map(result=None,items_list=[],target_area=None,overview=False):
    center = [38,35]
    zoom = 6
    if not items_list:
        items = result.items()
        m = folium.Map(location=center, zoom_start=zoom)
        if target_area is not None:
            geo=target_area.__geo_interface__
            m.add_child(folium.GeoJson(geo, name='Area of Study', 
                           style_function=lambda x: {'color': 'red', 'alpha': 0}))
        for item in items:
            if overview:
                #get overview band
                #get overview band
                band_url=item.assets['thumbnail']['href'] 
                pol=Polygon(item.geometry['coordinates'][0])
                folium.raster_layers.ImageOverlay(
                    image=band_url,
                    name=item.properties['sentinel:product_id'],
                    bounds=[[min(pol.exterior.coords.xy[1]),min(pol.exterior.coords.xy[0])],[max(pol.exterior.coords.xy[1]),max(pol.exterior.coords.xy[0])] ],
                    #bounds=[item.geometry['coordinates'][0][0], item.geometry['coordinates'][0][3]],
                    opacity=1,
                    interactive=True,
                    cross_origin=True,
                    zindex=1,
                    #alt="Wikipedia File:Mercator projection SW.jpg",
                ).add_to(m)   

                
            else:
                # Create a string for the name containing the path and row of this Polygon
                name = item.properties['sentinel:product_id']
                # Create the folium geometry of this Polygon 
                g = folium.GeoJson(item.geometry, name=name)
                # Add a folium Popup object with the name string
                g.add_child(folium.Popup(name))
                # Add the object to the map
                g.add_to(m)

        
    
        folium.LayerControl().add_to(m)
        return m

      
    else:
        m = folium.Map(location=center, zoom_start=zoom)
        if target_area is not None:
            geo=target_area.__geo_interface__
            m.add_child(folium.GeoJson(geo, name='Area of Study', 
                           style_function=lambda x: {'color': 'red', 'alpha': 0}))
             
        for item in items_list:
            if overview:
                #get overview band
                band_url=item.assets['thumbnail']['href'] 
                pol=Polygon(item.geometry['coordinates'][0])
                folium.raster_layers.ImageOverlay(
                    image=band_url,
                    name=item.properties['sentinel:product_id'],
                    bounds=[[min(pol.exterior.coords.xy[1]),min(pol.exterior.coords.xy[0])],[max(pol.exterior.coords.xy[1]),max(pol.exterior.coords.xy[0])] ],
                    #bounds=[item.geometry['coordinates'][0][0], item.geometry['coordinates'][0][3]],
                    opacity=1,
                    interactive=True,
                    cross_origin=True,
                    zindex=1,
                    #alt="Wikipedia File:Mercator projection SW.jpg",
                ).add_to(m)   

            else:
                # Create a string for the name containing the path and row of this Polygon
                name = item.properties['sentinel:product_id']
                # Create the folium geometry of this Polygon 
                g = folium.GeoJson(item.geometry, name=name)
                # Add a folium Popup object with the name string
                g.add_child(folium.Popup(name))
                # Add the object to the map
                g.add_to(m)
        
    
        folium.LayerControl().add_to(m)
        return m
 
def download_image(stac_result=None,item_id_list=[],item_list=[] ,band_list=[],download_path='./sentinel_cog',name_suffix='',auto_folder=True):
    #if #list here

    if auto_folder:
        name_suffix='${date}/${id}/${id}_'+name_suffix
    else:
        #if you use this method, you should change your suffix for each sentinel tile with python method such as f string.
        # name_suffix=f'outputfolder/image{i}'  i parameter comes from for loop.
        name_suffix=name_suffix
    if item_list:
        for item in item_list:
            if not band_list:
                item.download_assets(filename_template=download_path+'/'+name_suffix)
            else:
                for band in band_list:
                    item.download(band,filename_template=download_path+'/'+name_suffix)
        
        return download_path

    if item_id_list:
        # sampel list= ['S2A_MSIL2A_20200711T080611_N0214_R078_T37SDA_20200711T112854','S2A_MSIL2A_20200711T080611_N0214_R078_T37SDA_20200711T112854']
        items = stac_result.items()
        items.filter('sentinel:product_id',item_id_list)
    else:
        items = stac_result.items()        
    
    if not band_list :  
        items.download_assets(filename_template=download_path+'/'+name_suffix)
        return download_path
    else:
        for band in band_list:
            items.download(band, filename_template=download_path+'/'+name_suffix)
        return download_path


band_list=['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A',
           'B09', 'B11', 'B12', 'AOT', 'WVP', 'SCL', 'info', 'metadata',
           'visual', 'overview', 'thumbnail']

def download_subset_image(download_status=False,stac_result=None,item_id_list=[],
                        item_list=[],aoi=None,target_epsg='',
                        band_list=band_list[:-5],
                        download_path='./sentinel_cog',name_suffix='',auto_folder=True):
    result_list=[]
    if item_list:
        for item in item_list:
            bands_dict={}
            for band in band_list:
                img_name=item.properties['sentinel:product_id']
                bands_dict['image_name']=img_name
                band_url=item.assets[band]['href']
                rds = rioxarray.open_rasterio(band_url, masked=True, chunks=(512,512))
                #aoi data from http://geojson.io 
                # get aoi as geopandas df
                datajson=json.dumps(aoi)
                target_area=gpd.read_file(datajson)
                #https://geopandas.org/projections.html
                target_area=target_area.to_crs(rds.rio.crs.to_string())
                clipped =rds.rio.clip(target_area.geometry)
                 
                if target_epsg:
                    # target_epsg='epsg:4326'
                    clipped = clipped.rio.reproject(target_epsg)

                if download_status:
                    img_path=download_path+'/'+img_name
                    if not os.path.isdir(img_path):
                        os.mkdir(img_path)
                    img_name=band+'.tif'
                    clipped.rio.to_raster(img_path+'/'+img_name)
                   
                bands_dict[band]=clipped.copy()
                rds=None
            result_list.append(bands_dict)
        return result_list



# def find image from id https://github.com/sat-utils/sat-search/issues/52
            