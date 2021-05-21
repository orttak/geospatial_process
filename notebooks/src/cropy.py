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
import requests
from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date
import datetime
warnings.filterwarnings("ignore")

#https://nbviewer.jupyter.org/github/pangeo-data/cog-best-practices/blob/main/0-single-cog.ipynb
import os
os.environ['GDAL_DISABLE_READDIR_ON_OPEN']='EMPTY_DIR' #This is KEY! otherwise we send a bunch of HTTP GET requests to test for common sidecar metadata
os.environ['AWS_NO_SIGN_REQUEST']='YES' #Since this is a public bucket, we don't need authentication
os.environ['GDAL_MAX_RAW_BLOCK_CACHE_SIZE']='8000000000'  #200MB: Want this to be greater than size of uncompressed raster to overcome a 10 MB limit in the GeoTIFF driver for range request merging.
os.environ['GDAL_SWATH_SIZE']='8000000000'  #also increase this if increasing MAX_RAW_BLOCK_CACHE_SIZE
os.environ['VSI_CURL_CACHE_SIZE']='8000000000' #also increase this if increasing MAX_RAW_BLOCK_CACHE_SIZE8

class VectorProcessing():
    '''
    Burada isleme gore girdiler tanimlayabiliriz. ilk methodlarimiz hedef alani harita uzerinde gosterme
    ve sentine_tile ile cakistirma yaparak, hangi sentinel tile ID ler ile kesistigini gosteriyorz.
    '''
    def __init__(self,target_area,):
        #target_area tipine gore girdimizi tanimliyoruz
        #bu yapiyi ornegin, nokta verisi ile bir islem ekledigimiz zaman farkli bir isim ile onada yapabiliriz
        # nokta_input== xxx >> self.nokta_input=nokta_input seklinde
        if isinstance(target_area,dict):
            datajson=json.dumps(target_area)
            self.target_area=gpd.read_file(datajson)
        elif isinstance(target_area,gpd.geodataframe.GeoDataFrame):
            self.target_area=target_area
        else:
            raise DataError

        
    
    # methoda bir class objesi yaratmadan hemde class degiskenelrini kullanmak icin @classmethod kullandik
    @classmethod
    def show_vector(cls,target_area):
        #get class variable
        target_area=cls(target_area=target_area).target_area
        xy = np.asarray(target_area.centroid[0].xy).squeeze()
        center = list(xy[::-1])
        zoom = 7
        # Create the most basic OSM folium map
        m = folium.Map(location=center, zoom_start=zoom)
        # Add the bounds GeoDataFrame in red
        m.add_child(folium.GeoJson(target_area.__geo_interface__, name='Area of Study', 
                                style_function=lambda x: {'color': 'red', 'alpha': 0}))
        return m
    
    @classmethod  
    def show_intersection(cls,target_area,base_vector_path):
        cls_inst=cls(target_area=target_area)
        target_area=cls_inst.target_area
        base_vector=gpd.read_file(base_vector_path)
        #intersect target area with base-vector path-row 
        intersection_df = base_vector[base_vector.intersects(target_area.geometry[0])]
        intersection_df.reset_index(drop=True,inplace=True)
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
        for i, row in intersection_df.iterrows():
            # Create a string for the name containing the path and row of this Polygon
            name = f'ID:{row[0]}' 
            # Create the folium geometry of this Polygon 
            g = folium.GeoJson(row.geometry.__geo_interface__, name=name)
            # Add a folium Popup object with the name string
            g.add_child(folium.Popup(name))
            # Add the object to the map
            g.add_to(m)

        return m,intersection_df

class Stac():
    bands_list=['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A',
           'B09', 'B11', 'B12', 'AOT', 'WVP', 'SCL', 'info', 'metadata',
           'visual', 'overview', 'thumbnail']
    #https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm

    def __init__(self,target_aoi,date,max_cloud=10):
        self.target_aoi=target_aoi
        self.date=date
        self.max_cloud=max_cloud

        self.URL='https://earth-search.aws.element84.com/v0'
        self.stac_result = satsearch.Search.search(url=self.URL,collections=['sentinel-s2-l2a-cogs'],
                                datetime=self.date,
                                bbox=self.target_aoi,
                                query={'eo:cloud_cover': {'lt':self.max_cloud}}, )
        
        self.stac_items=self.stac_result.items()
        self.tiles_list=[]
        self.min_coverage=95

    def create_tiles_list(self):
        items = self.stac_result.items()
        items_json=items.geojson()
        items_json=json.dumps(items_json)
        df=gpd.read_file(items_json)
        df['tile']=df.apply(lambda row: str(row['sentinel:utm_zone'])+row['sentinel:latitude_band']+row['sentinel:grid_square'], axis=1)
        self.tiles_list = sorted(df['tile'].unique().tolist())
        return self.tiles_list    
    
#### denemeeeeeeeeeeeeeeeeeeeeeeeeeee yapppppppppppppppp
    def find_time_series(self):
        if not self.tiles_list:
            self.create_tiles_list()
        tile_number_list=[]
        lat_band_list=[]
        grid_sq_list=[]
        for tile in self.tiles_list:            
            tile_number_list.append(int(tile[0:2]))
            lat_band_list.append(tile[2])
            grid_sq_list.append(tile[3:])
        self.stac_items.filter('sentinel:utm_zone',tile_number_list)
        self.stac_items.filter('sentinel:latitude_band',lat_band_list)
        self.stac_items.filter('sentinel:grid_square', grid_sq_list)   


    # __ (2*_) hidden method
    def __find_best_image(self):
        tile_result_list=[]
        for tile in self.tiles_list:            
            tile_number=int(tile[0:2])
            lat_band=tile[2]
            grid_sq=tile[3:]
            tmp_items=satstac.ItemCollection(self.stac_items)
            tmp_items.filter('sentinel:utm_zone',[tile_number,])
            tmp_items.filter('sentinel:latitude_band',[lat_band])
            tmp_items.filter('sentinel:grid_square', [grid_sq])
            try:
                coverage=sorted(tmp_items.properties('sentinel:data_coverage'))
                #self.min_coverage=95. 95 den buyuk coverage sahip goruntuleri aliyoruz
                coverage_sorted=[c for c in coverage if c>=self.min_coverage]
                if coverage_sorted:
                    tmp_items.filter('sentinel:data_coverage',coverage_sorted)
                else:
                    coverage_sorted=coverage[-2:]
                    tmp_items.filter('sentinel:data_coverage',coverage_sorted)

                if 100 in tmp_items.properties('sentinel:data_coverage'):
                    tmp_items.filter('sentinel:data_coverage',[100])
                    #get images cloud info
                    selected_item=sorted(tmp_items.properties('eo:cloud_cover'))[0]
                    tmp_items.filter('eo:cloud_cover', [selected_item])
                    
                    '''
                    #burada %2 den kucuk olana sorgu atip, tekrar sort yaptiktan sonra en dusuk olani almak ile
                    # direk en kucuk olani almak ayni sey:D 
                    #test ile kontrol edilecek
                    filtered_list=[tmp_items[x].properties['eo:cloud_cover'] for i, x in enumerate(dc_index) if tmp_items[x].properties['eo:cloud_cover']<self.max_cloud] 
                    # if data exist we use filter method
                    if filtered_list:
                        #get first cloud cover after sorting
                        selected_item=sorted(filtered_list)[0]
                        tmp_items.filter('sentinel:data_coverage',[100])

                    else:
                        #get the min cloud coverage
                        selected_item=sorted(tmp_items.properties('eo:cloud_cover'))[0]
                        tmp_items.filter('eo:cloud_cover', [selected_item])
                    '''
                    
                    tile_result_list.append(tmp_items[0])

                else:
                    selected_item=sorted(tmp_items.properties('eo:cloud_cover'))[0]
                    #select best image
                    tmp_items.filter('eo:cloud_cover', [selected_item])
                    #get newest image
                    tile_result_list.append(tmp_items[0])
                
                if not tmp_items:
                    raise IndexError
            except:
                tmp_items=satstac.ItemCollection(self.stac_items)
                tmp_items.filter('sentinel:utm_zone',[tile_number,])
                tmp_items.filter('sentinel:latitude_band',[lat_band])
                tmp_items.filter('sentinel:grid_square', [grid_sq])
                coverage=sorted(tmp_items.properties('sentinel:data_coverage'))[-2:]
                tmp_items.filter('sentinel:data_coverage',coverage)
                selected_item=sorted(tmp_items.properties('eo:cloud_cover'))[0]
                #select best image
                tmp_items.filter('eo:cloud_cover', [selected_item])
                #get newest image
                tile_result_list.append(tmp_items[0])

        return tile_result_list
    
    def find_sentinel_item(self):
        '''
        With this function, we return best Sentinel-2 image from your Stac search.
        Best image means: 
        -data_coverage>95 or highest data_coverage for target Sentinel-2 tile. If there is no tile that bigger than 95%, 
        function return min cloud percentage in highest 2 data_coverega
        -min_cloud coverage for target Sentinel-2 tile 
        '''
        if self.tiles_list:
            tile_result_list=self.__find_best_image()
            return tile_result_list

        else:
            # code define best image for each tile from stac result
            # function return list of tile's information
            self.tiles_list=self.create_tiles_list()
            #we collect each tile's result
            tile_result_list=self.__find_best_image()          
            
            return tile_result_list

    @staticmethod     
    def show_result_list(sentinel_items_list,band_list=bands_list):
        '''
        items_list=stac_result.show_result_list(sentinel_items_list=sentinel_items)
        '''
        # bu yapiyi direk web yada api ortamina gonderebiliriz
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

    @staticmethod
    def show_result_df(result=None,sentinel_items_list=[]):
        '''
        This method return pandas dataframe of your stac result or your items list
        df=stac_result.show_result_df(result=stac_result.stac_result) 

        df=stac_result.show_result_df(items_list=sentinel_items)
        '''
        # this function return stac result as a pandas dataframe
        # item_list come from find_sentinel_item() function
        if not sentinel_items_list:
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
            for item in sentinel_items_list:
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
            df.reset_index(inplace=True,drop=True)
            return df
    @staticmethod       
    def __add_map(items,target_area=None,overview=False):
        center = items[0].geometry['coordinates'][0][0]
        zoom = 6
        m = folium.Map(location=center, zoom_start=zoom)
        if target_area is not None:
            if isinstance(target_area,dict):
                datajson=json.dumps(target_area)
                target_area=gpd.read_file(datajson)
            elif isinstance(target_area,gpd.geodataframe.GeoDataFrame):
                target_area=target_area
            else:
                raise DataError
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

    @staticmethod
    def show_result_map(result=None,items_list=[],target_area=None,overview=False):      
        if not items_list:
            items = result.items()
            items_map=Stac.__add_map(items=items,target_area=target_area,overview=overview)
            return items_map
        else:
            items_map=Stac.__add_map(items=items_list,target_area=target_area,overview=overview)
            return items_map

    @staticmethod
    def check_image_url(url):
        """
        check Stac result URL 

        return: success or failure
        """
        resp = requests.get(f'http://cog-validate.radiant.earth/api/validate?url={url}')
        return resp.json()['status']

    
    @staticmethod
    def __create_log_file(target_text, filename):
        f = open(filename, "a")
        f.write(f'{target_text}')
        f.close()


    @staticmethod
    def download_error_image(img_date,geo_img,img_id,username,password):
        '''
        After read error file(image_error.txt) you can get image info which you failed from COG Sentinel-2, you can use this info with this function
        if you have more than 1 image, you can download with for loop.

        You can find img_date, geo_img and img_id information in image_error.txt file.

        api,target_image_id=download_error_image(img_date,geo_img,img_id,username,password)
        api.download(target_image_id,directory_path='.')
        api.download('7be30c50-31fc-48c4-ab45-fddea9be7877',directory_path='.')

        if you get error like >> Product 7be30c50-31fc-48c4-ab45-fddea9be7877 is not online. Triggering retrieval from long term archive.
        Go to https://sentinelsat.readthedocs.io/en/stable/api.html#lta-products

        username and password should be string
        '''
        api = SentinelAPI(username, password, 'https://scihub.copernicus.eu/dhus')
        day_before =img_date- datetime.timedelta(days=1)
        day_after =img_date + datetime.timedelta(days=1)
        footprint = geojson_to_wkt(geo_img)
        products = api.query(footprint,
                             #date = ('20181219', date(2018, 12, 29)),
                             date=(day_before,day_after),
                             platformname = 'Sentinel-2',
                             )
        sat_df=api.to_geodataframe(products)
        result=sat_df.loc[sat_df['title']==img_id]
        return api,result.index.values[0]

        
    @staticmethod
    def __download_items(item_list,band_list,download_path,name_suffix):
        for item in item_list:
            try:
                for band in band_list:
                    item.download(band,filename_template=download_path+'/'+name_suffix)
            except:
                txt=f'image_id:{item.properties["sentinel:product_id"]},geometry:{item.geometry},date:{item.date} \n'
                __create_log_file(target_text=txt,filename=download_path+f'/image_error_{item.id}.txt')
                continue
        
        return download_path

    @staticmethod
    def download_image(stac_result=None,item_id_list=[],item_list=[] ,band_list=bands_list[:-7],download_path='./sentinel_cog',name_suffix='',auto_folder=True):
        """
        There are 3 different download methods in this function. If you don't give any band list, function use default band list.

        1- Use item_list from "stac_result.find_sentinel_item()" method
        2- Use stac result and Sentinel image ID which you can get from show_result_df or show_result_list methods
        3- If you just give the stac result to this function, function download all images in your stac result with default bands. According your time range
        and target area, this method could take more time.
        default_bands=['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A',
           'B09', 'B11', 'B12']
        defautl download path > './sentinel_cog'
        """
        if auto_folder:
            # we cam get date/id info from stac item
            name_suffix='${date}/${id}/${id}_'+name_suffix
        else:
            #if you use this method, you should change your suffix for each sentinel tile with python method such as f string.
            # name_suffix=f'outputfolder/image{i}'  i parameter comes from for loop.
            name_suffix=name_suffix

        if item_list:
            return Stac.__download_items(item_list=item_list,band_list=band_list,
                                    download_path=download_path,name_suffix=name_suffix)
        
        elif item_id_list:
            # sampel list= ['S2A_MSIL2A_20200711T080611_N0214_R078_T37SDA_20200711T112854','S2A_MSIL2A_20200711T080611_N0214_R078_T37SDA_20200711T112854']
            items = stac_result.items()
            items.filter('sentinel:product_id',item_id_list)
            return Stac.__download_items(items,band_list,download_path,name_suffix)

        else:
            items = stac_result.items()
            return Stac.__download_items(items,band_list,download_path,name_suffix)       

  
    @staticmethod
    def __download_subset_items(download_status,item_list,aoi,target_epsg,band_list,
                            download_path):
        result_list=[]
        for item in item_list:
            bands_dict={}
            for band in band_list:
                img_name=item.properties['sentinel:product_id']
                bands_dict['image_name']=img_name
                band_url=item.assets[band]['href']
                try:
                    rds = rioxarray.open_rasterio(band_url, masked=False, chunks=(1, "auto", -1))
                except:
                    txt=f'image_id:{item.properties["sentinel:product_id"]},geometry:{item.geometry},date:{item.date} \n'
                    __create_log_file(target_text=txt,filename=download_path+f'/image_error_{item.id}.txt')
                    continue
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
                    if not os.path.isdir(download_path):
                        os.mkdir(download_path)                        
                    if not os.path.isdir(img_path):
                        os.mkdir(img_path)
                    img_name=band+'_subset.tif'
                    clipped.rio.to_raster(img_path+'/'+img_name)
                band_clipped=clipped.copy()
                bands_dict[band]=band_clipped
                rds=None
            ################### FEATURE #######################
            #clipped image could be none to avoid high memory usage. we should write test about it and then put new variables 
            #for manage the clipped image
            result_list.append(bands_dict)
        return result_list

    @staticmethod
    def download_subset_image(download_status=False,stac_result=None,item_id_list=[],
                            item_list=[],aoi=None,target_epsg='',
                            band_list=bands_list[:-5],
                            download_path='./sentinel_cog'):
        '''
        With this function, you can get subset data from your Stac items.You can get data as xarray that you can convert numpy array and use
        in another function. Also you can directly save the subset image with these parameters >> download_status=True and download_path='your_target_path'

        There are 3 different download methods in this function. Also, If you don't give any band list, function use default band list.

        1- Use item_list from "stac_result.find_sentinel_item()" method
        2- Use stac result and Sentinel image ID which you can get from show_result_df or show_result_list methods
        3- If you just give the stac result to this function, function download all images in your stac result with default bands. According your time range
        and target area, this method could take more time.

        default_bands=['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A',
           'B09', 'B11', 'B12']
        defautl download path > './sentinel_cog'
        '''


        if item_list:
            result_list=Stac.__download_subset_items(download_status=download_status,item_list=item_list,
            aoi=aoi,target_epsg=target_epsg,band_list=band_list,
                            download_path=download_path)
            return result_list

        elif item_id_list:
            # sampel list= ['S2A_MSIL2A_20200711T080611_N0214_R078_T37SDA_20200711T112854','S2A_MSIL2A_20200711T080611_N0214_R078_T37SDA_20200711T112854']
            items = stac_result.items()
            items.filter('sentinel:product_id',item_id_list)
            result_list=Stac.__download_subset_items(download_status=download_status,item_list=items,
            aoi=aoi,target_epsg=target_epsg,band_list=band_list,
                            download_path=download_path)
            return result_list

        else:
            items = stac_result.items()
            result_list=Stac.__download_subset_items(download_status=download_status,item_list=items,
            aoi=aoi,target_epsg=target_epsg,band_list=band_list,
                            download_path=download_path)
            return result_list

import numpy as np
import xarray as xr

def xarray_calc_stats(dataset,method,dim_name,input_nodata,input_dtype):
    dataset=dataset.astype(np.float64)
    #define nodata as np>nan to ignore nodata calculating statistic
    dataset.rio.write_nodata(np.nan, inplace=True)
    if method=='mean':
        dataset_mean = dataset.mean(dim=dim_name,skipna=True).compute()
        dataset_mean=dataset_mean.rio.write_nodata(input_nodata, inplace=True)
        dataset_mean=dataset_mean.astype(input_dtype)
        return dataset_mean

    elif method=='median':
        dataset_median = dataset.median(dim=dim_name,skipna=True).compute()
        dataset_median=dataset_median.astype(input_dtype)
        dataset_median.rio.write_nodata(input_nodata, inplace=True)
        return dataset_median

    elif method=='min':
        dataset_min = dataset.min(dim=dim_name,skipna=True).compute()
        dataset_min=dataset_min.astype(input_dtype)
        dataset_min.rio.write_nodata(input_nodata, inplace=True)
        return dataset_min

    elif method=='max':
        dataset_max = dataset.max(dim=dim_name,skipna=True).compute()
        dataset_max=dataset_max.astype(input_dtype)
        dataset_max.rio.write_nodata(input_nodata, inplace=True)
        return dataset_max

def open_image(input_img):
    rds = rioxarray.open_rasterio(input_img, masked=False, chunks=(1, "auto", -1))
    return rds

def calc_img_stat(stac_result:list =[],imgs_list:list=[],numberOfBand:list=[],inputtarget_epsg: str ='epsg:4326',statistic_method: str = 'mean',input_dim_name: str='band',output_dim_name: str='band',input_nodata:int=None,input_dtype=None):
    '''
    With this function, you can calculate mean,median, min and max values for list of images. If you don't define a numberOfBand list for stac_result, 
    the system calculate all band in your images list.

    You can use as input:
    > Result of download_subset_image result (from stac_query.download_subset_image)
    > If you use download_subset_image, you can get stats_result of each band as dictionary.
    
    From your local directory, you can give tif path as a list or before read with rioxarray and give that list
    > list of tif files
    > list of xarray Dataarray
    
    Methods
    > mean,median, min and max
    > statistic_method = 'mean' #default value is mean
    
    Input and output dim names are importan to stack the data, Algorithm use this variable.
    >input_dim_name='band' #default value
    >output_dim_name='time' #default value
    
    You can define input dtype or system read from first image of your list.
    >input_dtype= np.uint16   # it shoud be numpy dtype data

    Above methods, you can get statistic result of input images
    '''
           
    if stac_result:
        numberOfImages=len(stac_result)
        if not numberOfBand:
            numberOfBand=list(stac_result[0].keys())[1:]
        
        if input_nodata:
            input_nodata=input_nodata     
        else:
            target_img=stac_result[0][numberOfBand[0]]
            input_nodata=target_img.rio.nodata

        if not input_dtype:
            input_dtype=stac_result[0][numberOfBand[0]].dtype

        
        band_dict={}
        for band in numberOfBand:
            band_list=[]
            for img in stac_result:
                band_list.append(img[band])
            stack_bands = xr.concat(band_list, dim=input_dim_name, join='outer' ).rename(band=output_dim_name)
            stats_result=xarray_calc_stats(stack_bands,statistic_method,output_dim_name,input_nodata,input_dtype)
            band_dict[band]=stats_result
        return band_dict

    else:
        if type(imgs_list[0])!=str:
            stack_bands = xr.concat(imgs_list, dim=input_dim_name, join='outer' ).rename(band=output_dim_name)
            if input_nodata:
                input_nodata=input_nodata     
            else:
                target_img=imgs_list[0]
                input_nodata=target_img.rio.nodata 
            
            if not input_dtype:
                input_dtype=imgs_list[0].dtype

            stats_result=xarray_calc_stats(stack_bands,statistic_method,output_dim_name,input_nodata,input_dtype)
            return stats_result
        else:
            input_imgs_list=[open_image(img) for img in imgs_list]

            if input_nodata:
                input_nodata=input_nodata     
            else:
                target_img=input_imgs_list[0]
                input_nodata=target_img.rio.nodata

            if not input_dtype:
                input_dtype=input_imgs_list[0].dtype 
            stack_bands = xr.concat(input_imgs_list, dim=input_dim_name, join='outer' ).rename(band=output_dim_name)
            stats_result=xarray_calc_stats(stack_bands,statistic_method,output_dim_name,input_nodata,input_dtype)
            return stats_result

def control_crs(imgs_list:list):
    unmatched_list=[]
    base_img=imgs_list[0]
    for i in range(len(imgs_list[1:])):
        if base_img.rio.crs!=imgs_list[i+1]:
            unmatched_list.append(i+1)
    return unmatched_list

from rioxarray.merge import merge_arrays


def create_stack(stac_result:list =[],imgs_list:list=[],numberOfBand:list=[],inputtarget_epsg: str ='epsg:4326',input_dim_name: str='band',output_dim_name: str='band',input_nodata:int=None):
    '''
    With this function, you can create stack image from your list. If you don't define a numberOfBand list, the system calculate all band in your images list.
    You can use as input:
    > result of download_subset_image result
    If you use download_subset_image, you can get stats_result of each band as dictionary.

    > list of tif files
    > list of xarray Dataarray
    
    Above methods, you can get statistic result of input images
    '''
    
    if stac_result:
        numberOfImages=len(stac_result)
        if not numberOfBand:
            numberOfBand=list(stac_result[0].keys())[1:]
        
        if input_nodata:
            input_nodata=input_nodata     
        else:
            target_img=stac_result[0][numberOfBand[0]]
            input_nodata=target_img.rio.nodata
        
        band_dict={}
        for band in numberOfBand:
            band_list=[]
            for img in stac_result:
                band_list.append(img[band])
            stack_bands = xr.concat(band_list, dim=input_dim_name, join='outer').rename(band=output_dim_name)
            stack_bands=stack_bands.rio.write_nodata(input_nodata, inplace=True)
            band_dict[band]=stack_bands
        return band_dict

    else:
        if type(imgs_list[0])!=str:
            if input_nodata:
                input_nodata=input_nodata     
            else:
                target_img=imgs_list[0]
                input_nodata=target_img.rio.nodata 
            stack_bands = xr.concat(imgs_list, dim=input_dim_name, join='outer' ).rename(band=output_dim_name)
            stack_bands=stack_bands.rio.write_nodata(input_nodata, inplace=True)
            return stack_bands
        else:
            input_imgs_list=[open_image(img) for img in imgs_list]
            
            if input_nodata:
                input_nodata=input_nodata     
            else:
                target_img=input_imgs_list[0]
                input_nodata=target_img.rio.nodata

            stack_bands = xr.concat(input_imgs_list, dim=input_dim_name, join='outer' ).rename(band=output_dim_name)
            stack_bands=stack_bands.rio.write_nodata(input_nodata, inplace=True)
            return stack_bands


def create_mosaic(imgs_list:list,reproject=False,target_epsg:str='EPSG:4326',):
    '''
    With this function, you can create mosaic image from images list. You can give image path or xarray object read data before use this function.
    If you define reproject=True, system automatically convert all images into epsg:4326 and mosaic them.
    If you don't define reproject behaviour, system will controll your epsg code according to first image of your list.
    If the next images do not match the base image, they convert to hte base image epsg. Be careful when you work with UTM system

    imgs_list= ['target_path3/S2A_MSIL2A_20181030T081041_N0209_R078_T37SDA_20181030T091317/B02_subset.tif',
    'target_path3/S2A_MSIL2A_20181030T081041_N0209_R078_T37SDA_20181030T091317/B03_subset.tif',
    'target_path3/S2A_MSIL2A_20181030T081041_N0209_R078_T37SDA_20181030T091317/B04_subset.tif',]
    '''
    if type(imgs_list[0])==str:
            imgs_list=[open_image(img) for img in imgs_list]
    if reproject:
        reprojected_list=[]
        for img in imgs_list:
            xds_lonlat = img.rio.reproject("EPSG:4326")
            reprojected_list.append(xds_lonlat)
            # 0 olan yani nodata lar birlestirirken ne yapiyor kontrol etmek gerekiyor
        merged = merge_arrays(reprojected_list)
        return merged

    base_img=imgs_list[0]
    for i in range(len(imgs_list)-1):
        if base_img.rio.crs!=imgs_list[i+1].rio.crs:
            reproject_img=imgs_list[i+1]
            reproject_img=reproject_img.rio.reproject_match(base_img)
            imgs_list[i+1]=reproject_img
    
    merged = merge_arrays(imgs_list)
    return merged

def save_image(input_xarray=None,target_path='target_name.tif',data_type='uint16',nodata_value=0):
    input_xarray.rio.to_raster(target_path,dtype=data_type,tags={'_FillValue': nodata_value})
    return target_path

'''
concat icin bir tane stack function yazilabilir

merge yani mosaic icin fonksiyon yazacaz. girdi olarak verilenleri CRS kontrolu yapilacak. eger CRS uymayan varsa
tum goruntuleri 4326 mi yapmak lazim yoksa once kullanciya bir uyari mi donmek lazim. Kullaniciya bir kontrol fonksiyonu tanimlayip mosaic lemeden once goruntu
CRS icin bir kontrol yapmasini onerebiliriz. Eger buna ragmen mosaiclemek istiyorsa target epsg yi belirtmeli.yoksa 4326 defaul olarak alacaz

bu kontrol fonksiyonu true/false donse guzel olur ama kullanici hangi goruntu farklilik yaratiyor donsekte iyi olabilir. birini gizli fonksiyon digerini kullanici gorsun diye acik sekilde yapabiliriz.
bun ne kadar dogru bir soralim.

'''