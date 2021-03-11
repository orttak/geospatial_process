from dask.distributed import Client, LocalCluster
import multiprocessing as mp
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray

def calculate_ndvi(red,nir):
    with LocalCluster(n_workers=int(0.6 * mp.cpu_count()),
        processes=False,
        threads_per_worker=1,
        memory_limit='2GB',
        #ip='tcp://localhost:9895',
        ) as cluster, Client(cluster) as client:

            red_xarray=red
            nir_xarray=nir
            red=red_xarray.persist()
            nir=nir_xarray.persist()
            red=red.values
            nir=nir.values
            ndvi = (nir.astype(float) - red.astype(float))/(nir + red)

    return ndvi

def direct_ndvi(red,nir):
    with LocalCluster(n_workers=int(0.6 * mp.cpu_count()),
        processes=False,
        threads_per_worker=1,
        memory_limit='2GB',
        #ip='tcp://localhost:9895',
        ) as cluster, Client(cluster) as client:
            red_xarray=red
            nir_xarray=nir
            ndvi = (nir_xarray.astype(float) - red_xarray.astype(float))/(nir_xarray + red_xarray)

    return ndvi
