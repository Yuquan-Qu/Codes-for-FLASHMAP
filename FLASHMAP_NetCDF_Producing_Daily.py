import os
import pandas as pd
import numpy as np
import multiprocessing
from netCDF4 import Dataset

'''Produce Daily FLASHMAP NetCDF Files'''

# Date range and spatial resolution
start_day = '2025-01-01'
end_day = '2025-12-31'
resolution = 0.25  # degrees

# Input directory
IN_DIR_PAR = '/gpfs/work2/0/einf3869/GLD360/Integrator_era/Parquet/2025_Time_Format'
# Output directory
OUT_DIR_NC = f'/gpfs/work2/0/einf3869/GLD360/Integrator_era/FLASHMAP/Daily_NetCDFs'


def GLD360_vector_to_daily_grid(day_range, resolution=resolution, IN_DIR_PAR=IN_DIR_PAR, OUT_DIR_NC=OUT_DIR_NC):

    """
    Generate daily gridded NetCDF from the GLD360 data.

    Parameters
    ----------
    day_range : str
        Date string in the format 'YYYY-MM-DD'.
    resolution : float
        Spatial resolution of the output grid in degrees.
    IN_DIR_PAR : str
        Directory path containing the input parquet files.
    OUT_DIR_NC : str
        Directory path where the output NetCDF files will be saved.

    Returns
    -------
    NetCDF files
    """


    '''
    Read daily GLD360 parquet file.
    '''
    fname = f'GLD360_{day_range}.parquet'
    year = day_range[0:4]
    month = day_range[5:7]
    GLD360_data = pd.read_parquet(f'{IN_DIR_PAR}/{fname}')


    '''
    Separate strokes by lightning type and polarity.
    '''
    CG_data = GLD360_data[GLD360_data['cloud'] == False].copy()
    IC_data = GLD360_data[GLD360_data['cloud'] == True].copy()
    CG_positive_data = CG_data[CG_data['signalStrengthKA'] > 0].copy()
    CG_negative_data = CG_data[CG_data['signalStrengthKA'] < 0].copy()
    IC_positive_data = IC_data[IC_data['signalStrengthKA'] > 0].copy()
    IC_negative_data = IC_data[IC_data['signalStrengthKA'] < 0].copy()


    '''
    Separate flashes by lightning type.
    The definition of multiplicity in GLD360 is the count of CG strokes a flash contains.
    The flash multiplicity shows with the first stroke (either CG or IC) in a flash.
    '''
    GLD360_flash_data = GLD360_data.sort_values('time').drop_duplicates(subset=['flashId'], keep='first')  # The first stroke in flash
    CG_flash_data = GLD360_flash_data[GLD360_flash_data['multiplicity'] > 0].copy()  # This flash contains at least one CG stroke
    IC_flash_data = GLD360_flash_data[GLD360_flash_data['multiplicity'] == 0].copy()  # This flash contains no CG strokes


    '''
    Define the gridding structure.
    Latitude and longitude bins are generated based on the specified spatial resolution.
    Each lightning event will be assigned to a corresponding grid cell.
    '''
    lat_bins = pd.interval_range(-90, 90, freq=resolution)
    lon_bins = pd.interval_range(-180, 180, freq=resolution)
    GLD360_data['lat_bin'] = pd.cut(GLD360_data['latitude'], bins=lat_bins)
    GLD360_data['lon_bin'] = pd.cut(GLD360_data['longitude'], bins=lon_bins)
    CG_data['lat_bin'] = pd.cut(CG_data['latitude'], bins=lat_bins)
    CG_data['lon_bin'] = pd.cut(CG_data['longitude'], bins=lon_bins)
    IC_data['lat_bin'] = pd.cut(IC_data['latitude'], bins=lat_bins)
    IC_data['lon_bin'] = pd.cut(IC_data['longitude'], bins=lon_bins)
    CG_positive_data['lat_bin'] = pd.cut(CG_positive_data['latitude'], bins=lat_bins)
    CG_positive_data['lon_bin'] = pd.cut(CG_positive_data['longitude'], bins=lon_bins)
    CG_negative_data['lat_bin'] = pd.cut(CG_negative_data['latitude'], bins=lat_bins)
    CG_negative_data['lon_bin'] = pd.cut(CG_negative_data['longitude'], bins=lon_bins)
    IC_positive_data['lat_bin'] = pd.cut(IC_positive_data['latitude'], bins=lat_bins)
    IC_positive_data['lon_bin'] = pd.cut(IC_positive_data['longitude'], bins=lon_bins)
    IC_negative_data['lat_bin'] = pd.cut(IC_negative_data['latitude'], bins=lat_bins)
    IC_negative_data['lon_bin'] = pd.cut(IC_negative_data['longitude'], bins=lon_bins)
    GLD360_flash_data['lat_bin'] = pd.cut(GLD360_flash_data['latitude'], bins=lat_bins)
    GLD360_flash_data['lon_bin'] = pd.cut(GLD360_flash_data['longitude'], bins=lon_bins)
    CG_flash_data['lat_bin'] = pd.cut(CG_flash_data['latitude'], bins=lat_bins)
    CG_flash_data['lon_bin'] = pd.cut(CG_flash_data['longitude'], bins=lon_bins)
    IC_flash_data['lat_bin'] = pd.cut(IC_flash_data['latitude'], bins=lat_bins)
    IC_flash_data['lon_bin'] = pd.cut(IC_flash_data['longitude'], bins=lon_bins)


    '''
    Stroke counts:
    Count the number of strokes per grid cell for each type and polarity:
        CG positive / CG negative
        IC positive / IC negative
    '''
    CG_positive_stroke_count_grouped = CG_positive_data.groupby(['lat_bin', 'lon_bin'], observed=False)['flashId'].count().reset_index()
    CG_positive_stroke_count_grid = CG_positive_stroke_count_grouped.pivot(index='lat_bin', columns='lon_bin', values='flashId')

    CG_negative_stroke_count_grouped = CG_negative_data.groupby(['lat_bin', 'lon_bin'], observed=False)['flashId'].count().reset_index()
    CG_negative_stroke_count_grid = CG_negative_stroke_count_grouped.pivot(index='lat_bin', columns='lon_bin', values='flashId')

    IC_positive_stroke_count_grouped = IC_positive_data.groupby(['lat_bin', 'lon_bin'], observed=False)['flashId'].count().reset_index()
    IC_positive_stroke_count_grid = IC_positive_stroke_count_grouped.pivot(index='lat_bin', columns='lon_bin', values='flashId')

    IC_negative_stroke_count_grouped = IC_negative_data.groupby(['lat_bin', 'lon_bin'], observed=False)['flashId'].count().reset_index()
    IC_negative_stroke_count_grid = IC_negative_stroke_count_grouped.pivot(index='lat_bin', columns='lon_bin', values='flashId')


    '''
    Peak current statistics:
    For each grid cell, calculate statistical measures of peak current (signalStrengthKA),
    including mean, median, and standard deviation for each lightning category.
    '''
    # MEAN
    CG_positive_mean_peak_current_grouped = CG_positive_data.groupby(['lat_bin', 'lon_bin'], observed=False)['signalStrengthKA'].mean().reset_index()
    CG_positive_mean_peak_current_grid = CG_positive_mean_peak_current_grouped.pivot(index='lat_bin', columns='lon_bin', values='signalStrengthKA')
    CG_positive_mean_peak_current_grid = np.round(CG_positive_mean_peak_current_grid, 1)
    CG_positive_mean_peak_current_grid[CG_positive_mean_peak_current_grid.isna()] = 0

    CG_negative_mean_peak_current_grouped = CG_negative_data.groupby(['lat_bin', 'lon_bin'], observed=False)['signalStrengthKA'].mean().reset_index()
    CG_negative_mean_peak_current_grid = CG_negative_mean_peak_current_grouped.pivot(index='lat_bin', columns='lon_bin', values='signalStrengthKA')
    CG_negative_mean_peak_current_grid = np.round(CG_negative_mean_peak_current_grid, 1)
    CG_negative_mean_peak_current_grid[CG_negative_mean_peak_current_grid.isna()] = 0

    IC_positive_mean_peak_current_grouped = IC_positive_data.groupby(['lat_bin', 'lon_bin'], observed=False)['signalStrengthKA'].mean().reset_index()
    IC_positive_mean_peak_current_grid = IC_positive_mean_peak_current_grouped.pivot(index='lat_bin', columns='lon_bin', values='signalStrengthKA')
    IC_positive_mean_peak_current_grid = np.round(IC_positive_mean_peak_current_grid, 1)
    IC_positive_mean_peak_current_grid[IC_positive_mean_peak_current_grid.isna()] = 0

    IC_negative_mean_peak_current_grouped = IC_negative_data.groupby(['lat_bin', 'lon_bin'], observed=False)['signalStrengthKA'].mean().reset_index()
    IC_negative_mean_peak_current_grid = IC_negative_mean_peak_current_grouped.pivot(index='lat_bin', columns='lon_bin', values='signalStrengthKA')
    IC_negative_mean_peak_current_grid = np.round(IC_negative_mean_peak_current_grid, 1)
    IC_negative_mean_peak_current_grid[IC_negative_mean_peak_current_grid.isna()] = 0

    # MEDIAN
    CG_positive_median_peak_current_grouped = CG_positive_data.groupby(['lat_bin', 'lon_bin'], observed=False)['signalStrengthKA'].quantile(0.5).reset_index()
    CG_positive_median_peak_current_grid = CG_positive_median_peak_current_grouped.pivot(index='lat_bin', columns='lon_bin', values='signalStrengthKA')
    CG_positive_median_peak_current_grid = np.round(CG_positive_median_peak_current_grid, 1)
    CG_positive_median_peak_current_grid[CG_positive_median_peak_current_grid.isna()] = 0

    CG_negative_median_peak_current_grouped = CG_negative_data.groupby(['lat_bin', 'lon_bin'], observed=False)['signalStrengthKA'].quantile(0.5).reset_index()
    CG_negative_median_peak_current_grid = CG_negative_median_peak_current_grouped.pivot(index='lat_bin', columns='lon_bin', values='signalStrengthKA')
    CG_negative_median_peak_current_grid = np.round(CG_negative_median_peak_current_grid, 1)
    CG_negative_median_peak_current_grid[CG_negative_median_peak_current_grid.isna()] = 0

    IC_positive_median_peak_current_grouped = IC_positive_data.groupby(['lat_bin', 'lon_bin'], observed=False)['signalStrengthKA'].quantile(0.5).reset_index()
    IC_positive_median_peak_current_grid = IC_positive_median_peak_current_grouped.pivot(index='lat_bin', columns='lon_bin', values='signalStrengthKA')
    IC_positive_median_peak_current_grid = np.round(IC_positive_median_peak_current_grid, 1)
    IC_positive_median_peak_current_grid[IC_positive_median_peak_current_grid.isna()] = 0

    IC_negative_median_peak_current_grouped = IC_negative_data.groupby(['lat_bin', 'lon_bin'], observed=False)['signalStrengthKA'].quantile(0.5).reset_index()
    IC_negative_median_peak_current_grid = IC_negative_median_peak_current_grouped.pivot(index='lat_bin', columns='lon_bin', values='signalStrengthKA')
    IC_negative_median_peak_current_grid = np.round(IC_negative_median_peak_current_grid, 1)
    IC_negative_median_peak_current_grid[IC_negative_median_peak_current_grid.isna()] = 0

    # STD DEV
    CG_positive_stddev_peak_current_grouped = CG_positive_data.groupby(['lat_bin', 'lon_bin'], observed=False)['signalStrengthKA'].std().reset_index()
    CG_positive_stddev_peak_current_grid = CG_positive_stddev_peak_current_grouped.pivot(index='lat_bin', columns='lon_bin', values='signalStrengthKA')
    CG_positive_stddev_peak_current_grid = np.round(CG_positive_stddev_peak_current_grid, 1)
    CG_positive_stddev_peak_current_grid[CG_positive_stddev_peak_current_grid.isna()] = 0

    CG_negative_stddev_peak_current_grouped = CG_negative_data.groupby(['lat_bin', 'lon_bin'], observed=False)['signalStrengthKA'].std().reset_index()
    CG_negative_stddev_peak_current_grid = CG_negative_stddev_peak_current_grouped.pivot(index='lat_bin', columns='lon_bin', values='signalStrengthKA')
    CG_negative_stddev_peak_current_grid = np.round(CG_negative_stddev_peak_current_grid, 1)
    CG_negative_stddev_peak_current_grid[CG_negative_stddev_peak_current_grid.isna()] = 0

    IC_positive_stddev_peak_current_grouped = IC_positive_data.groupby(['lat_bin', 'lon_bin'], observed=False)['signalStrengthKA'].std().reset_index()
    IC_positive_stddev_peak_current_grid = IC_positive_stddev_peak_current_grouped.pivot(index='lat_bin', columns='lon_bin', values='signalStrengthKA')
    IC_positive_stddev_peak_current_grid = np.round(IC_positive_stddev_peak_current_grid, 1)
    IC_positive_stddev_peak_current_grid[IC_positive_stddev_peak_current_grid.isna()] = 0

    IC_negative_stddev_peak_current_grouped = IC_negative_data.groupby(['lat_bin', 'lon_bin'], observed=False)['signalStrengthKA'].std().reset_index()
    IC_negative_stddev_peak_current_grid = IC_negative_stddev_peak_current_grouped.pivot(index='lat_bin', columns='lon_bin', values='signalStrengthKA')
    IC_negative_stddev_peak_current_grid = np.round(IC_negative_stddev_peak_current_grid, 1)
    IC_negative_stddev_peak_current_grid[IC_negative_stddev_peak_current_grid.isna()] = 0


    '''
    Flash counts:
    Count the CG and IC flashes separately.
    Here, a flash location is defined by the location of the first stroke in the flash
    '''
    CG_flash_count_grouped = CG_flash_data.groupby(['lat_bin', 'lon_bin'], observed=False)['flashId'].count().reset_index()
    CG_flash_count_grid = CG_flash_count_grouped.pivot(index='lat_bin', columns='lon_bin', values='flashId')

    IC_flash_count_grouped = IC_flash_data.groupby(['lat_bin', 'lon_bin'], observed=False)['flashId'].count().reset_index()
    IC_flash_count_grid = IC_flash_count_grouped.pivot(index='lat_bin', columns='lon_bin', values='flashId')


    '''
    Flash multiplicity:
    Compute multiplicity statistics (mean, median, std) per grid cell.
    The multiplicity of a CG flash is the count of CG strokes it contains. 
    The multiplicity of an IC flash is the count of IC strokes it contains.
    For CG flashes, multiplicity is derived from 'multiplicity'.
    For IC flashes, multiplicity is derived from 'numICPulses'.
    '''
    CG_flash_mean_multiplicity_grouped = CG_flash_data.groupby(['lat_bin', 'lon_bin'], observed=False)['multiplicity'].mean().reset_index()
    CG_flash_mean_multiplicity_grid = CG_flash_mean_multiplicity_grouped.pivot(index='lat_bin', columns='lon_bin', values='multiplicity')
    CG_flash_mean_multiplicity_grid = np.round(CG_flash_mean_multiplicity_grid, 1)
    CG_flash_mean_multiplicity_grid[CG_flash_mean_multiplicity_grid.isna()] = 0

    CG_flash_median_multiplicity_grouped = CG_flash_data.groupby(['lat_bin', 'lon_bin'], observed=False)['multiplicity'].quantile(0.5).reset_index()
    CG_flash_median_multiplicity_grid = CG_flash_median_multiplicity_grouped.pivot(index='lat_bin', columns='lon_bin', values='multiplicity')
    CG_flash_median_multiplicity_grid = np.round(CG_flash_median_multiplicity_grid, 1)
    CG_flash_median_multiplicity_grid[CG_flash_median_multiplicity_grid.isna()] = 0

    CG_flash_stddev_multiplicity_grouped = CG_flash_data.groupby(['lat_bin', 'lon_bin'], observed=False)['multiplicity'].std().reset_index()
    CG_flash_stddev_multiplicity_grid = CG_flash_stddev_multiplicity_grouped.pivot(index='lat_bin', columns='lon_bin', values='multiplicity')
    CG_flash_stddev_multiplicity_grid = np.round(CG_flash_stddev_multiplicity_grid, 1)
    CG_flash_stddev_multiplicity_grid[CG_flash_stddev_multiplicity_grid.isna()] = 0

    IC_flash_mean_multiplicity_grouped = IC_flash_data.groupby(['lat_bin', 'lon_bin'], observed=False)['numCloudPulses'].mean().reset_index()
    IC_flash_mean_multiplicity_grid = IC_flash_mean_multiplicity_grouped.pivot(index='lat_bin', columns='lon_bin', values='numCloudPulses')
    IC_flash_mean_multiplicity_grid = np.round(IC_flash_mean_multiplicity_grid, 1)
    IC_flash_mean_multiplicity_grid[IC_flash_mean_multiplicity_grid.isna()] = 0

    IC_flash_median_multiplicity_grouped = IC_flash_data.groupby(['lat_bin', 'lon_bin'], observed=False)['numCloudPulses'].quantile(0.5).reset_index()
    IC_flash_median_multiplicity_grid = IC_flash_median_multiplicity_grouped.pivot(index='lat_bin', columns='lon_bin', values='numCloudPulses')
    IC_flash_median_multiplicity_grid = np.round(IC_flash_median_multiplicity_grid, 1)
    IC_flash_median_multiplicity_grid[IC_flash_median_multiplicity_grid.isna()] = 0

    IC_flash_stddev_multiplicity_grouped = IC_flash_data.groupby(['lat_bin', 'lon_bin'], observed=False)['numCloudPulses'].std().reset_index()
    IC_flash_stddev_multiplicity_grid = IC_flash_stddev_multiplicity_grouped.pivot(index='lat_bin', columns='lon_bin', values='numCloudPulses')
    IC_flash_stddev_multiplicity_grid = np.round(IC_flash_stddev_multiplicity_grid, 1)
    IC_flash_stddev_multiplicity_grid[IC_flash_stddev_multiplicity_grid.isna()] = 0


    '''
    Write outputs to NetCDF files:
    Three separate files are generated per day:
          (1) Flashes: includes flash counts and multiplicity
          (2) CG_strokes: includes CG stroke counts and peak current statistics
          (3) IC_strokes: includes IC stroke counts and peak current statistics
    '''


    '''
    Initialize Flashes NetCDF file
    '''
    flash_file_to_export = f'FLASHMAP_flash_daily_025deg_{day_range}.nc'
    # Create output directory if it does not exist
    directory = f'{OUT_DIR_NC}/FLASHES/{year}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    flash_file = Dataset(f'{OUT_DIR_NC}/FLASHES/{year}/{flash_file_to_export}', 'w')


    '''
    Add global attributes
    '''
    flash_file.title = f'FLASHMAP daily 0.25° flash dataset: {day_range}'
    flash_file.projection = 'EPSG:4326'
    flash_file.geospatial_lat_min = -90
    flash_file.geospatial_lat_max = 90
    flash_file.geospatial_lon_min = -180
    flash_file.geospatial_lon_max = 180
    flash_file.license = 'Creative Commons Attribution 4.0 International (CC BY 4.0)'
    flash_file.source = 'https://doi.org/10.5281/zenodo.17376946'
    flash_file.institution = 'Vrije Universiteit Amsterdam, NL; University of East Anglia, UK'
    flash_file.contact = 'y.qu@vu.nl; s.s.n.veraverbeke@vu.nl; matthew.w.jones@uea.ac.uk'


    '''
    Define latitude and longitude dimensions based on spatial resolution
    '''
    dim_lat = int(180 / resolution)
    dim_lon = int(360 / resolution)
    lat = flash_file.createDimension('lat', dim_lat)
    lon = flash_file.createDimension('lon', dim_lon)


    '''
    Create variables
    '''
    # LAT, LON
    lat = flash_file.createVariable('lat', 'f4', ('lat',), compression='zlib')
    lon = flash_file.createVariable('lon', 'f4', ('lon',), compression='zlib')

    # FLASH COUNT
    CG_flash_count = flash_file.createVariable('CG_flash_count', 'int32', ('lat', 'lon',), compression='zlib')
    IC_flash_count = flash_file.createVariable('IC_flash_count', 'int32', ('lat', 'lon',), compression='zlib')

    # FLASH Multiplicity
    IC_flash_multiplicity_mean = flash_file.createVariable('IC_flash_multiplicity_mean', 'f4', ('lat', 'lon',), compression='zlib')
    IC_flash_multiplicity_median = flash_file.createVariable('IC_flash_multiplicity_median', 'f4', ('lat', 'lon',), compression='zlib')
    IC_flash_multiplicity_stddev = flash_file.createVariable('IC_flash_multiplicity_stddev', 'f4', ('lat', 'lon',), compression='zlib')
    CG_flash_multiplicity_mean = flash_file.createVariable('CG_flash_multiplicity_mean', 'f4', ('lat', 'lon',), compression='zlib')
    CG_flash_multiplicity_median = flash_file.createVariable('CG_flash_multiplicity_median', 'f4', ('lat', 'lon',), compression='zlib')
    CG_flash_multiplicity_stddev = flash_file.createVariable('CG_flash_multiplicity_stddev', 'f4', ('lat', 'lon',), compression='zlib')


    '''
    Assign data to variables
    '''
    # LAT, LON
    lat[:] = np.arange(-90, 90, resolution)
    lon[:] = np.arange(-180, 180, resolution)

    # FLASH COUNT
    CG_flash_count[:, :] = CG_flash_count_grid
    IC_flash_count[:, :] = IC_flash_count_grid

    # Multiplicity
    IC_flash_multiplicity_mean[:, :] = IC_flash_mean_multiplicity_grid
    IC_flash_multiplicity_median[:, :] = IC_flash_median_multiplicity_grid
    IC_flash_multiplicity_stddev[:, :] = IC_flash_stddev_multiplicity_grid
    CG_flash_multiplicity_mean[:, :] = CG_flash_mean_multiplicity_grid
    CG_flash_multiplicity_median[:, :] = CG_flash_median_multiplicity_grid
    CG_flash_multiplicity_stddev[:, :] = CG_flash_stddev_multiplicity_grid


    '''
    Add metadata for all variables
    '''
    lat.unit = 'degrees north'
    lat.long_name = 'Latitude of lower-left grid cell corner'
    lon.unit = 'degrees east'
    lon.long_name = 'Longitude of lower-left grid cell corner'

    CG_flash_count.long_name = 'Number of CG flashes'
    CG_flash_count.coordinates = 'lat lon'
    IC_flash_count.long_name = 'Number of IC flashes'
    IC_flash_count.coordinates = 'lat lon'

    IC_flash_multiplicity_mean.long_name = 'Mean multiplicity of IC flashes'
    IC_flash_multiplicity_mean.coordinates = 'lat lon'
    IC_flash_multiplicity_median.long_name = 'Median multiplicity of IC flashes'
    IC_flash_multiplicity_median.coordinates = 'lat lon'
    IC_flash_multiplicity_stddev.long_name = 'Standard deviation of multiplicity of IC flashes'
    IC_flash_multiplicity_stddev.coordinates = 'lat lon'
    CG_flash_multiplicity_mean.long_name = 'Mean multiplicity of CG flashes'
    CG_flash_multiplicity_mean.coordinates = 'lat lon'
    CG_flash_multiplicity_median.long_name = 'Median multiplicity of CG flashes'
    CG_flash_multiplicity_median.coordinates = 'lat lon'
    CG_flash_multiplicity_stddev.long_name = 'Standard deviation of multiplicity of CG flashes'
    CG_flash_multiplicity_stddev.coordinates = 'lat lon'


    '''
    Close file
    '''
    flash_file.close()


    '''
    Initialize the CG stroke NetCDF file
    '''
    CG_file_to_export = f'FLASHMAP_CG_stroke_daily_025deg_{day_range}.nc'
    # Create output directory if it does not exist
    directory = f'{OUT_DIR_NC}/CG_STROKES/{year}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    CG_file = Dataset(f'{OUT_DIR_NC}/CG_STROKES/{year}/{CG_file_to_export}', 'w')


    '''
    Add global attributes
    '''
    CG_file.title = f'FLASHMAP daily 0.25° CG stroke dataset: {day_range}'
    CG_file.projection = 'EPSG:4326'
    CG_file.geospatial_lat_min = -90
    CG_file.geospatial_lat_max = 90
    CG_file.geospatial_lon_min = -180
    CG_file.geospatial_lon_max = 180
    CG_file.license = 'Creative Commons Attribution 4.0 International (CC BY 4.0)'
    CG_file.source = 'https://doi.org/10.5281/zenodo.17376946'
    CG_file.institution = 'Vrije Universiteit Amsterdam, NL; University of East Anglia, UK'
    CG_file.contact = 'y.qu@vu.nl; s.s.n.veraverbeke@vu.nl; matthew.w.jones@uea.ac.uk'


    '''
    Define latitude and longitude dimensions based on spatial resolution
    '''
    dim_lat = int(180 / resolution)
    dim_lon = int(360 / resolution)
    lat = CG_file.createDimension('lat', dim_lat)
    lon = CG_file.createDimension('lon', dim_lon)


    '''
    Create variables
    '''
    # LAT, LON
    lat = CG_file.createVariable('lat', 'f4', ('lat',), compression='zlib')
    lon = CG_file.createVariable('lon', 'f4', ('lon',), compression='zlib')

    # STROKE COUNTS
    CG_positive_stroke_count = CG_file.createVariable('CG_positive_stroke_count', 'int32', ('lat', 'lon',), compression='zlib')
    CG_negative_stroke_count = CG_file.createVariable('CG_negative_stroke_count', 'int32', ('lat', 'lon',), compression='zlib')

    # PEAK CURRENT
    CG_positive_peak_current_mean = CG_file.createVariable('CG_positive_peak_current_mean', 'f4', ('lat', 'lon',), compression='zlib')
    CG_positive_peak_current_median = CG_file.createVariable('CG_positive_peak_current_median', 'f4', ('lat', 'lon',), compression='zlib')
    CG_positive_peak_current_stddev = CG_file.createVariable('CG_positive_peak_current_stddev', 'f4', ('lat', 'lon',), compression='zlib')
    CG_negative_peak_current_mean = CG_file.createVariable('CG_negative_peak_current_mean', 'f4', ('lat', 'lon',), compression='zlib')
    CG_negative_peak_current_median = CG_file.createVariable('CG_negative_peak_current_median', 'f4', ('lat', 'lon',), compression='zlib')
    CG_negative_peak_current_stddev = CG_file.createVariable('CG_negative_peak_current_stddev', 'f4', ('lat', 'lon',), compression='zlib')


    '''
    Assign data to variables
    '''
    # LAT, LON
    lat[:] = np.arange(-90, 90, resolution)
    lon[:] = np.arange(-180, 180, resolution)

    # STROKE COUNTS
    CG_positive_stroke_count[:, :] = CG_positive_stroke_count_grid
    CG_negative_stroke_count[:, :] = CG_negative_stroke_count_grid

    # PEAK CURRENT
    CG_positive_peak_current_mean[:, :] = CG_positive_mean_peak_current_grid
    CG_positive_peak_current_median[:, :] = CG_positive_median_peak_current_grid
    CG_positive_peak_current_stddev[:, :] = CG_positive_stddev_peak_current_grid
    CG_negative_peak_current_mean[:, :] = CG_negative_mean_peak_current_grid
    CG_negative_peak_current_median[:, :] = CG_negative_median_peak_current_grid
    CG_negative_peak_current_stddev[:, :] = CG_negative_stddev_peak_current_grid


    '''
    Add metadata for all variables
    '''
    lat.unit = 'degrees north'
    lat.long_name = 'Latitude of lower-left grid cell corner'
    lon.unit = 'degrees east'
    lon.long_name = 'Longitude of lower-left grid cell corner'

    CG_positive_stroke_count.long_name = 'Number of CG strokes with positive polarity'
    CG_positive_stroke_count.coordinates = 'lat lon'
    CG_negative_stroke_count.long_name = 'Number of CG strokes with negative polarity'
    CG_negative_stroke_count.coordinates = 'lat lon'

    CG_positive_peak_current_mean.unit = 'kA'
    CG_positive_peak_current_mean.long_name = 'Mean peak current of CG strokes with positive polarity'
    CG_positive_peak_current_mean.coordinates = 'lat lon'
    CG_positive_peak_current_median.unit = 'kA'
    CG_positive_peak_current_median.long_name = 'Median peak current of CG strokes with positive polarity'
    CG_positive_peak_current_median.coordinates = 'lat lon'
    CG_positive_peak_current_stddev.unit = 'kA'
    CG_positive_peak_current_stddev.long_name = 'Standard deviation of peak current of CG strokes with positive polarity'
    CG_positive_peak_current_stddev.coordinates = 'lat lon'

    CG_negative_peak_current_mean.unit = 'kA'
    CG_negative_peak_current_mean.long_name = 'Mean peak current of CG strokes with negative polarity'
    CG_negative_peak_current_mean.coordinates = 'lat lon'
    CG_negative_peak_current_median.unit = 'kA'
    CG_negative_peak_current_median.long_name = 'Median peak current of CG strokes with negative polarity'
    CG_negative_peak_current_median.coordinates = 'lat lon'
    CG_negative_peak_current_stddev.unit = 'kA'
    CG_negative_peak_current_stddev.long_name = 'Standard deviation of peak current of CG strokes with negative polarity'
    CG_negative_peak_current_stddev.coordinates = 'lat lon'


    '''
    Close file
    '''
    CG_file.close()


    '''
    Initialize IC stroke NetCDF file
    '''
    IC_file_to_export = f'FLASHMAP_IC_stroke_daily_025deg_{day_range}.nc'
    # Create output directory if it does not exist
    directory = f'{OUT_DIR_NC}/IC_STROKES/{year}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    IC_file = Dataset(f'{OUT_DIR_NC}/IC_STROKES/{year}/{IC_file_to_export}', 'w')


    '''
    Add global attributes
    '''
    IC_file.title = f'FLASHMAP daily 0.25° IC stroke dataset: {day_range}'
    # More to be added
    IC_file.projection = 'EPSG:4326'
    IC_file.geospatial_lat_min = -90
    IC_file.geospatial_lat_max = 90
    IC_file.geospatial_lon_min = -180
    IC_file.geospatial_lon_max = 180
    IC_file.license = 'Creative Commons Attribution 4.0 International (CC BY 4.0)'
    IC_file.source = 'https://doi.org/10.5281/zenodo.17376946'
    IC_file.institution = 'Vrije Universiteit Amsterdam, NL; University of East Anglia, UK'
    IC_file.contact = 'y.qu@vu.nl; s.s.n.veraverbeke@vu.nl; matthew.w.jones@uea.ac.uk'


    '''
    Define latitude and longitude dimensions based on spatial resolution
    '''
    dim_lat = int(180 / resolution)
    dim_lon = int(360 / resolution)
    lat = IC_file.createDimension('lat', dim_lat)
    lon = IC_file.createDimension('lon', dim_lon)


    '''
    Create variables
    '''
    # LAT, LON
    lat = IC_file.createVariable('lat', 'f4', ('lat',), compression='zlib')
    lon = IC_file.createVariable('lon', 'f4', ('lon',), compression='zlib')

    # STROKE COUNTS
    IC_positive_stroke_count = IC_file.createVariable('IC_positive_stroke_count', 'int32', ('lat', 'lon',), compression='zlib')
    IC_negative_stroke_count = IC_file.createVariable('IC_negative_stroke_count', 'int32', ('lat', 'lon',), compression='zlib')

    # PEAK CURRENT
    IC_positive_peak_current_mean = IC_file.createVariable('IC_positive_peak_current_mean', 'f4', ('lat', 'lon',), compression='zlib')
    IC_positive_peak_current_median = IC_file.createVariable('IC_positive_peak_current_median', 'f4', ('lat', 'lon',), compression='zlib')
    IC_positive_peak_current_stddev = IC_file.createVariable('IC_positive_peak_current_stddev', 'f4', ('lat', 'lon',), compression='zlib')
    IC_negative_peak_current_mean = IC_file.createVariable('IC_negative_peak_current_mean', 'f4', ('lat', 'lon',), compression='zlib')
    IC_negative_peak_current_median = IC_file.createVariable('IC_negative_peak_current_median', 'f4', ('lat', 'lon',), compression='zlib')
    IC_negative_peak_current_stddev = IC_file.createVariable('IC_negative_peak_current_stddev', 'f4', ('lat', 'lon',), compression='zlib')


    '''
    Assign data to variables
    '''
    # LAT, LON
    lat[:] = np.arange(-90, 90, resolution)
    lon[:] = np.arange(-180, 180, resolution)

    # STROKE COUNTS
    IC_positive_stroke_count[:, :] = IC_positive_stroke_count_grid
    IC_negative_stroke_count[:, :] = IC_negative_stroke_count_grid

    # PEAK CURRENT
    IC_positive_peak_current_mean[:, :] = IC_positive_mean_peak_current_grid
    IC_positive_peak_current_median[:, :] = IC_positive_median_peak_current_grid
    IC_positive_peak_current_stddev[:, :] = IC_positive_stddev_peak_current_grid
    IC_negative_peak_current_mean[:, :] = IC_negative_mean_peak_current_grid
    IC_negative_peak_current_median[:, :] = IC_negative_median_peak_current_grid
    IC_negative_peak_current_stddev[:, :] = IC_negative_stddev_peak_current_grid


    '''
    Add metadata for all variables
    '''
    lat.unit = 'degrees north'
    lat.long_name = 'Latitude of lower-left grid cell corner'
    lon.unit = 'degrees east'
    lon.long_name = 'Longitude of lower-left grid cell corner'

    IC_positive_stroke_count.long_name = 'Number of IC strokes with positive polarity'
    IC_positive_stroke_count.coordinates = 'lat lon'
    IC_negative_stroke_count.long_name = 'Number of IC strokes with negative polarity'
    IC_negative_stroke_count.coordinates = 'lat lon'

    IC_positive_peak_current_mean.unit = 'kA'
    IC_positive_peak_current_mean.long_name = 'Mean peak current of IC strokes with positive polarity'
    IC_positive_peak_current_mean.coordinates = 'lat lon'
    IC_positive_peak_current_median.unit = 'kA'
    IC_positive_peak_current_median.long_name = 'Median peak current of IC strokes with positive polarity'
    IC_positive_peak_current_median.coordinates = 'lat lon'
    IC_positive_peak_current_stddev.unit = 'kA'
    IC_positive_peak_current_stddev.long_name = 'Standard deviation of peak current of IC strokes with positive polarity'
    IC_positive_peak_current_stddev.coordinates = 'lat lon'

    IC_negative_peak_current_mean.unit = 'kA'
    IC_negative_peak_current_mean.long_name = 'Mean peak current of IC strokes with negative polarity'
    IC_negative_peak_current_mean.coordinates = 'lat lon'
    IC_negative_peak_current_median.unit = 'kA'
    IC_negative_peak_current_median.long_name = 'Median peak current of IC strokes with negative polarity'
    IC_negative_peak_current_median.coordinates = 'lat lon'
    IC_negative_peak_current_stddev.unit = 'kA'
    IC_negative_peak_current_stddev.long_name = 'Standard deviation of peak current of IC strokes with negative polarity'
    IC_negative_peak_current_stddev.coordinates = 'lat lon'


    '''
    Close file
    '''
    IC_file.close()


def main():
    day_range = (pd.date_range(start=str(start_day), end=str(end_day), freq='D')).strftime('%Y_%m_%d').tolist()
    # Use multiprocessing Pool to run the loop in parallel
    n_cores = 144
    pool = multiprocessing.Pool(processes=n_cores)
    pool.map(GLD360_vector_to_daily_grid, day_range)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
