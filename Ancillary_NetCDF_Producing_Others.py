import os
import pandas as pd
import numpy as np
import multiprocessing
from netCDF4 import Dataset

'''Produce FLASHMAP Ancillary Data NetCDF Files'''

# Month range and spatial resolution
year = 2025
start_month = '2025-01'
end_month = '2025-12'
resolution = 0.25  # degrees

# Input directories
IN_DIR_PAR = f'/projects/0/prjs1409/GLD360_Parquets/Gap_Filled_Data/{year}'
# Output directories
OUT_DIR_NC = f'/gpfs/work2/0/einf3869/GLD360/Integrator_era/FLASHMAP/Ancillary_NetCDFs'


def GLD360_vector_to_monthly_grid(month_range, resolution=resolution, IN_DIR_PAR=IN_DIR_PAR, OUT_DIR_NC=OUT_DIR_NC):
    """
    Generate monthly gridded ancillary NetCDF from the GLD360 data.

    Parameters
    ----------
    month_range : str
        Month string in the format 'YYYY-MM'.
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
    Read GLD360 parquet files for a given month.
    '''
    year = int(month_range[0:4])
    month = int(month_range[5:])

    # Create list of daily filenames for the month
    if month != 12:
        day_range = pd.date_range(str(year) + '-' + str(month), str(year) + '-' + str(month + 1), freq='D', inclusive='left').strftime('%Y_%m_%d').tolist()
    else:
        day_range = pd.date_range(str(year) + '-' + str(month), str(year + 1) + '-01', freq='D', inclusive='left').strftime('%Y_%m_%d').tolist()

    # Build a list of daily file paths for the given month
    fnames = []
    for day in day_range:
        fname = f'{IN_DIR_PAR}/GLD360_{day}.parquet'
        fnames.append(fname)

    # Read and concatenate all existing parquet files for the month
    df_list = []
    for file in fnames:
        if os.path.exists(file):
            df_list.append(pd.read_parquet(file))

    # Combine all daily data into one DataFrame for the month
    GLD360_data = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    GLD360_data['abssignalStrengthKA'] = np.abs(GLD360_data['signalStrengthKA'])

    '''
    Separate strokes by lightning type and peak current ranges.
    '''
    GLD360_0_10kA_data = GLD360_data[(GLD360_data['abssignalStrengthKA'] > 0) & (GLD360_data['abssignalStrengthKA'] <= 10)].copy()
    GLD360_10_20kA_data = GLD360_data[(GLD360_data['abssignalStrengthKA'] > 10) & (GLD360_data['abssignalStrengthKA'] <= 20)].copy()
    GLD360_20_30kA_data = GLD360_data[(GLD360_data['abssignalStrengthKA'] > 20) & (GLD360_data['abssignalStrengthKA'] <= 30)].copy()
    GLD360_30kA_plus_data = GLD360_data[GLD360_data['abssignalStrengthKA'] > 30].copy()

    CG_data = GLD360_data[GLD360_data['cloud'] == False].copy()
    IC_data = GLD360_data[GLD360_data['cloud'] == True].copy()

    CG_0_10kA_data = CG_data[(CG_data['abssignalStrengthKA'] > 0) & (CG_data['abssignalStrengthKA'] <= 10)].copy()
    CG_10_20kA_data = CG_data[(CG_data['abssignalStrengthKA'] > 10) & (CG_data['abssignalStrengthKA'] <= 20)].copy()
    CG_20_30kA_data = CG_data[(CG_data['abssignalStrengthKA'] > 20) & (CG_data['abssignalStrengthKA'] <= 30)].copy()
    CG_30kA_plus_data = CG_data[CG_data['abssignalStrengthKA'] > 30].copy()
    IC_0_10kA_data = IC_data[(IC_data['abssignalStrengthKA'] > 0) & (IC_data['abssignalStrengthKA'] <= 10)].copy()
    IC_10_20kA_data = IC_data[(IC_data['abssignalStrengthKA'] > 10) & (IC_data['abssignalStrengthKA'] <= 20)].copy()
    IC_20_30kA_data = IC_data[(IC_data['abssignalStrengthKA'] > 20) & (IC_data['abssignalStrengthKA'] <= 30)].copy()
    IC_30kA_plus_data = IC_data[IC_data['abssignalStrengthKA'] > 30].copy()

    '''
    Define the gridding structure.
    Latitude and longitude bins are generated based on the specified spatial resolution.
    Each lightning event will be assigned to a corresponding grid cell.
    '''
    lat_bins = pd.interval_range(-90, 90, freq=resolution)
    lon_bins = pd.interval_range(-180, 180, freq=resolution)

    GLD360_data['lat_bin'] = pd.cut(GLD360_data['latitude'], bins=lat_bins)
    GLD360_data['lon_bin'] = pd.cut(GLD360_data['longitude'], bins=lon_bins)

    GLD360_0_10kA_data['lat_bin'] = pd.cut(GLD360_0_10kA_data['latitude'], bins=lat_bins)
    GLD360_0_10kA_data['lon_bin'] = pd.cut(GLD360_0_10kA_data['longitude'], bins=lon_bins)
    GLD360_10_20kA_data['lat_bin'] = pd.cut(GLD360_10_20kA_data['latitude'], bins=lat_bins)
    GLD360_10_20kA_data['lon_bin'] = pd.cut(GLD360_10_20kA_data['longitude'], bins=lon_bins)
    GLD360_20_30kA_data['lat_bin'] = pd.cut(GLD360_20_30kA_data['latitude'], bins=lat_bins)
    GLD360_20_30kA_data['lon_bin'] = pd.cut(GLD360_20_30kA_data['longitude'], bins=lon_bins)
    GLD360_30kA_plus_data['lat_bin'] = pd.cut(GLD360_30kA_plus_data['latitude'], bins=lat_bins)
    GLD360_30kA_plus_data['lon_bin'] = pd.cut(GLD360_30kA_plus_data['longitude'], bins=lon_bins)

    CG_0_10kA_data['lat_bin'] = pd.cut(CG_0_10kA_data['latitude'], bins=lat_bins)
    CG_0_10kA_data['lon_bin'] = pd.cut(CG_0_10kA_data['longitude'], bins=lon_bins)
    CG_10_20kA_data['lat_bin'] = pd.cut(CG_10_20kA_data['latitude'], bins=lat_bins)
    CG_10_20kA_data['lon_bin'] = pd.cut(CG_10_20kA_data['longitude'], bins=lon_bins)
    CG_20_30kA_data['lat_bin'] = pd.cut(CG_20_30kA_data['latitude'], bins=lat_bins)
    CG_20_30kA_data['lon_bin'] = pd.cut(CG_20_30kA_data['longitude'], bins=lon_bins)
    CG_30kA_plus_data['lat_bin'] = pd.cut(CG_30kA_plus_data['latitude'], bins=lat_bins)
    CG_30kA_plus_data['lon_bin'] = pd.cut(CG_30kA_plus_data['longitude'], bins=lon_bins)

    IC_0_10kA_data['lat_bin'] = pd.cut(IC_0_10kA_data['latitude'], bins=lat_bins)
    IC_0_10kA_data['lon_bin'] = pd.cut(IC_0_10kA_data['longitude'], bins=lon_bins)
    IC_10_20kA_data['lat_bin'] = pd.cut(IC_10_20kA_data['latitude'], bins=lat_bins)
    IC_10_20kA_data['lon_bin'] = pd.cut(IC_10_20kA_data['longitude'], bins=lon_bins)
    IC_20_30kA_data['lat_bin'] = pd.cut(IC_20_30kA_data['latitude'], bins=lat_bins)
    IC_20_30kA_data['lon_bin'] = pd.cut(IC_20_30kA_data['longitude'], bins=lon_bins)
    IC_30kA_plus_data['lat_bin'] = pd.cut(IC_30kA_plus_data['latitude'], bins=lat_bins)
    IC_30kA_plus_data['lon_bin'] = pd.cut(IC_30kA_plus_data['longitude'], bins=lon_bins)


    '''
    50% confidence ellipses median
    '''
    CG_0_10kA_ellipse_median_grouped = CG_0_10kA_data.groupby(['lat_bin', 'lon_bin'], observed=False)['ellSemiMajM'].quantile(0.5).reset_index()
    CG_0_10kA_ellipse_median_grid = CG_0_10kA_ellipse_median_grouped.pivot(index='lat_bin', columns='lon_bin', values='ellSemiMajM')
    CG_0_10kA_ellipse_median_grid = np.round(CG_0_10kA_ellipse_median_grid, 1)
    CG_0_10kA_ellipse_median_grid[CG_0_10kA_ellipse_median_grid.isna()] = 0

    CG_10_20kA_ellipse_median_grouped = CG_10_20kA_data.groupby(['lat_bin', 'lon_bin'], observed=False)['ellSemiMajM'].quantile(0.5).reset_index()
    CG_10_20kA_ellipse_median_grid = CG_10_20kA_ellipse_median_grouped.pivot(index='lat_bin', columns='lon_bin', values='ellSemiMajM')
    CG_10_20kA_ellipse_median_grid = np.round(CG_10_20kA_ellipse_median_grid, 1)
    CG_10_20kA_ellipse_median_grid[CG_10_20kA_ellipse_median_grid.isna()] = 0

    CG_20_30kA_ellipse_median_grouped = CG_20_30kA_data.groupby(['lat_bin', 'lon_bin'], observed=False)['ellSemiMajM'].quantile(0.5).reset_index()
    CG_20_30kA_ellipse_median_grid = CG_20_30kA_ellipse_median_grouped.pivot(index='lat_bin', columns='lon_bin', values='ellSemiMajM')
    CG_20_30kA_ellipse_median_grid = np.round(CG_20_30kA_ellipse_median_grid, 1)
    CG_20_30kA_ellipse_median_grid[CG_20_30kA_ellipse_median_grid.isna()] = 0

    CG_30kA_plus_ellipse_median_grouped = CG_30kA_plus_data.groupby(['lat_bin', 'lon_bin'], observed=False)['ellSemiMajM'].quantile(0.5).reset_index()
    CG_30kA_plus_ellipse_median_grid = CG_30kA_plus_ellipse_median_grouped.pivot(index='lat_bin', columns='lon_bin', values='ellSemiMajM')
    CG_30kA_plus_ellipse_median_grid = np.round(CG_30kA_plus_ellipse_median_grid, 1)
    CG_30kA_plus_ellipse_median_grid[CG_30kA_plus_ellipse_median_grid.isna()] = 0

    IC_0_10kA_ellipse_median_grouped = IC_0_10kA_data.groupby(['lat_bin', 'lon_bin'], observed=False)['ellSemiMajM'].quantile(0.5).reset_index()
    IC_0_10kA_ellipse_median_grid = IC_0_10kA_ellipse_median_grouped.pivot(index='lat_bin', columns='lon_bin', values='ellSemiMajM')
    IC_0_10kA_ellipse_median_grid = np.round(IC_0_10kA_ellipse_median_grid, 1)
    IC_0_10kA_ellipse_median_grid[IC_0_10kA_ellipse_median_grid.isna()] = 0

    IC_10_20kA_ellipse_median_grouped = IC_10_20kA_data.groupby(['lat_bin', 'lon_bin'], observed=False)['ellSemiMajM'].quantile(0.5).reset_index()
    IC_10_20kA_ellipse_median_grid = IC_10_20kA_ellipse_median_grouped.pivot(index='lat_bin', columns='lon_bin', values='ellSemiMajM')
    IC_10_20kA_ellipse_median_grid = np.round(IC_10_20kA_ellipse_median_grid, 1)
    IC_10_20kA_ellipse_median_grid[IC_10_20kA_ellipse_median_grid.isna()] = 0

    IC_20_30kA_ellipse_median_grouped = IC_20_30kA_data.groupby(['lat_bin', 'lon_bin'], observed=False)['ellSemiMajM'].quantile(0.5).reset_index()
    IC_20_30kA_ellipse_median_grid = IC_20_30kA_ellipse_median_grouped.pivot(index='lat_bin', columns='lon_bin', values='ellSemiMajM')
    IC_20_30kA_ellipse_median_grid = np.round(IC_20_30kA_ellipse_median_grid, 1)
    IC_20_30kA_ellipse_median_grid[IC_20_30kA_ellipse_median_grid.isna()] = 0

    IC_30kA_plus_ellipse_median_grouped = IC_30kA_plus_data.groupby(['lat_bin', 'lon_bin'], observed=False)['ellSemiMajM'].quantile(0.5).reset_index()
    IC_30kA_plus_ellipse_median_grid = IC_30kA_plus_ellipse_median_grouped.pivot(index='lat_bin', columns='lon_bin', values='ellSemiMajM')
    IC_30kA_plus_ellipse_median_grid = np.round(IC_30kA_plus_ellipse_median_grid, 1)
    IC_30kA_plus_ellipse_median_grid[IC_30kA_plus_ellipse_median_grid.isna()] = 0

    '''
    5th percentile total lightning signal strength
    '''
    GLD360_5th_percentile_grouped = GLD360_data.groupby(['lat_bin', 'lon_bin'], observed=False)['abssignalStrengthKA'].quantile(0.05).reset_index()
    GLD360_5th_percentile_grid = GLD360_5th_percentile_grouped.pivot(index='lat_bin', columns='lon_bin', values='abssignalStrengthKA')
    GLD360_5th_percentile_grid = np.round(GLD360_5th_percentile_grid, 1)
    GLD360_5th_percentile_grid[GLD360_5th_percentile_grouped.isna()] = 0

    '''
    95th percentile total lightning signal strength
    '''
    GLD360_95th_percentile_grouped = GLD360_data.groupby(['lat_bin', 'lon_bin'], observed=False)['abssignalStrengthKA'].quantile(0.95).reset_index()
    GLD360_95th_percentile_grid = GLD360_95th_percentile_grouped.pivot(index='lat_bin', columns='lon_bin', values='abssignalStrengthKA')
    GLD360_95th_percentile_grid = np.round(GLD360_95th_percentile_grid, 1)
    GLD360_95th_percentile_grid[GLD360_95th_percentile_grouped.isna()] = 0

    '''
    Number of sensor reports median
    '''
    GLD360_0_10kA_sensors_median_grouped = GLD360_0_10kA_data.groupby(['lat_bin', 'lon_bin'], observed=False)['numSensors'].quantile(0.5).reset_index()
    GLD360_0_10kA_sensors_median_grid = GLD360_0_10kA_sensors_median_grouped.pivot(index='lat_bin', columns='lon_bin', values='numSensors')
    GLD360_0_10kA_sensors_median_grid = np.round(GLD360_0_10kA_sensors_median_grid, 1)
    GLD360_0_10kA_sensors_median_grid[GLD360_0_10kA_sensors_median_grid.isna()] = 0

    GLD360_10_20kA_sensors_median_grouped = GLD360_10_20kA_data.groupby(['lat_bin', 'lon_bin'], observed=False)['numSensors'].quantile(0.5).reset_index()
    GLD360_10_20kA_sensors_median_grid = GLD360_10_20kA_sensors_median_grouped.pivot(index='lat_bin', columns='lon_bin', values='numSensors')
    GLD360_10_20kA_sensors_median_grid = np.round(GLD360_10_20kA_sensors_median_grid, 1)
    GLD360_10_20kA_sensors_median_grid[GLD360_10_20kA_sensors_median_grid.isna()] = 0

    GLD360_20_30kA_sensors_median_grouped = GLD360_20_30kA_data.groupby(['lat_bin', 'lon_bin'], observed=False)['numSensors'].quantile(0.5).reset_index()
    GLD360_20_30kA_sensors_median_grid = GLD360_20_30kA_sensors_median_grouped.pivot(index='lat_bin', columns='lon_bin', values='numSensors')
    GLD360_20_30kA_sensors_median_grid = np.round(GLD360_20_30kA_sensors_median_grid, 1)
    GLD360_20_30kA_sensors_median_grid[GLD360_20_30kA_sensors_median_grid.isna()] = 0

    GLD360_30kA_plus_sensors_median_grouped = GLD360_30kA_plus_data.groupby(['lat_bin', 'lon_bin'], observed=False)['numSensors'].quantile(0.5).reset_index()
    GLD360_30kA_plus_sensors_median_grid = GLD360_30kA_plus_sensors_median_grouped.pivot(index='lat_bin', columns='lon_bin', values='numSensors')
    GLD360_30kA_plus_sensors_median_grid = np.round(GLD360_30kA_plus_sensors_median_grid, 1)
    GLD360_30kA_plus_sensors_median_grid[GLD360_30kA_plus_sensors_median_grid.isna()] = 0

    '''
    Write outputs to NetCDF files:
    Three separate files are generated per day:
          (1) Location accuracy
          (2) Peak current percentiles
          (3) Number of sensor reports
    '''

    '''
    Initialize location accuracy NetCDF file
    '''
    file_to_export = f'Location_accuracy_monthly_025deg_{month_range}.nc'
    # Create output directory if it does not exist
    directory = f'{OUT_DIR_NC}/Location_accuracy/{year}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    LA_file = Dataset(f'{OUT_DIR_NC}/Location_accuracy/{year}/{file_to_export}', 'w')

    '''
    Add global attributes
    '''
    LA_file.title = f'FLASHMAP ancillary data: monthly 0.25° location accuracy ({month_range})'
    LA_file.projection = 'EPSG:4326'
    LA_file.geospatial_lat_min = -90
    LA_file.geospatial_lat_max = 90
    LA_file.geospatial_lon_min = -180
    LA_file.geospatial_lon_max = 180
    LA_file.license = 'Creative Commons Attribution 4.0 International (CC BY 4.0)'
    LA_file.source = 'https://doi.org/10.5281/zenodo.17376946'
    LA_file.institution = 'Vrije Universiteit Amsterdam, NL; University of East Anglia, UK'
    LA_file.contact = 'y.qu@vu.nl; s.s.n.veraverbeke@vu.nl; matthew.w.jones@uea.ac.uk'

    '''
    Define latitude and longitude dimensions based on spatial resolution
    '''
    dim_lat = int(180 / resolution)
    dim_lon = int(360 / resolution)
    lat = LA_file.createDimension('lat', dim_lat)
    lon = LA_file.createDimension('lon', dim_lon)

    '''
    Create variables
    '''
    # LAT, LON
    lat = LA_file.createVariable('lat', 'f4', ('lat',), compression='zlib')
    lon = LA_file.createVariable('lon', 'f4', ('lon',), compression='zlib')

    # 50% ELLIPSE MEDIAN
    CG_0_10kA_ellipse_median = LA_file.createVariable('CG_LA_0_10kA_median', 'f4', ('lat', 'lon',), compression='zlib')
    CG_10_20kA_ellipse_median = LA_file.createVariable('CG_LA_10_20kA_median', 'f4', ('lat', 'lon',), compression='zlib')
    CG_20_30kA_ellipse_median = LA_file.createVariable('CG_LA_20_30kA_median', 'f4', ('lat', 'lon',), compression='zlib')
    CG_30kA_plus_ellipse_median = LA_file.createVariable('CG_LA_30kA_plus_median', 'f4', ('lat', 'lon',), compression='zlib')
    IC_0_10kA_ellipse_median = LA_file.createVariable('IC_LA_0_10kA_median', 'f4', ('lat', 'lon',), compression='zlib')
    IC_10_20kA_ellipse_median = LA_file.createVariable('IC_LA_10_20kA_median', 'f4', ('lat', 'lon',), compression='zlib')
    IC_20_30kA_ellipse_median = LA_file.createVariable('IC_LA_20_30kA_median', 'f4', ('lat', 'lon',), compression='zlib')
    IC_30kA_plus_ellipse_median = LA_file.createVariable('IC_LA_30kA_plus_median', 'f4', ('lat', 'lon',), compression='zlib')

    '''
    Assign data to variables
    '''
    # LAT, LON
    lat[:] = np.arange(-90, 90, resolution)
    lon[:] = np.arange(-180, 180, resolution)

    # 50% ELLIPSE MEDIAN
    CG_0_10kA_ellipse_median[:, :] = CG_0_10kA_ellipse_median_grid
    CG_10_20kA_ellipse_median[:, :] = CG_10_20kA_ellipse_median_grid
    CG_20_30kA_ellipse_median[:, :] = CG_20_30kA_ellipse_median_grid
    CG_30kA_plus_ellipse_median[:, :] = CG_30kA_plus_ellipse_median_grid
    IC_0_10kA_ellipse_median[:, :] = IC_0_10kA_ellipse_median_grid
    IC_10_20kA_ellipse_median[:, :] = IC_10_20kA_ellipse_median_grid
    IC_20_30kA_ellipse_median[:, :] = IC_20_30kA_ellipse_median_grid
    IC_30kA_plus_ellipse_median[:, :] = IC_30kA_plus_ellipse_median_grid

    '''
    Add metadata for all variables
    '''
    lat.unit = 'degrees north'
    lat.long_name = 'Latitude of lower-left grid cell corner'
    lon.unit = 'degrees east'
    lon.long_name = 'Longitude of lower-left grid cell corner'

    CG_0_10kA_ellipse_median.unit = 'm'
    CG_0_10kA_ellipse_median.long_name = 'Median location accuracy of CG strokes with absolute peak current between 0–10 kA'
    CG_0_10kA_ellipse_median.coordinates = 'lat lon'

    CG_10_20kA_ellipse_median.unit = 'm'
    CG_10_20kA_ellipse_median.long_name = 'Median location accuracy of CG strokes with absolute peak current between 10–20 kA'
    CG_10_20kA_ellipse_median.coordinates = 'lat lon'

    CG_20_30kA_ellipse_median.unit = 'm'
    CG_20_30kA_ellipse_median.long_name = 'Median location accuracy of CG strokes with absolute peak current between 20–30 kA'
    CG_20_30kA_ellipse_median.coordinates = 'lat lon'

    CG_30kA_plus_ellipse_median.unit = 'm'
    CG_30kA_plus_ellipse_median.long_name = 'Median location accuracy of CG strokes with absolute peak current above 30 kA'
    CG_30kA_plus_ellipse_median.coordinates = 'lat lon'

    IC_0_10kA_ellipse_median.unit = 'm'
    IC_0_10kA_ellipse_median.long_name = 'Median location accuracy of IC strokes with absolute peak current between 0–10 kA'
    IC_0_10kA_ellipse_median.coordinates = 'lat lon'

    IC_10_20kA_ellipse_median.unit = 'm'
    IC_10_20kA_ellipse_median.long_name = 'Median location accuracy of IC strokes with absolute peak current between 10–20 kA'
    IC_10_20kA_ellipse_median.coordinates = 'lat lon'

    IC_20_30kA_ellipse_median.unit = 'm'
    IC_20_30kA_ellipse_median.long_name = 'Median location accuracy of IC strokes with absolute peak current between 20–30 kA'
    IC_20_30kA_ellipse_median.coordinates = 'lat lon'

    IC_30kA_plus_ellipse_median.unit = 'm'
    IC_30kA_plus_ellipse_median.long_name = 'Median location accuracy of IC strokes with absolute peak current above 30 kA'
    IC_30kA_plus_ellipse_median.coordinates = 'lat lon'

    '''
    Close file
    '''
    LA_file.close()

    '''
    Initialize peak current percentile NetCDF file
    '''
    file_to_export = f'Peak_current_percentile_monthly_025deg_{month_range}.nc'
    # Create output directory if it does not exist
    directory = f'{OUT_DIR_NC}/Peak_current_percentile/{year}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    PC_file = Dataset(f'{OUT_DIR_NC}/Peak_current_percentile/{year}/{file_to_export}', 'w')

    '''
    Add global attributes
    '''
    PC_file.title = f'FLASHMAP ancillary data: monthly 0.25° peak current percentile ({month_range})'
    PC_file.projection = 'EPSG:4326'
    PC_file.geospatial_lat_min = -90
    PC_file.geospatial_lat_max = 90
    PC_file.geospatial_lon_min = -180
    PC_file.geospatial_lon_max = 180
    PC_file.license = 'Creative Commons Attribution 4.0 International (CC BY 4.0)'
    PC_file.source = 'https://doi.org/10.5281/zenodo.17376946'
    PC_file.institution = 'Vrije Universiteit Amsterdam, NL; University of East Anglia, UK'
    PC_file.contact = 'y.qu@vu.nl; s.s.n.veraverbeke@vu.nl; matthew.w.jones@uea.ac.uk'

    '''
    Define latitude and longitude dimensions based on spatial resolution
    '''
    dim_lat = int(180 / resolution)
    dim_lon = int(360 / resolution)
    lat = PC_file.createDimension('lat', dim_lat)
    lon = PC_file.createDimension('lon', dim_lon)

    '''
    Create variables
    '''
    # LAT, LON
    lat = PC_file.createVariable('lat', 'f4', ('lat',), compression='zlib')
    lon = PC_file.createVariable('lon', 'f4', ('lon',), compression='zlib')

    # 5th, 95th PEAK CURRENT PERCENTILE
    Total_5th_percentile = PC_file.createVariable('total_5th_percentile_peak_current', 'f4', ('lat', 'lon',), compression='zlib')
    Total_95th_percentile = PC_file.createVariable('total_95th_percentile_peak_current', 'f4', ('lat', 'lon',), compression='zlib')

    '''
    Assign data to variables
    '''
    # LAT, LON
    lat[:] = np.arange(-90, 90, resolution)
    lon[:] = np.arange(-180, 180, resolution)

    # 5th, 95th PEAK CURRENT PERCENTILE
    Total_5th_percentile[:, :] = GLD360_5th_percentile_grid
    Total_95th_percentile[:, :] = GLD360_95th_percentile_grid

    '''
    Add metadata for all variables
    '''
    lat.unit = 'degrees north'
    lat.long_name = 'Latitude of lower-left grid cell corner'
    lon.unit = 'degrees east'
    lon.long_name = 'Longitude of lower-left grid cell corner'

    Total_5th_percentile.unit = 'kA'
    Total_5th_percentile.long_name = 'The 5th percentile absolute peak current of total strokes'
    Total_5th_percentile.coordinates = 'lat lon'

    Total_95th_percentile.unit = 'kA'
    Total_95th_percentile.long_name = 'The 95th percentile absolute peak current of total strokes'
    Total_95th_percentile.coordinates = 'lat lon'

    '''
    Close file
    '''
    PC_file.close()

    '''
    Initialize number of sensor report NetCDF file
    '''
    file_to_export = f'Number_of_sensor_reports_monthly_025deg_{month_range}.nc'
    # Create output directory if it does not exist
    directory = f'{OUT_DIR_NC}/Number_of_sensor_reports/{year}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    SR_file = Dataset(f'{OUT_DIR_NC}/Number_of_sensor_reports/{year}/{file_to_export}', 'w')

    '''
    Add global attributes
    '''
    SR_file.title = f'FLASHMAP ancillary data: monthly 0.25° number of sensor reports ({month_range})'
    SR_file.projection = 'EPSG:4326'
    SR_file.geospatial_lat_min = -90
    SR_file.geospatial_lat_max = 90
    SR_file.geospatial_lon_min = -180
    SR_file.geospatial_lon_max = 180
    SR_file.license = 'Creative Commons Attribution 4.0 International (CC BY 4.0)'
    SR_file.source = 'https://doi.org/10.5281/zenodo.17376946'
    SR_file.institution = 'Vrije Universiteit Amsterdam, NL; University of East Anglia, UK'
    SR_file.contact = 'y.qu@vu.nl; s.s.n.veraverbeke@vu.nl; matthew.w.jones@uea.ac.uk'

    '''
    Define latitude and longitude dimensions based on spatial resolution
    '''
    dim_lat = int(180 / resolution)
    dim_lon = int(360 / resolution)
    lat = SR_file.createDimension('lat', dim_lat)
    lon = SR_file.createDimension('lon', dim_lon)

    '''
    Create variables
    '''
    # LAT, LON
    lat = SR_file.createVariable('lat', 'f4', ('lat',), compression='zlib')
    lon = SR_file.createVariable('lon', 'f4', ('lon',), compression='zlib')

    # NUMBER OF SENSOR REPORTS MEDIAN
    Total_0_10kA_sensors_median = SR_file.createVariable('total_NSR_0_10kA_median', 'f4', ('lat', 'lon',), compression='zlib')
    Total_10_20kA_sensors_median = SR_file.createVariable('total_NSR_10_20kA_median', 'f4', ('lat', 'lon',), compression='zlib')
    Total_20_30kA_sensors_median = SR_file.createVariable('total_NSR_20_30kA_median', 'f4', ('lat', 'lon',), compression='zlib')
    Total_30kA_plus_sensors_median = SR_file.createVariable('total_NSR_30kA_plus_median', 'f4', ('lat', 'lon',), compression='zlib')

    '''
    Assign data to variables
    '''
    # LAT, LON
    lat[:] = np.arange(-90, 90, resolution)
    lon[:] = np.arange(-180, 180, resolution)

    # NUMBER OF SENSOR REPORTS MEDIAN
    Total_0_10kA_sensors_median[:, :] = GLD360_0_10kA_sensors_median_grid
    Total_10_20kA_sensors_median[:, :] = GLD360_10_20kA_sensors_median_grid
    Total_20_30kA_sensors_median[:, :] = GLD360_20_30kA_sensors_median_grid
    Total_30kA_plus_sensors_median[:, :] = GLD360_30kA_plus_sensors_median_grid

    '''
    Add metadata for all variables
    '''
    lat.unit = 'degrees north'
    lat.long_name = 'Latitude of lower-left grid cell corner'
    lon.unit = 'degrees east'
    lon.long_name = 'Longitude of lower-left grid cell corner'

    Total_0_10kA_sensors_median.long_name = 'Median number of sensor reports of total strokes with absolute peak current between 0–10 kA'
    Total_0_10kA_sensors_median.coordinates = 'lat lon'
    Total_10_20kA_sensors_median.long_name = 'Median number of sensor reports of total strokes with absolute peak current between 10–20 kA'
    Total_10_20kA_sensors_median.coordinates = 'lat lon'
    Total_20_30kA_sensors_median.long_name = 'Median number of sensor reports of total strokes with absolute peak current between 20–30 kA'
    Total_20_30kA_sensors_median.coordinates = 'lat lon'
    Total_30kA_plus_sensors_median.long_name = 'Median number of sensor reports of total strokes with absolute peak current above 30 kA'
    Total_30kA_plus_sensors_median.coordinates = 'lat lon'

    '''
    Close file
    '''
    SR_file.close()


def main():
    month_range = (pd.date_range(start=str(start_month), end=str(end_month), freq='MS')).strftime('%Y_%m').tolist()
    for month in month_range:
        GLD360_vector_to_monthly_grid(month)


if __name__ == '__main__':
    main()