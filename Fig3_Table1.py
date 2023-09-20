#-------------------------------------------------------
# Fig3_Table1.py
#
# Kwun Yip Fung
# 10 Mar 2023
# 
# 1. Create barplot of ADCH changes for each strategy
#--------------------------------------------------------
from netCDF4 import Dataset
from wrf import getvar, ALL_TIMES, extract_times
from glob import glob
import pandas as pd
import numpy as np
import os
from wrf import ll_to_xy
import matplotlib.pyplot as plt
import time
import string
import geopandas
import datetime as dt

ipath           = '/scratch/05898/kf22523/data/WRF_out/Houston/Manuscript/'
opath           = '/home1/05898/kf22523/graphs/graphs/Houston/Heatwaves/'

UTC_zone            = -6
urb_region_lat      = [29.3, 30.5]
urb_region_lon      = [-96, -94.8]
#
cases           = ['20170728-20170801_HW1',\
                   '20180722-20180726_HW2',\
                   '20180820-20180824_HW3',\
                   '20190813-20190817_HW4',\
                   '20190904-20190908_HW5']
                   
ncases          = len(cases)
dpi             = 800


legend_0912 = ['LCZ_GLO/E5_AHgr0tc0ar0ag0_d03',\
               'LCZ_GLO/E5_AHgr0tc0arHag0_d03',\
               'LCZ_GLO/E5_AHgr0tc0arLag0_d03',\
               'LCZ_GLO/E5_AHgr0tcHar0ag0_d03',\
               'LCZ_GLO/E5_AHgr0tcLar0ag0_d03',\
               'LCZ_GLO/E5_AHgrHtc0ar0ag0_d03',\
               'LCZ_GLO/E5_AHgrLtc0ar0ag0_d03']
legend_name = ['Ctl',\
               'Cool Roofs (H)',\
               'Cool Roofs (L)',\
               'Urban Trees (H)',\
               'Urban Trees (L)',\
               'Green Roofs (H)',\
               'Green Roofs (L)']
nlegend = len(legend_0912)


ipath_nc_0912   = ipath+cases[0]+'/'
geo_em_nc       = ipath+'/geo_em/LCZ_GLO/geo_em.d03.nc'
wrffile_0912    = [glob(ipath_nc_0912 + ipre + '*.nc')[0] for ipre in legend_0912]
wrfin_0912      = Dataset(wrffile_0912[0])
geo_em          = Dataset(geo_em_nc)
nexpts          = len(legend_0912)
ntime_0912      = wrfin_0912.dimensions['Time'].size
nx              = wrfin_0912.dimensions['south_north'].size
ny              = wrfin_0912.dimensions['west_east'].size
nlev            = wrfin_0912.dimensions['bottom_top'].size
lat             = getvar(wrfin_0912, "lat"  , timeidx = 0, meta = False)
lon             = getvar(wrfin_0912, "lon"  , timeidx = 0, meta = False)
t2_0            = getvar(wrfin_0912, "T2"   , timeidx = 0, meta = True)#, method="join")
time_0912       = pd.to_datetime(extract_times(wrfin_0912, timeidx = ALL_TIMES))
LU              = getvar(geo_em    , "LU_INDEX", timeidx = 0        , meta = False)
#LU[LU>30]  = 30
LU[LU==13] = 30

# Find the i,j index for the intersted region
ll = ll_to_xy(Dataset(wrffile_0912[0]), urb_region_lat[0], urb_region_lon[0])
ul = ll_to_xy(Dataset(wrffile_0912[0]), urb_region_lat[1], urb_region_lon[0])
lr = ll_to_xy(Dataset(wrffile_0912[0]), urb_region_lat[0], urb_region_lon[1])
ur = ll_to_xy(Dataset(wrffile_0912[0]), urb_region_lat[1], urb_region_lon[1])
i_s = np.min([ll[1], ul[1], lr[1], ur[1]])
i_e = np.max([ll[1], ul[1], lr[1], ur[1]])
j_s = np.min([ll[0], ul[0], lr[0], ur[0]])
j_e = np.max([ll[0], ul[0], lr[0], ur[0]])

# Polygon centroids
shapefile   = geopandas.read_file(ipath + "/SVI_Houston_WRFd03/SVI_Houston_WRFd03.shp")
lats        = shapefile.geometry.centroid.y.values  # Get the centroid lat of the polygons
lons        = shapefile.geometry.centroid.x.values  # Get the centroid lon of the polygons
y, x        = ll_to_xy(wrfin_0912, lats, lons)      # Get the resepective x, y WRF coordinate 
npolygons   = len(shapefile.geometry)


# Only extract indices in LCZ
lat_trim = lat[i_s:i_e, j_s:j_e][np.where(LU[i_s:i_e, j_s:j_e]>=31)]
lon_trim = lon[i_s:i_e, j_s:j_e][np.where(LU[i_s:i_e, j_s:j_e]>=31)]

points = geopandas.GeoDataFrame({
    "id": range(len(lat_trim.ravel())),
    "geometry": geopandas.points_from_xy(lon_trim.ravel(), lat_trim.ravel())
})
#joined          = geopandas.sjoin(points, shape_pd_new, op="within")
joined          = geopandas.sjoin(points, shapefile, op="within")
lats_join       = joined.geometry.centroid.y.values  # Get the centroid lat of the polygons
lons_join       = joined.geometry.centroid.x.values  # Get the centroid lon of the polygons
y_join, x_join  = ll_to_xy(wrfin_0912, lats_join, lons_join)   # Get the resepective x, y WRF coordinate 

LU[x_join, y_join]



# SVI components
df                = pd.DataFrame()
df['OBJECTID'   ] = joined.OBJECTID.values
df['I_SVI'      ] = np.array([shapefile[shapefile.OBJECTID == i].RPL_THEMES.values[0] for i in joined.OBJECTID])
df_census_tract = df.groupby('OBJECTID').mean().reset_index(drop = False)

# Read WRF values
wrf_utci10      = np.zeros((nlegend, ncases, ntime_0912, len(x_join)))
wrf_utci50      = np.zeros((nlegend, ncases, ntime_0912, len(x_join)))
wrf_utci90      = np.zeros((nlegend, ncases, ntime_0912, len(x_join)))


start_time = time.time()
for icase in range(len(cases)):
    print("case: " + cases[icase])
    ipath_nc_0912   = ipath+cases[icase]+'/'
    wrffile_0912    = [glob(ipath_nc_0912 + ipre + '*.nc')[0] for ipre in legend_0912]
    
    for i, legend in enumerate(legend_0912):
        print("read: " + legend_name[i])
        wrffile = wrffile_0912[i]
        wrf_utci10[i,icase,:,:] = getvar(Dataset(wrffile), 'COMF_10'  , timeidx = ALL_TIMES, meta=False)[:,x_join, y_join]
        wrf_utci50[i,icase,:,:] = getvar(Dataset(wrffile), 'COMF_50'  , timeidx = ALL_TIMES, meta=False)[:,x_join, y_join]
        wrf_utci90[i,icase,:,:] = getvar(Dataset(wrffile), 'COMF_90'  , timeidx = ALL_TIMES, meta=False)[:,x_join, y_join]

print("--- %s seconds ---" % (time.time() - start_time))

wrf_utci10[np.where(wrf_utci10 <=0)] = np.nan
wrf_utci50[np.where(wrf_utci50 <=0)] = np.nan
wrf_utci90[np.where(wrf_utci90 <=0)] = np.nan


# Calculate the amount of UTCI >= 26
# (UTCI - 26) * amount of hours
threshold = 26 # 26 # 32 # 38 # 46
pltvar                        = wrf_utci50[:,:,1::,:].reshape(nlegend, ncases, int(ntime_0912/24), 24, len(x_join))-threshold
pltvar[np.where(pltvar <= 0)] = np.nan
pltvar_dur                    = pltvar.copy()
pltvar_dur[np.where(pltvar_dur >= 0)] = 1
pltvar                        = np.nanmean(np.nanmean(np.nanmean(pltvar    ,2),1), 2)
pltvar_dur                    = np.nanmean(np.nanmean(np.nanmean(pltvar_dur,2),1), 2)
day                           = list(np.arange(6,19))
night                         = list(np.arange(0,6)) + list(np.arange(19,24))
#wrf_dutci50_exposure_all      = np.nanmean(np.nanmean(np.nanmean((wrf_utci50[:,:,1::,:] - 26), 1), 1), 1)
wrf_dutci50_exposure_all      = np.nansum(np.roll(pltvar    , UTC_zone, axis = 1)[:,:    ], 1)
wrf_dutci50_exposure_day      = np.nansum(np.roll(pltvar    , UTC_zone, axis = 1)[:,day  ], 1)
wrf_dutci50_exposure_night    = np.nansum(np.roll(pltvar    , UTC_zone, axis = 1)[:,night], 1)
wrf_dutci50_duration_all      = np.nansum(np.roll(pltvar_dur, UTC_zone, axis = 1)[:,:    ], 1)
wrf_dutci50_duration_day      = np.nansum(np.roll(pltvar_dur, UTC_zone, axis = 1)[:,day  ], 1)
wrf_dutci50_duration_night    = np.nansum(np.roll(pltvar_dur, UTC_zone, axis = 1)[:,night], 1)

wrf_dutci50_exp_percent_all   = (wrf_dutci50_exposure_all  [1::] - wrf_dutci50_exposure_all  [0])*100/wrf_dutci50_exposure_all  [0]
wrf_dutci50_exp_percent_day   = (wrf_dutci50_exposure_day  [1::] - wrf_dutci50_exposure_day  [0])*100/wrf_dutci50_exposure_day  [0]
wrf_dutci50_exp_percent_night = (wrf_dutci50_exposure_night[1::] - wrf_dutci50_exposure_night[0])*100/wrf_dutci50_exposure_night[0]
wrf_dutci50_dur_percent_all   = (wrf_dutci50_duration_all  [1::] - wrf_dutci50_duration_all  [0])*100/wrf_dutci50_duration_all  [0]
wrf_dutci50_dur_percent_day   = (wrf_dutci50_duration_day  [1::] - wrf_dutci50_duration_day  [0])*100/wrf_dutci50_duration_day  [0]
wrf_dutci50_dur_percent_night = (wrf_dutci50_duration_night[1::] - wrf_dutci50_duration_night[0])*100/wrf_dutci50_duration_night[0]

print('All day')
for i,ilegend in enumerate(legend_name[1::]):
    print('Intensity*Duration UTCI50 changes ' + ilegend + ': {:.2f}%'.format(wrf_dutci50_exp_percent_all[i]))

print('Day')
for i,ilegend in enumerate(legend_name[1::]):
    print('Intensity*Duration UTCI50 changes ' + ilegend + ': {:.2f}%'.format(wrf_dutci50_exp_percent_day[i]))

print('Night')
for i,ilegend in enumerate(legend_name[1::]):
    print('Intensity*Duration UTCI50 changes ' + ilegend + ': {:.2f}%'.format(wrf_dutci50_exp_percent_night[i]))



# Get day and nighttime index
ndays       = 4
daytime     = [np.arange((iday)*24+12, (iday)*24+24) for iday in range(ndays)]
nighttime   = [np.arange((iday)*24+0 , (iday)*24+12) for iday in range(ndays)]
whole       = list(np.arange(0, ndays*24))
daytime     = list(itertools.chain.from_iterable(daytime))
nighttime   = list(itertools.chain.from_iterable(nighttime))
daytime     = list(np.array(daytime  )[np.where(np.array(daytime  ) < ntime_0912)]) # confirm index is within ntime
nighttime   = list(np.array(nighttime)[np.where(np.array(nighttime) < ntime_0912)]) # confirm index is within ntime
day         = list(np.arange(6,19))
night       = list(np.arange(0,6)) + list(np.arange(19,24))


dtdf_caseavg_Whole = pd.DataFrame()
dtdf_caseavg_Day   = pd.DataFrame()
dtdf_caseavg_Night = pd.DataFrame()

dtdf_caseavg_Whole['OBJECTID'] = joined.OBJECTID.values
dtdf_caseavg_Day  ['OBJECTID'] = joined.OBJECTID.values
dtdf_caseavg_Night['OBJECTID'] = joined.OBJECTID.values
# Time and case averaged
for (i,legend) in enumerate(legend_name[1::]):
    utci50m26 = wrf_utci50-threshold
    utci50m26[np.where(utci50m26 <=0)] = np.nan
    utci50m26    = utci50m26[:,:,1::,:].reshape(nlegend, ncases, int(ntime_0912/24), 24, len(x_join))
    utci50m26exp = utci50m26[1::] - utci50m26[0]
    
    dtdf_caseavg_Whole[legend+'.'+'dUTCI50_e%'] = np.nanmean(np.nanmean(np.nanmean(utci50m26exp[:,:,:,:,:], 3), 2),1)[i,:]*100 / np.nanmean(np.nanmean(np.nanmean(utci50m26[0,:,:,:,:], 2),1),0)
    dtdf_caseavg_Day  [legend+'.'+'dUTCI50_e%'] = np.nanmean(np.nanmean(np.nanmean(utci50m26exp[:,:,:,day,:], 3), 2),1)[i,:]*100/np.nanmean(np.nanmean(np.nanmean(utci50m26[0,:,:,day,:], 2),1),0)
    dtdf_caseavg_Night[legend+'.'+'dUTCI50_e%'] = np.nanmean(np.nanmean(np.nanmean(utci50m26exp[:,:,:,night,:], 3), 2),1)[i,:]*100/np.nanmean(np.nanmean(np.nanmean(utci50m26[0,:,:,night,:], 2),1),0)

# Group into census tract
dtdf_caseavg_Whole = dtdf_caseavg_Whole.groupby('OBJECTID').mean().reset_index(drop = False)
dtdf_caseavg_Day   = dtdf_caseavg_Day.groupby('OBJECTID').mean().reset_index(drop = False)
dtdf_caseavg_Night = dtdf_caseavg_Night.groupby('OBJECTID').mean().reset_index(drop = False)


#---------- Plot Bar Chart -----------# (Fig. 3) 
# Plot bar chart of SVI with 0.1 as bin and dT as y-axis
Y_metrics = ['dUTCI50_e%']
Y_label   = ['%']
SVIvar    = 'I_SVI'
X_label   = 'SVI'

for imetric,metric in enumerate(Y_metrics):
    
    plt_name = 'Nature_Bar.regression.LCZ_GLO.'+metric+'.vs.'+SVIvar+'threshold32.png'
    print('Drawing bar plot: ' + plt_name)
    #plt.figure(figsize=(15, 9))
    #for iplt,legend in enumerate(legend_name[1::]):
    #plt.figure(figsize=(15, 9))
    #for iplt,legend in enumerate(['Cool Roofs (H)', 'Green Roofs (H)', 'Urban Trees (H)', 
    #                              'Cool Roofs (L)', 'Green Roofs (L)', 'Urban Trees (L)']):
    #   ax = plt.subplot(2,3,iplt+1)
    plt.figure(figsize=(15, 5.5))
    for iplt,legend in enumerate(['Cool Roofs (H)', 'Green Roofs (H)', 'Urban Trees (H)']):
        ax = plt.subplot(1,3,iplt+1)
        colname_i = legend+'.'+metric
        Bar_Whole_df = pd.DataFrame()
        Bar_Whole_df = dtdf_caseavg_Whole[['OBJECTID', colname_i]]
        Bar_Whole_df[SVIvar] = df_census_tract[SVIvar]#
        Bar_Whole_df = Bar_Whole_df.loc[~(df == -999).any(axis=1)].reset_index(drop = True)
        # Bin the data into intervals of 0.1 using pd.cut()
        Bar_Whole_df['Interval'] = pd.cut(Bar_Whole_df[SVIvar], bins=[i/10 for i in range(0, 11)])
        # Group the data by the intervals and Mean of the counts
        grouped = Bar_Whole_df.groupby('Interval')[colname_i].mean()
        
        q3      = Bar_Whole_df.groupby('Interval')[colname_i].quantile(0.75)
        q1      = Bar_Whole_df.groupby('Interval')[colname_i].quantile(0.25)
        iqr     = (q3-q1)/2
        
        ax = grouped.plot(kind='bar', yerr = iqr, width=0.8, capsize=3, error_kw=dict(lw=0.8, capthick=0.8))
        ax.set_xlabel(X_label)
        ax.set_ylabel(Y_label[imetric])
        ax.set_title(string.ascii_lowercase[iplt] +') '+ legend, fontsize=14, pad=10, loc = 'left')
    
    plt.suptitle(metric+' vs '+SVIvar)
    SMALL_SIZE = 14
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 14
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.tight_layout()#pad=0.4, w_pad=0.5, h_pad=1.0)
    #plt.savefig(opath + plt_name, dpi=dpi, transparent=False, bbox_inches='tight')
    #plt.close()
    plt.show()


# For Table 1 values
import warnings
warnings.filterwarnings("ignore")

metric = Y_metrics[0]
SVIvar = 'I_SVI'
print('All day')
for i,legend in enumerate(legend_name[1::]):
    colname_i = legend+'.'+metric
    Bar_Whole_df = pd.DataFrame()
    Bar_Whole_df = dtdf_caseavg_Whole[['OBJECTID', colname_i]]
    Bar_Whole_df[SVIvar] = df_census_tract[SVIvar]#
    Bar_Whole_df['SVIxHS'] = Bar_Whole_df[colname_i] * Bar_Whole_df[SVIvar]
    Bar_Whole_df = Bar_Whole_df.loc[~(Bar_Whole_df < -998).any(axis=1)].reset_index(drop = True)
    print('SVI*Intensity*Duration UTCI50 changes ' + legend + ': {:.2f}%'.format(Bar_Whole_df['SVIxHS'].mean()))

print('Day')
for i,legend in enumerate(legend_name[1::]):
    colname_i = legend+'.'+metric
    Bar_Whole_df = pd.DataFrame()
    Bar_Whole_df = dtdf_caseavg_Day[['OBJECTID', colname_i]]
    Bar_Whole_df[SVIvar] = df_census_tract[SVIvar]#
    Bar_Whole_df['SVIxHS'] = Bar_Whole_df[colname_i] * Bar_Whole_df[SVIvar]
    Bar_Whole_df = Bar_Whole_df.loc[~(Bar_Whole_df < -998).any(axis=1)].reset_index(drop = True)
    print('SVI*Intensity*Duration UTCI50 changes ' + legend + ': {:.2f}%'.format(Bar_Whole_df['SVIxHS'].mean()))

print('Night')
for i,legend in enumerate(legend_name[1::]):
    colname_i = legend+'.'+metric
    Bar_Whole_df = pd.DataFrame()
    Bar_Whole_df = dtdf_caseavg_Night[['OBJECTID', colname_i]]
    Bar_Whole_df[SVIvar] = df_census_tract[SVIvar]#
    Bar_Whole_df['SVIxHS'] = Bar_Whole_df[colname_i] * Bar_Whole_df[SVIvar]
    Bar_Whole_df = Bar_Whole_df.loc[~(Bar_Whole_df < -998).any(axis=1)].reset_index(drop = True)
    print('SVI*Intensity*Duration UTCI50 changes ' + legend + ': {:.2f}%'.format(Bar_Whole_df['SVIxHS'].mean()))

