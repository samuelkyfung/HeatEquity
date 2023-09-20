#-------------------------------------------------------
# Fig2.py
#
# Kwun Yip Fung
# 10 Mar 2023
# 
# 1. Create diurnal cycle plot for urban areaveraged UTCI 
#    and its components
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
wrf_T2          = np.zeros((nlegend, ncases, ntime_0912, len(x_join)))
wrf_RH2         = np.zeros((nlegend, ncases, ntime_0912, len(x_join)))
wrf_MRT         = np.zeros((6, nlegend, ncases, ntime_0912, len(x_join)))
wrf_MRT1        = np.zeros((nlegend, ncases, ntime_0912, len(x_join)))
wrf_MRT2        = np.zeros((nlegend, ncases, ntime_0912, len(x_join)))
wrf_MRT3        = np.zeros((nlegend, ncases, ntime_0912, len(x_join)))
wrf_MRT4        = np.zeros((nlegend, ncases, ntime_0912, len(x_join)))
wrf_MRT5        = np.zeros((nlegend, ncases, ntime_0912, len(x_join)))
wrf_MRT6        = np.zeros((nlegend, ncases, ntime_0912, len(x_join)))
wrf_10wspd      = np.zeros((nlegend, ncases, ntime_0912, len(x_join)))
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
        wrf_T2    [i,icase,:,:] = getvar(Dataset(wrffile), 'T2'       , timeidx = ALL_TIMES, meta=False)[:,x_join, y_join]
        wrf_RH2   [i,icase,:,:] = getvar(Dataset(wrffile), 'rh2'      , timeidx = ALL_TIMES, meta=False)[:,x_join, y_join]
        wrf_MRT1  [i,icase,:,:] = getvar(Dataset(wrffile), 'TMR_11'   , timeidx = ALL_TIMES, meta=False)[:,x_join, y_join]
        wrf_MRT2  [i,icase,:,:] = getvar(Dataset(wrffile), 'TMR_12'   , timeidx = ALL_TIMES, meta=False)[:,x_join, y_join]
        wrf_MRT3  [i,icase,:,:] = getvar(Dataset(wrffile), 'TMR_13'   , timeidx = ALL_TIMES, meta=False)[:,x_join, y_join]
        wrf_MRT4  [i,icase,:,:] = getvar(Dataset(wrffile), 'TMR_21'   , timeidx = ALL_TIMES, meta=False)[:,x_join, y_join]
        wrf_MRT5  [i,icase,:,:] = getvar(Dataset(wrffile), 'TMR_22'   , timeidx = ALL_TIMES, meta=False)[:,x_join, y_join]
        wrf_MRT6  [i,icase,:,:] = getvar(Dataset(wrffile), 'TMR_23'   , timeidx = ALL_TIMES, meta=False)[:,x_join, y_join]
        wrf_10wspd[i,icase,:,:] = getvar(Dataset(wrffile), 'metwspd10', timeidx = ALL_TIMES, meta=False)[:,x_join, y_join]
        wrf_utci10[i,icase,:,:] = getvar(Dataset(wrffile), 'COMF_10'  , timeidx = ALL_TIMES, meta=False)[:,x_join, y_join]
        wrf_utci50[i,icase,:,:] = getvar(Dataset(wrffile), 'COMF_50'  , timeidx = ALL_TIMES, meta=False)[:,x_join, y_join]
        wrf_utci90[i,icase,:,:] = getvar(Dataset(wrffile), 'COMF_90'  , timeidx = ALL_TIMES, meta=False)[:,x_join, y_join]

print("--- %s seconds ---" % (time.time() - start_time))

wrf_utci10[np.where(wrf_utci10 <=0)] = np.nan
wrf_utci50[np.where(wrf_utci50 <=0)] = np.nan
wrf_utci90[np.where(wrf_utci90 <=0)] = np.nan
wrf_MRT1  [np.where(wrf_MRT1   <=0)] = np.nan
wrf_MRT2  [np.where(wrf_MRT2   <=0)] = np.nan
wrf_MRT3  [np.where(wrf_MRT3   <=0)] = np.nan
wrf_MRT4  [np.where(wrf_MRT4   <=0)] = np.nan
wrf_MRT5  [np.where(wrf_MRT5   <=0)] = np.nan
wrf_MRT6  [np.where(wrf_MRT6   <=0)] = np.nan
wrf_MRT   [0,:,:,:,:]      = wrf_MRT1
wrf_MRT   [1,:,:,:,:]      = wrf_MRT2
wrf_MRT   [2,:,:,:,:]      = wrf_MRT3
wrf_MRT   [3,:,:,:,:]      = wrf_MRT4
wrf_MRT   [4,:,:,:,:]      = wrf_MRT5
wrf_MRT   [5,:,:,:,:]      = wrf_MRT6
wrf_MRT = np.mean(wrf_MRT, 0)



#------ Plot Diurnal time series of Temperature metris -------# (Fig. 2)
empty = wrf_utci50.copy()
empty[:] = np.nan
pltvar_list = [wrf_utci50      , wrf_T2           , wrf_RH2                 , wrf_10wspd       , wrf_MRT                   , empty           ]
pltvar_name = ['UTCI'          , '2-m Temperature', '2-m Relative Humidity' , '10-m Wind Speed', 'Mean Radiant Temperature', 'UTCI'          ]
pltvar_unit = ['K'             , 'K'              , '%'                     , 'm/s'            , 'K'                       , 'K'             ]

legend_color= ['blue', 'blue', 'red', 'red', 'green', 'green']
legend_width= [3, 1, 3, 1, 3, 1]
legend_types= ['-', '--', '-', '--', '-', '--']


ndays = 4
plt.figure(figsize=(12, 8))
for iplt in range(len(pltvar_list)):
    ax = plt.subplot(2,3,iplt+1)
    
    if (iplt == 5):
        pltvar = pltvar_list[iplt][:,:,0:-1,:].reshape(nlegend, ncases, int(ntime_0912/24), 24, len(x_join))
        pltvar = np.nanmean(np.nanmean(np.nanmean(pltvar[:,:,0:ndays,:,:], 1),1),2) #0-2 days
        pltvar = np.roll(pltvar, UTC_zone, axis = 1)
        pltvar = np.hstack((pltvar, pltvar[:, 0:1])) # Repeat the 0Z as 24Z
        for (ilegend, legend) in enumerate(legend_name[1::]):
            ax.plot([],[], 
                           label = legend + ' - Ctl', 
                           color = legend_color[ilegend],
                           linewidth = legend_width[ilegend], 
                           linestyle = legend_types[ilegend])
        ax.legend(fontsize = 12)
    
    else:
        pltvar = pltvar_list[iplt][:,:,0:-1,:].reshape(nlegend, ncases, int(ntime_0912/24), 24, len(x_join))
        pltvar = np.nanmean(np.nanmean(np.nanmean(pltvar[:,:,0:ndays,:,:], 1),1),2) #0-2 days
        pltvar = np.roll(pltvar, UTC_zone, axis = 1)
        pltvar = np.hstack((pltvar, pltvar[:, 0:1])) # Repeat the 0Z as 24Z
        ax.axhline(y = 0, color = 'grey', linestyle = '--')
        
        for (ilegend, legend) in enumerate(legend_name[1::]):
            ax.plot(pltvar[ilegend+1, :]-pltvar[0, :], 
                           label = legend + ' - Ctl', 
                           color = legend_color[ilegend],
                           linewidth = legend_width[ilegend], 
                           linestyle = legend_types[ilegend])
        
        ax.axvspan(0 , 6, alpha=0.3, color='grey')
        ax.axvspan(18, 24, alpha=0.3, color='grey')
        ax.set_title(string.ascii_lowercase[iplt]+') '+pltvar_name[iplt], 
                     fontsize=12, pad=10, loc = 'left')
        ax.set_xticks(np.arange(0, 25, 6))
        ax.set_xlim([0, 24])
        ax.set_xlabel('Local Time')
        ax.set_ylabel(pltvar_unit[iplt])
        ax.tick_params(axis='x', rotation=0)

plt.suptitle('Diurnal cycle of Temperature metrics in {:01d} days'.format(ndays))
plt.tight_layout()
plt.savefig(opath+'Nature_LCZ_GLO.mitigation.Tmetrics_caseavg_T_HI_UTCI_RH_WSPD_MRT{:01d}days_DiuTS2'.format(ndays)+'.png',dpi=dpi, bbox_inches='tight')
#plt.close()
plt.show()

