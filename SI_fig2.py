#-------------------------------------------------------
# SI_fig2.py
#
# Kwun Yip Fung
# 10 Mar 2023
# 
# 1. Spatial plot of MODIS and experiment LST
# 2. Calculate the RMSE, MAE, MB again MODIS
#--------------------------------------------------------
from netCDF4 import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import NaturalEarthFeature
from wrf import getvar, to_np, ALL_TIMES, extract_times, ll_to_xy
from dateutil.parser import parse
from glob import glob
import time
import pandas as pd
from geocat.viz import util as gvutil
import string
from scipy.stats import pearsonr
import geopandas
import datetime as dt

opath               = '/home1/05898/kf22523/graphs/graphs/Houston/Heatwaves/'
ipath               = '/scratch/05898/kf22523/data/WRF_out/Houston/Manuscript/'
MOD_EXPT            = ["MOD21A1D", "MOD21A1N",  "MYD21A1D", "MYD21A1N"]
urb_region_lat      = [29.3, 30.5]
urb_region_lon      = [-96, -94.8]
consider_threshold  = 0.8
Time_zone           = -5
#
cases           = ['20170728-20170801_HW1',\
                   '20180722-20180726_HW2',\
                   '20180820-20180824_HW3',\
                   '20190813-20190817_HW4',\
                   '20190904-20190908_HW5']
                   
ncases          = len(cases)
dpi             = 800

countyfile      = "/work/05898/kf22523/stampede2/dataset/Houston_shapefile/cb_2018_us_county_5m.shp"
#houstonfile     = '/work/05898/kf22523/stampede2/dataset/Houston_shapefile/dissolve_houston.shp'


legend_0912     = ['LCZ_D/E5_AHgr0tc0ar0ag0_d03',\
                   'LCZ_GLO/E5_AHgr0tc0ar0ag0_d03']
legend          = ['LCZ_D'  ,\
                   'LCZ_GLO']
nlegend         = len(legend_0912)
ipath_nc_0912   = ipath+cases[0]+'/'

wrffile_0912    = [glob(ipath_nc_0912 + ipre + '*.nc')[0] for ipre in legend_0912]
geo_em_nc       = ipath+'/geo_em/LCZ_GLO/geo_em.d03.nc'
wrfin_0912      = Dataset(wrffile_0912[0])
geo_em          = Dataset(geo_em_nc)
nexpts          = len(legend_0912)
ntime_0912      = wrfin_0912.dimensions['Time'].size
nx              = wrfin_0912.dimensions['south_north'].size
ny              = wrfin_0912.dimensions['west_east'].size
nlev            = wrfin_0912.dimensions['bottom_top'].size
lat             = getvar(wrfin_0912, "lat"  , timeidx = 0        , meta = False)
lon             = getvar(wrfin_0912, "lon"  , timeidx = 0        , meta = False)
#z               = getvar(wrfin_0912, "z"    , timeidx = 0        , meta = True)
t2_0            = getvar(wrfin_0912, "T2", timeidx = 0, meta = True)#, method="join")
time_0912       = pd.to_datetime(extract_times(wrfin_0912, timeidx = ALL_TIMES))
LU              = getvar(geo_em    , "LU_INDEX", timeidx = 0     , meta = False)

# Find the i,j index for the intersted region
ll = ll_to_xy(Dataset(wrffile_0912[0]), urb_region_lat[0], urb_region_lon[0])
ul = ll_to_xy(Dataset(wrffile_0912[0]), urb_region_lat[1], urb_region_lon[0])
lr = ll_to_xy(Dataset(wrffile_0912[0]), urb_region_lat[0], urb_region_lon[1])
ur = ll_to_xy(Dataset(wrffile_0912[0]), urb_region_lat[1], urb_region_lon[1])
i_s = np.min([ll[1], ul[1], lr[1], ur[1]])
i_e = np.max([ll[1], ul[1], lr[1], ur[1]])
j_s = np.min([ll[0], ul[0], lr[0], ur[0]])
j_e = np.max([ll[0], ul[0], lr[0], ur[0]])



import geopandas
from wrf import ll_to_xy
import datetime as dt

# Polygon centroids
shapefile   = geopandas.read_file(ipath + "/SVI_Houston_WRFd03/SVI_Houston_WRFd03.shp")
lats        = shapefile.geometry.centroid.y.values  # Get the centroid lat of the polygons
lons        = shapefile.geometry.centroid.x.values  # Get the centroid lon of the polygons
y, x        = ll_to_xy(wrfin_0912, lats, lons)      # Get the resepective x, y WRF coordinate 
npolygons   = len(shapefile.geometry)

# Only extract indices in LCZ
lat_trim = lat[i_s:i_e, j_s:j_e]
lon_trim = lon[i_s:i_e, j_s:j_e]
points = geopandas.GeoDataFrame({
    "id": range(len(lat_trim.ravel())),
    "geometry": geopandas.points_from_xy(lon_trim.ravel(), lat_trim.ravel())
})
#joined          = geopandas.sjoin(points, shape_pd_new, op="within")
joined          = geopandas.sjoin(points, shapefile, op="within")
lats_join       = joined.geometry.centroid.y.values  # Get the centroid lat of the polygons
lons_join       = joined.geometry.centroid.x.values  # Get the centroid lon of the polygons
y_join, x_join  = ll_to_xy(wrfin_0912, lats_join, lons_join)   # Get the resepective x, y WRF coordinate 
SVI_join        = np.array([shapefile[shapefile.OBJECTID == i].RPL_THEMES.values[0] for i in joined.OBJECTID])

# Remove index where SVI < 0
SVI             = SVI_join[np.where(SVI_join >= 0)]
y_join_f        = y_join  [np.where(SVI_join >= 0)]
x_join_f        = x_join  [np.where(SVI_join >= 0)]


# Calculate the RMSE of LST
wrf_LST         = np.zeros((nlegend, len(cases), len(MOD_EXPT), int(ntime_0912/24), nx, ny))
mod_LST         = np.zeros((1      , len(cases), len(MOD_EXPT), int(ntime_0912/24), nx, ny))

nonWeighted_RMSE= np.zeros((nlegend, len(cases), len(MOD_EXPT), int(ntime_0912/24)))
nonWeighted_MAE = np.zeros((nlegend, len(cases), len(MOD_EXPT), int(ntime_0912/24)))
nonWeighted_RMSE= np.zeros((nlegend, len(cases), len(MOD_EXPT), int(ntime_0912/24)))
nonWeighted_MB  = np.zeros((nlegend, len(cases), len(MOD_EXPT), int(ntime_0912/24)))
Correlation     = np.zeros((nlegend, len(cases), len(MOD_EXPT), int(ntime_0912/24)))

nonWeighted_MAE [:] = np.nan
nonWeighted_RMSE[:] = np.nan
nonWeighted_MB  [:] = np.nan
Correlation     [:] = np.nan

for icase in range(len(cases)):
    print("case: " + cases[icase])
    ipath_nc_0912   = ipath+cases[icase]+'/'
    wrffile_0912    = [glob(ipath_nc_0912 + ipre + '*.nc')[0] for ipre in legend_0912]
    
    for imodexpt in range(len(MOD_EXPT)):
        MOD_DATA         = MOD_EXPT[imodexpt]
        aqua_regird_path = ipath+"/MODIS_LST_regrided/"+MOD_DATA+"/"+cases[icase]+"/"
        aqua_file        = sorted(glob(aqua_regird_path   + "*" + ".WRFd04grid.nc"))
        
        # MODIS file (each day) loop
        for iobs, obsfile in enumerate(aqua_file):
            mod_lst     = Dataset(obsfile).variables['T2_regird_WRF'][i_s:i_e, j_s:j_e]
            urbind_trim = np.where(LU[i_s:i_e, j_s:j_e] >= 30)
            mod_lst_urb = mod_lst[urbind_trim]
            mod_time    = Dataset(obsfile).variables['time_regrid_WRF']
            
            # MODIS use local time --> convert to UTC as WRF
            mod_time_UTC = pd.to_datetime(mod_time.units, format='hours since %Y-%m-%d %H:%M:%S') + \
            - dt.timedelta(hours = int(Time_zone)) \
            + dt.timedelta(hours = int(np.round(np.nanmean(mod_time[:]))))
            #- dt.timedelta(hours = 24)
            
            # Calculate how many effective grids 
            considered = False
            #if (len(list(np.where(mod_lst.mask == False)[0])) / (mod_lst.shape[0]*mod_lst.shape[1])  >= consider_threshold):
            if (len(list(np.where(mod_lst_urb.mask == False)[0])) / (mod_lst_urb.shape[0])  >= consider_threshold):
                considered = True
                print('Time: '+ str(mod_time_UTC) + ', MOD_EXPT: ' + MOD_DATA+ ', imodexpt: ' + str(imodexpt)+ ', iobs    : ' + str(iobs))
            
            # Ensemble loop
            if considered:
                for i in range (nlegend):
                    wrffile = wrffile_0912[i]
                    
                    # Find the corresponding time frame in wrf
                    time_0912       = pd.to_datetime(extract_times(Dataset(wrffile), timeidx = ALL_TIMES))
                    time_0912       = time_0912.round(freq = 'H')
                    try:
                        wrf_aqua_frame = np.where(time_0912 == mod_time_UTC)[0][0]
                    except IndexError:
                        print('Days not in simulation')
                        considered = False
                        break
                    
                    if considered:
                        wrf_LST [i, icase, imodexpt, iobs] = Dataset(wrffile).variables['TSK_SAT'][wrf_aqua_frame, :, :]
                        mod_LST [0, icase, imodexpt, iobs] = Dataset(obsfile).variables['T2_regird_WRF'][:,:]
                        validind = np.where(mod_LST [0, icase, imodexpt, iobs] > 0) # Only consider non-missing index for correaltion 
                        
                        tsk_abserror = np.abs( Dataset(wrffile).variables['TSK_SAT'][wrf_aqua_frame,:,:]  - Dataset(obsfile).variables['T2_regird_WRF'][:,:])
                        tsk_errorsq  =       ( Dataset(wrffile).variables['TSK_SAT'][wrf_aqua_frame,:,:]  - Dataset(obsfile).variables['T2_regird_WRF'][:,:])**2
                        tsk_error    =         Dataset(wrffile).variables['TSK_SAT'][wrf_aqua_frame,:,:]  - Dataset(obsfile).variables['T2_regird_WRF'][:,:]
                        corr         =pearsonr(wrf_LST [i, icase, imodexpt, iobs][validind].ravel(),  mod_LST [0, icase, imodexpt, iobs][validind].ravel())[0]
                        Total_grids = 0
                        Sum_of_nonweighted_error_MAE = 0
                        Sum_of_nonweighted_error_RMSE= 0
                        Sum_of_nonweighted_error_MB  = 0                            
                        if ((not np.all(tsk_abserror[x_join_f, y_join_f].mask)) ):
                            Total_grids = Total_grids + len(x_join_f)
                            Sum_of_nonweighted_error_MAE = Sum_of_nonweighted_error_MAE + \
                                                tsk_abserror[x_join_f, y_join_f].sum()
                            Sum_of_nonweighted_error_RMSE= Sum_of_nonweighted_error_RMSE+ \
                                                tsk_errorsq[x_join_f, y_join_f].sum()
                            Sum_of_nonweighted_error_MB  = Sum_of_nonweighted_error_MB + \
                                                tsk_error[x_join_f, y_join_f].sum()
                        
                        nonWeighted_MAE [i, icase, imodexpt, iobs] = Sum_of_nonweighted_error_MAE /Total_grids
                        nonWeighted_RMSE[i, icase, imodexpt, iobs] = np.sqrt(Sum_of_nonweighted_error_RMSE/Total_grids)
                        nonWeighted_MB  [i, icase, imodexpt, iobs] = Sum_of_nonweighted_error_MB  /Total_grids
                        Correlation     [i, icase, imodexpt, iobs] = corr



Error_indics_Day   = pd.DataFrame(index = legend)
Error_indics_Night = pd.DataFrame(index = legend)

Error_indics_Day  ['non-weighted MAE' ] = np.nanmean(np.nanmean(np.nanmean(nonWeighted_MAE [:,:,[0,2],:], axis =1), axis =1), axis =1)
Error_indics_Day  ['non-weighted RMSE'] = np.nanmean(np.nanmean(np.nanmean(nonWeighted_RMSE[:,:,[0,2],:], axis =1), axis =1), axis =1)
Error_indics_Day  ['non-weighted MB'  ] = np.nanmean(np.nanmean(np.nanmean(nonWeighted_MB  [:,:,[0,2],:], axis =1), axis =1), axis =1)
Error_indics_Day  ['Corr']              = np.nanmean(np.nanmean(np.nanmean(Correlation     [:,:,[0,2],:], axis =1), axis =1), axis =1)

Error_indics_Night['non-weighted MAE' ] = np.nanmean(np.nanmean(np.nanmean(nonWeighted_MAE [:,:,[1,3],:], axis =1), axis =1), axis =1)
Error_indics_Night['non-weighted RMSE'] = np.nanmean(np.nanmean(np.nanmean(nonWeighted_RMSE[:,:,[1,3],:], axis =1), axis =1), axis =1)
Error_indics_Night['non-weighted MB'  ] = np.nanmean(np.nanmean(np.nanmean(nonWeighted_MB  [:,:,[1,3],:], axis =1), axis =1), axis =1)
Error_indics_Night['Corr']              = np.nanmean(np.nanmean(np.nanmean(Correlation     [:,:,[1,3],:], axis =1), axis =1), axis =1)


# Print numerical results of MAE, RMSE, MB
print('non-weighted MAE')
#np.nanmean(np.nanmean(np.nanmean(nonWeighted_MAE[:,[0,3],:,:], axis =1), axis =1), axis =1)
print('LCZ_D,        LCZ_GLO')
print('Day:  ' +str(np.nanmean(np.nanmean(np.nanmean(nonWeighted_MAE[:,:,[0,2],:], axis =1), axis =1), axis =1)))
print('Night:' +str(np.nanmean(np.nanmean(np.nanmean(nonWeighted_MAE[:,:,[1,3],:], axis =1), axis =1), axis =1)))
print('non-weighted RMSE')
#np.nanmean(np.nanmean(np.nanmean(nonWeighted_RMSE[:,[0,3],:,:], axis =1), axis =1), axis =1)
print('LCZ_D,        LCZ_GLO')
print('Day:  ' +str(np.nanmean(np.nanmean(np.nanmean(nonWeighted_RMSE[:,:,[0,2],:], axis =1), axis =1), axis =1)))
print('Night:' +str(np.nanmean(np.nanmean(np.nanmean(nonWeighted_RMSE[:,:,[1,3],:], axis =1), axis =1), axis =1)))
print('non-weighted MB')
#np.nanmean(np.nanmean(np.nanmean(nonWeighted_MB[:,[0,3],:,:], axis =1), axis =1), axis =1)
print('LCZ_D,        LCZ_GLO')
print('Day:  ' +str(np.nanmean(np.nanmean(np.nanmean(nonWeighted_MB[:,:,[0,2],:], axis =1), axis =1), axis =1)))
print('Night:' +str(np.nanmean(np.nanmean(np.nanmean(nonWeighted_MB[:,:,[1,3],:], axis =1), axis =1), axis =1)))



# Spatial plot for SI fig.2 
mod_LST[np.where(mod_LST < 1)] = np.nan
wrf_LST[np.where(wrf_LST < 1)] = np.nan

plt_legend      = ['MODIS (Day)'  , 'LCZ_D (Day)'  , 'LCZ_GLO (Day)',\
                   'MODIS (Night)', 'LCZ_D (Night)', 'LCZ_GLO (Night)']
plt_LST         = np.zeros((len(plt_legend), nx, ny), float)
plt_LST         = np.zeros((len(plt_legend), nx, ny), float)
plt_LST_abs     = plt_LST
plt_LST_ano     = plt_LST

cmap            = 'rainbow'

plt_LST_abs[0,:,:]   = np.nanmean(np.nanmean(np.nanmean(np.nanmean(mod_LST[:,:,[0,2],:], axis = 0), axis = 0), axis = 0), axis = 0)
plt_LST_abs[1:3,:,:] =            np.nanmean(np.nanmean(np.nanmean(wrf_LST[:,:,[0,2],:], axis = 1), axis = 1), axis = 1) 
plt_LST_abs[3,:,:]   = np.nanmean(np.nanmean(np.nanmean(np.nanmean(mod_LST[:,:,[1,3],:], axis = 0), axis = 0), axis = 0), axis = 0)
plt_LST_abs[4:6,:,:] =            np.nanmean(np.nanmean(np.nanmean(wrf_LST[:,:,[1,3],:], axis = 1), axis = 1), axis = 1) 

# Calculate the Z score (LST - mean) / SD
for i in range(plt_LST_abs.shape[0]):
    plt_LST_ano[i,:,:] = (plt_LST_abs[i,:,:]-\
                          np.nanmean(plt_LST_abs[:, x_join_f, y_join_f], axis = 1)[i]) /\
                          np.nanstd (plt_LST_abs[:, x_join_f, y_join_f], axis = 1)[i]

plt_LST = plt_LST_ano
#plt_LST = plt_LST_abs

plt_name        = 'LST_Validate'
levels          = [np.arange(-3, 3, 0.2)]*len(plt_legend)
#levels          = [np.arange(305, 325.2, 0.2)]*3 + [np.arange(295, 305.2, 0.2)]*3


plt.figure(figsize=(15, 9))
for (iplt, iname) in enumerate(plt_legend):
    # Cases + MOD experiment + time averaged LST
    ax = plt.subplot(2,3,iplt+1, projection=ccrs.PlateCarree())
    # Plot features
    ax.add_feature(cfeature.LAND, color='lightgrey')            # Add land feature 
    states    = NaturalEarthFeature(category = 'cultural', scale = '10m', \
                facecolor = 'none',name = 'admin_1_states_provinces_shp')
    ax.add_feature(states, linewidth =0.5, edgecolor = 'black')
    ax.add_geometries(Reader(countyfile).geometries(), ccrs.PlateCarree(), 
    edgecolor='dimgrey', facecolor='none', linewidth = 0.05)#, hatch='xxxx') # Add shapefiles (states, counties)
    ax.coastlines('10m', linewidth = 0.8)
    
    im = ax.contourf(to_np(lon), to_np(lat), to_np(plt_LST[iplt, :, :]), 10, 
                transform= ccrs.PlateCarree(), 
                levels = levels[iplt], cmap = cmap, extend = 'both')#, norm = norm)#, alpha = 0)
    
    urb = ax.contour(to_np(lon), to_np(lat), to_np(LU), 3, 
                transform= ccrs.PlateCarree(), 
                levels = [30], colors = 'purple', linewidths = 1.0)#, alpha = 0)
    
    plt.colorbar(im, ax=ax, shrink = .9, label='', orientation = 'vertical')
    
    # Setting
    #font = font_manager.FontProperties(family='Calbri', size=6)    # legend resources #weight='bold'
    ax.set_extent([np.min(urb_region_lon), np.max(urb_region_lon), \
    np.min(urb_region_lat), np.max(urb_region_lat)], crs=None)                     # Set map extend    
    gvutil.add_lat_lon_ticklabels   (ax)                            # Add lat lon ticks        
    gvutil.set_axes_limits_and_ticks(ax, xticks=np.arange(-96.0, -94.5, 0.5),
                                         yticks=np.arange(29.5, 31.0, 0.5))
    
    ax.set_title(string.ascii_lowercase[iplt] + ') ' +iname, loc = 'left')

plt.tight_layout()#pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(opath+plt_name+'.png',dpi=dpi, bbox_inches='tight')
plt.close('all')
plt.show()
















