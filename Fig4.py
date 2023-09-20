#-------------------------------------------------------
# Fig4.py
#
# Kwun Yip Fung
# 10 Mar 2023
# 
# - Calculte the street fraction, tree cover, and added 
#   tree cover for urban trees high strategies
# --------------------------------------------------------
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


dpi             = 800
urb_region_lat  = [29.3, 30.5]
urb_region_lon  = [-96, -94.8]
#

# LU   31   32   33   34   35   36   37   38   39   40
#      LCZ1 LCZ2 LCZ3 LCZ4 LCZ5 LCZ6 LCZ7 LCZ8 LCZ9 LCZ10
tco = [10,  20,  0,   14,  13,  22,  0,   16,  30,  14]    # tree coverage original
tcl = [18,  22,  0,   21,  21,  28,  0,   24,  36,  23]    # tree coverage original
tch = [26,  25,  0,   29,  29,  34,  0,   31,  42,  33]    # tree coverage original


opath           = '/home1/05898/kf22523/graphs/graphs/Houston/Heatwaves/'
ipath           = '/scratch/05898/kf22523/data/WRF_out/Houston/Manuscript/'

wrfin_0912      = Dataset(ipath + 'geo_em/LCZ_GLO/geo_em.d03.nc')
nx              = wrfin_0912.dimensions['south_north'].size
ny              = wrfin_0912.dimensions['west_east'].size
lat             = getvar(wrfin_0912, "lat"      , timeidx = 0        , meta = False)
lon             = getvar(wrfin_0912, "lon"      , timeidx = 0        , meta = False)
LU              = getvar(wrfin_0912, "LU_INDEX" , timeidx = 0        , meta = False)
UPARM           = getvar(wrfin_0912, "URB_PARAM", timeidx = 0        , meta = False) # (132, nx, ny)
UFRAC           = getvar(wrfin_0912, "FRC_URB2D", timeidx = 0        , meta = False) # (132, nx, ny)
PAF             = UPARM[90,:,:] # Plan area fraction

# Find the i,j index for the intersted region
ll = ll_to_xy(wrfin_0912, urb_region_lat[0], urb_region_lon[0])
ul = ll_to_xy(wrfin_0912, urb_region_lat[1], urb_region_lon[0])
lr = ll_to_xy(wrfin_0912, urb_region_lat[0], urb_region_lon[1])
ur = ll_to_xy(wrfin_0912, urb_region_lat[1], urb_region_lon[1])
i_s = np.min([ll[1], ul[1], lr[1], ur[1]])
i_e = np.max([ll[1], ul[1], lr[1], ur[1]])
j_s = np.min([ll[0], ul[0], lr[0], ur[0]])
j_e = np.max([ll[0], ul[0], lr[0], ur[0]])

TCO = LU.copy(); TCO[:] = np.nan
TCL = LU.copy(); TCL[:] = np.nan
TCH = LU.copy(); TCH[:] = np.nan
for i, idx in enumerate(range(31, 41)):
    TCO[np.where(LU == idx)] = tco[i]
    TCL[np.where(LU == idx)] = tcl[i]
    TCH[np.where(LU == idx)] = tch[i]

# Read SVI
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


SVI_join        = np.array([shapefile[shapefile.OBJECTID == i].RPL_THEMES.values[0] for i in joined.OBJECTID])
# Remove index where SVI < 0
SVI             = SVI_join[np.where(SVI_join >= 0)]
y_join_f        = y_join  [np.where(SVI_join >= 0)]
x_join_f        = x_join  [np.where(SVI_join >= 0)]

# SVI components
df                = pd.DataFrame()
df['OBJECTID'   ] = joined.OBJECTID.values
df['I_SVI'      ] = np.array([shapefile[shapefile.OBJECTID == i].RPL_THEMES.values[0] for i in joined.OBJECTID])
df = df.iloc[np.where(SVI_join >= 0)].reset_index(drop=True)
df_census_tract = df.groupby('OBJECTID').mean().reset_index(drop = False)


# Just get the 
df['PAF']               = PAF[x_join_f, y_join_f]*100
df['UFRAC']             = UFRAC[x_join_f, y_join_f]
df['SF' ]               = (100 - df['PAF'])*df['UFRAC']
df['LU' ]               = LU [x_join_f, y_join_f]
df['TCO']               = TCO[x_join_f, y_join_f]
df['TCL']               = TCL[x_join_f, y_join_f]
df['TCH']               = TCH[x_join_f, y_join_f]
df['addedTCH']          = TCH[x_join_f, y_join_f] - TCO[x_join_f, y_join_f]
df_census_tract_PAF     = df.groupby('OBJECTID').mean().reset_index(drop = False)



#---------- Plot Bar Chart FOR Tree coverage original -----------# Fig. 4
# Plot bar chart of SVI with 0.1 as bin and dT as y-axis
plt.figure(figsize=(12,10))
plt_name = 'Nature_Bar.regression.LCZ_GLO.TreeCoverageOriginalAndAdded.vs.SVI'
print('Drawing bar plot: ' + plt_name)

ax = plt.subplot(2,2,1)
ax.plot([],[])
ax.set_title('a) Schematic diagram of Street fraction', loc = 'left', fontsize = 12, pad = 10)

ax = plt.subplot(2,2,2)
Bar_Whole_df            = pd.DataFrame()
Bar_Whole_df            = df_census_tract_PAF[['OBJECTID']]
Bar_Whole_df['I_SVI']   = df_census_tract_PAF['I_SVI']#
Bar_Whole_df['SF'   ]   = df_census_tract_PAF['SF' ]#
# Bin the data into intervals of 0.1 using pd.cut()
Bar_Whole_df['Interval'] = pd.cut(Bar_Whole_df['I_SVI'], bins=[i/10 for i in range(0, 11)])
# Group the data by the intervals and Mean of the counts
grouped = Bar_Whole_df.groupby('Interval')['SF'].mean()
q3      = Bar_Whole_df.groupby('Interval')['SF'].quantile(0.75)
q1      = Bar_Whole_df.groupby('Interval')['SF'].quantile(0.25)
iqr     = (q3-q1)/2
ax = grouped.plot(kind='bar', yerr = iqr, width=0.8, capsize=3, error_kw=dict(lw=0.8, capthick=0.8))
ax.set_xlabel('SVI')
ax.set_ylabel('Street fraction (%)')
ax.set_title('b) Street fraction against SVI', loc = 'left', fontsize = 12, pad = 10)


ax = plt.subplot(2,2,3)
Bar_Whole_df            = pd.DataFrame()
Bar_Whole_df            = df_census_tract_PAF[['OBJECTID']]
Bar_Whole_df['I_SVI']   = df_census_tract_PAF['I_SVI']#
Bar_Whole_df['Current Tree Coverage'  ]   = df_census_tract_PAF['TCO']#
Bar_Whole_df['Added Tree Coverage']= df_census_tract_PAF['addedTCH']
# Bin the data into intervals of 0.1 using pd.cut()
Bar_Whole_df['Interval'] = pd.cut(Bar_Whole_df['I_SVI'], bins=[i/10 for i in range(0, 11)])
# Group the data by the intervals and Mean of the counts
grouped = Bar_Whole_df.groupby('Interval')[['Current Tree Coverage', 'Added Tree Coverage']].mean()
q3      = Bar_Whole_df.groupby('Interval')[['Current Tree Coverage', 'Added Tree Coverage']].quantile(0.75)
q1      = Bar_Whole_df.groupby('Interval')[['Current Tree Coverage', 'Added Tree Coverage']].quantile(0.25)
iqr     = (q3-q1)/2
ax = grouped['Current Tree Coverage'].plot(kind='bar', yerr = iqr['Current Tree Coverage'], width=0.8, capsize=3, error_kw=dict(lw=0.8, capthick=0.8))
ax.set_xlabel('SVI')
ax.set_ylabel('Tree coverage (%)')
#ax.set_ylim([19, 22.5])
ax.set_ylim([18, 23.5])
ax.set_title('c) Current Tree Cover', fontsize=12, pad=10, loc = 'left')


ax = plt.subplot(2,2,4)
ax = grouped['Added Tree Coverage'].plot(kind = 'bar', yerr = iqr['Added Tree Coverage'], width=0.8, capsize=3, error_kw=dict(lw=0.8, capthick=0.8))
ax.set_xlabel('SVI')
ax.set_ylabel('Added Tree coverage (%)')
#ax.set_ylim([12, 13.1])
ax.set_ylim([11.5, 14.1])
ax.set_title('d) Added Tree Cover', fontsize=12, pad=10, loc = 'left')


plt.suptitle('SVI'+' vs '+'Tree Coverage')
SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.tight_layout(pad=1.5, w_pad=4, h_pad=4)#pad=0.4, w_pad=0.5, h_pad=1.0)
#plt.savefig(opath + plt_name + '.png', format = 'png', dpi=dpi, transparent=False, bbox_inches='tight')
#plt.close()
plt.show()

