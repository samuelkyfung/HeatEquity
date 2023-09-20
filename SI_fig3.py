#-------------------------------------------------------
# SI_fig3.py
#
# Kwun Yip Fung
# 20 Aug 2022
#
# - Evaluate WRF simulations against ISD station data
#--------------------------------------------------------
import Ngl
import numpy as np
from glob import glob
import pandas as pd
from dateutil.parser import parse
import os
import matplotlib.pyplot as plt
import itertools
import string

ipath           = '/scratch/05898/kf22523/data/WRF_out/Houston/Manuscript/'
opath           = '/home1/05898/kf22523/graphs/graphs/Houston/Heatwaves/Manuscript_Fig/'

UrbNUrb           = ['NUrb', 'Urb']
dpi               = 800
# Read files
cases             = ["20170728-20170801_HW1", "20180722-20180726_HW2", "20180820-20180824_HW3", "20190813-20190817_HW4", "20190904-20190908_HW5"]
STIME_TXT_all     = ["2017-07-28 00:00:00"  , "2018-07-22 00:00:00"  , "2018-08-20 00:00:00"  , "2019-08-13 00:00:00"  , "2019-09-04 00:00:00"  ]
ETIME_TXT_all     = ["2017-08-01 00:00:00"  , "2018-07-26 00:00:00"  , "2018-08-24 00:00:00"  , "2019-08-17 00:00:00"  , "2019-09-08 00:00:00"  ]
csv_legend   = ['LCZ-',\
                'LCZ+GLO-']

# Get day and nighttime index
ndays       = (parse(ETIME_TXT_all[0]) - parse(STIME_TXT_all[0])).days
UTC         = -6
daytime     = [np.arange((iday)*24+12, (iday)*24+24) for iday in range(ndays)]
nighttime   = [np.arange((iday)*24+0 , (iday)*24+12) for iday in range(ndays)]
daytime     = list(itertools.chain.from_iterable(daytime))
nighttime   = list(itertools.chain.from_iterable(nighttime))

def KGE(x, y, s1 = 1.0, s2 = 1.0, s3 = 1.0):
    import numpy as np
    
    R           = np.corrcoef(x, y)[0,1]
    Mean_obs    = np.mean(x)
    Mean_sim    = np.mean(y)
    SD_obs      = np.std (x)
    SD_sim      = np.std (y)
    
    # Coefficient for correlation component
    s1 = 1.0
    # Coefficient for SD component
    s2 = 1.0
    # Coefficient for mean component
    s3 = 1.0
    
    C1  = s1*(R-1)**2
    C2  = s2*((SD_sim/SD_obs)-1)**2
    C3  = s3*((Mean_sim/Mean_obs)-1)**2
    ED  = np.sqrt(C1+C2+C3)
    kge = 1-ED
    
    return kge

def return_indices_of_a(a, b):
  b_set = set(b)
  return [i for i, v in enumerate(a) if v in b_set]



# Read the ISD stations
df_t2_UrbNUrb   = {}
df_rh2_UrbNUrb  = {}
Obs_df_t2_dict  = {}    # For holding station hourly data
Obs_df_rh2_dict = {}    # For holding station hourly data
Mod_df_t2_dict  = {}    # For holding station hourly data
Mod_df_rh2_dict = {}    # For holding station hourly data

for iurb in range(len(UrbNUrb)): #range(1): #
    df_t2_allcases  = {}
    df_rh2_allcases = {}
    
    Obs_df_t2_dict_allcases  = {}
    Obs_df_rh2_dict_allcases = {}
    Mod_df_t2_dict_allcases  = {}
    Mod_df_rh2_dict_allcases = {}
    
    for (case,STIME_TXT,ETIME_TXT) in zip(cases, STIME_TXT_all, ETIME_TXT_all):
        print("case: " + case)
        ipath = ipath + "/ISD_WRF_T2RH2/"+case+"/"+UrbNUrb[iurb]+"/"
        sdate = parse(STIME_TXT)
        edate = parse(ETIME_TXT)
        
        model_csv  = [sorted(glob (ipath + ilegend + "*-WRF-T2RH2-WRFtime.csv")) for ilegend in csv_legend]
        station_id = [icsv.replace(ipath, '').replace(csv_legend[0],'').replace('-WRF-T2RH2-WRFtime.csv','') for icsv in model_csv[0]]
                
        Obs_df_t2_dict_allstations      = pd.DataFrame()
        Obs_df_rh2_dict_allstations     = pd.DataFrame()
        Mod_df_t2_dict_allstations_dict = {}
        Mod_df_rh2_dict_allstations_dict= {}
        for istation in station_id:
            print('Station: ' + istation)            
            model_csv_this = [glob(ipath + ilegend + istation +"*-WRF-T2RH2-WRFtime.csv")[0] for ilegend in csv_legend]
            obs_csv_this   = glob(ipath + istation +"*-ISD-T2RH2-WRFtime.csv")[0]
            
            obs_df_tmp         = pd.read_csv (obs_csv_this, parse_dates = [0])
            obs_df_tmp['time'] = obs_df_tmp['time'].dt.round('H')
            obs_df_tmp         = obs_df_tmp.drop_duplicates(subset = ['time']).reset_index(drop=True)
            
            # Data cleaning of observation station data
            obs_df             = pd.DataFrame(columns = ['time', ' T2', ' RH2'])
            obs_df['time']     = pd.date_range(start=sdate, end=edate, freq = 'H')
            
            obs_df [' T2'].iloc[return_indices_of_a(obs_df['time'], obs_df_tmp['time'])] = obs_df_tmp [' T2'].iloc[return_indices_of_a(obs_df_tmp['time'], obs_df['time'])].to_list()
            obs_df[' RH2'].iloc[return_indices_of_a(obs_df['time'], obs_df_tmp['time'])] = obs_df_tmp[' RH2'].iloc[return_indices_of_a(obs_df_tmp['time'], obs_df['time'])].to_list()
            
            obsstd_t2  = np.nanstd(obs_df[' T2'].to_list())
            obsstd_rh2 = np.nanstd(obs_df[' RH2'].to_list())
            
            Obs_df_t2_dict_allstations [istation]   = obs_df [' T2' ]
            Obs_df_rh2_dict_allstations[istation]   = obs_df [' RH2']
            Obs_df_t2_dict_allstations.index        = obs_df.time.index
            Obs_df_rh2_dict_allstations.index       = obs_df.time.index
            
            Mod_df_t2_dict_allexp   = pd.DataFrame(index = Obs_df_t2_dict_allstations.index)
            Mod_df_rh2_dict_allexp  = pd.DataFrame(index = Obs_df_rh2_dict_allstations.index)
            
            for i in range(len(csv_legend)):
                mod_df_tmp   = pd.read_csv (model_csv_this[i], parse_dates = [0])
                # Data cleaning 
                mod_df             = pd.DataFrame(columns = ['time', ' T2', ' RH2'])
                mod_df['time']     = pd.date_range(start=sdate, end=edate, freq = 'H')
                mod_df [' T2'].iloc[return_indices_of_a(mod_df['time'], mod_df_tmp['time'])] = mod_df_tmp [' T2'].iloc[return_indices_of_a(mod_df_tmp['time'], mod_df['time'])].reset_index(drop = True)
                mod_df[' RH2'].iloc[return_indices_of_a(mod_df['time'], mod_df_tmp['time'])] = mod_df_tmp[' RH2'].iloc[return_indices_of_a(mod_df_tmp['time'], mod_df['time'])].reset_index(drop = True)
                
                Mod_df_t2_dict_allexp [csv_legend[i]] = mod_df [' T2' ].values
                Mod_df_rh2_dict_allexp[csv_legend[i]] = mod_df [' RH2'].values
                    
                
                # Extract only non-missing value
                valid_index_t2  = np.where(np.array(obs_df[' T2' ].to_list()) > 0)  # non-missing value
                valid_index_rh2 = np.where(np.array(obs_df[' RH2'].to_list()) > 0)  # non-missing value
                t2_mod  = list(np.array(mod_df[' T2' ].to_list())[valid_index_t2 ])
                rh2_mod = list(np.array(mod_df[' RH2'].to_list())[valid_index_rh2])
                t2_obs  = list(np.array(obs_df[' T2' ].to_list())[valid_index_t2 ])
                rh2_obs = list(np.array(obs_df[' RH2'].to_list())[valid_index_rh2])
            
            Mod_df_t2_dict_allstations_dict[istation]  = Mod_df_t2_dict_allexp
            Mod_df_rh2_dict_allstations_dict[istation] = Mod_df_rh2_dict_allexp
        
        Mod_df_t2_dict_allstations  = pd.concat(Mod_df_t2_dict_allstations_dict , axis=1)
        Mod_df_rh2_dict_allstations = pd.concat(Mod_df_rh2_dict_allstations_dict, axis=1)
        Mod_df_t2_dict_allstations.columns  = Mod_df_t2_dict_allstations.columns.swaplevel(1, 0)
        Mod_df_rh2_dict_allstations.columns = Mod_df_rh2_dict_allstations.columns.swaplevel(1, 0)
        Mod_df_t2_dict_allstations  = Mod_df_t2_dict_allstations.sort_index(axis=1)
        Mod_df_rh2_dict_allstations = Mod_df_rh2_dict_allstations.sort_index(axis=1)
        
        Obs_df_t2_dict_allcases[case]  = Obs_df_t2_dict_allstations
        Obs_df_rh2_dict_allcases[case] = Obs_df_rh2_dict_allstations
        Mod_df_t2_dict_allcases[case]  = Mod_df_t2_dict_allstations
        Mod_df_rh2_dict_allcases[case] = Mod_df_rh2_dict_allstations
    
    Obs_df_t2_dict[UrbNUrb[iurb]] = pd.concat(Obs_df_t2_dict_allcases , axis=1)
    Obs_df_rh2_dict[UrbNUrb[iurb]]= pd.concat(Obs_df_rh2_dict_allcases, axis=1)    
    Mod_df_t2_dict[UrbNUrb[iurb]] = pd.concat(Mod_df_t2_dict_allcases , axis=1)
    Mod_df_rh2_dict[UrbNUrb[iurb]]= pd.concat(Mod_df_rh2_dict_allcases, axis=1)  

# Set the missing value in observation into np.nan
Obs_df_rh2_dict['Urb' ].iloc[np.where(Obs_df_rh2_dict['Urb' ] < 0)] = np.nan
Obs_df_rh2_dict['NUrb'].iloc[np.where(Obs_df_rh2_dict['NUrb'] < 0)] = np.nan

# Move the experiment name to the upper most level
Mod_df_t2_dict ['Urb' ].columns = Mod_df_t2_dict ['Urb' ].columns.swaplevel(1, 0)
Mod_df_rh2_dict['Urb' ].columns = Mod_df_rh2_dict['Urb' ].columns.swaplevel(1, 0)
Mod_df_t2_dict ['NUrb'].columns = Mod_df_t2_dict ['NUrb'].columns.swaplevel(1, 0)
Mod_df_rh2_dict['NUrb'].columns = Mod_df_rh2_dict['NUrb'].columns.swaplevel(1, 0)

# Sort the columns
Mod_df_t2_dict ['Urb' ]         = Mod_df_t2_dict ['Urb' ].sort_index(axis = 1)
Mod_df_rh2_dict['Urb' ]         = Mod_df_rh2_dict['Urb' ].sort_index(axis = 1)
Mod_df_t2_dict ['NUrb']         = Mod_df_t2_dict ['NUrb'].sort_index(axis = 1)
Mod_df_rh2_dict['NUrb']         = Mod_df_rh2_dict['NUrb'].sort_index(axis = 1)
Obs_df_t2_dict ['Urb' ]         = Obs_df_t2_dict ['Urb' ].sort_index(axis = 1)
Obs_df_rh2_dict['Urb' ]         = Obs_df_rh2_dict['Urb' ].sort_index(axis = 1)
Obs_df_t2_dict ['NUrb']         = Obs_df_t2_dict ['NUrb'].sort_index(axis = 1)
Obs_df_rh2_dict['NUrb']         = Obs_df_rh2_dict['NUrb'].sort_index(axis = 1)



# Calculate T2 RMSE, MB, Normalized SD, Correlation of Day, Night, whole day
Metrics_t2_Day    = {}
Metrics_t2_Night  = {}
Metrics_t2_Whole  = {}

for iurb in range(len(UrbNUrb)): #range(1): #
    Metrics_t2_UrbNUrb_Day    = pd.DataFrame(index = np.sort(csv_legend), columns = ['RMSE', 'MB', 'SDNORM', 'CORR'])
    Metrics_t2_UrbNUrb_Night  = pd.DataFrame(index = np.sort(csv_legend), columns = ['RMSE', 'MB', 'SDNORM', 'CORR'])
    Metrics_t2_UrbNUrb_Whole  = pd.DataFrame(index = np.sort(csv_legend), columns = ['RMSE', 'MB', 'SDNORM', 'CORR'])
    
    for ilegend in np.sort(csv_legend):
        Merge_data          = pd.concat([Mod_df_t2_dict [UrbNUrb[iurb]][ilegend], Obs_df_t2_dict [UrbNUrb[iurb]]], axis=1, keys=['modelled', 'observed'], names=['level_0', 'level_1', 'level_2'])
        #first interpolate the missing value (just for correaltion calculation)
        Merge_data['observed'] = ((Merge_data['observed'].interpolate(method='linear', limit_direction='both', axis=0).fillna(method='bfill', inplace=False) +\
                                   Merge_data['observed'].interpolate(method='linear', limit_direction='both', axis=0).fillna(method='ffill', inplace=False))/2).fillna(method='bfill', inplace=False)
        # RMSE      
        rmse_daytime        = np.sqrt(((Merge_data['modelled'] - Merge_data['observed']) ** 2).mean(axis = 1).iloc[daytime  ].mean())
        rmse_nighttime      = np.sqrt(((Merge_data['modelled'] - Merge_data['observed']) ** 2).mean(axis = 1).iloc[nighttime].mean())
        rmse_wholeday       = np.sqrt(((Merge_data['modelled'] - Merge_data['observed']) ** 2).mean(axis = 1).mean())
        # MB        
        mb_data             = (Merge_data['modelled'] - Merge_data['observed']).mean(axis = 1)
        mb_daytime          = mb_data[daytime  ].mean()
        mb_nighttime        = mb_data[nighttime].mean()
        mb_wholeday         = mb_data.mean()
        # Normalized SD
        sdnorm_daytime      =(Merge_data['modelled'].iloc[daytime  ].std() / Merge_data['observed'].iloc[daytime  ].std()).mean(axis = 0)
        sdnorm_nighttime    =(Merge_data['modelled'].iloc[nighttime].std() / Merge_data['observed'].iloc[nighttime].std()).mean(axis = 0)
        sdnorm_wholeday     =(Merge_data['modelled'].std() / Merge_data['observed'].std()).mean(axis = 0)
        # Correlation
        corr_daytime   = Merge_data['modelled'].iloc[daytime  ].corrwith(Merge_data['observed'].iloc[daytime  ], axis = 0).mean()
        corr_nighttime = Merge_data['modelled'].iloc[nighttime].corrwith(Merge_data['observed'].iloc[nighttime], axis = 0).mean()
        corr_wholeday  = Merge_data['modelled'].corrwith(Merge_data['observed'], axis = 0).mean()
        
        # Save metrics
        Metrics_t2_UrbNUrb_Day.loc  [ilegend, 'RMSE'  ] = rmse_daytime  
        Metrics_t2_UrbNUrb_Night.loc[ilegend, 'RMSE'  ] = rmse_nighttime
        Metrics_t2_UrbNUrb_Whole.loc[ilegend, 'RMSE'  ] = rmse_wholeday 
        Metrics_t2_UrbNUrb_Day.loc  [ilegend, 'MB'    ] = mb_daytime  
        Metrics_t2_UrbNUrb_Night.loc[ilegend, 'MB'    ] = mb_nighttime
        Metrics_t2_UrbNUrb_Whole.loc[ilegend, 'MB'    ] = mb_wholeday 
        Metrics_t2_UrbNUrb_Day.loc  [ilegend, 'SDNORM'] = sdnorm_daytime  
        Metrics_t2_UrbNUrb_Night.loc[ilegend, 'SDNORM'] = sdnorm_nighttime
        Metrics_t2_UrbNUrb_Whole.loc[ilegend, 'SDNORM'] = sdnorm_wholeday 
        Metrics_t2_UrbNUrb_Day.loc  [ilegend, 'CORR'  ] = corr_daytime  
        Metrics_t2_UrbNUrb_Night.loc[ilegend, 'CORR'  ] = corr_nighttime
        Metrics_t2_UrbNUrb_Whole.loc[ilegend, 'CORR'  ] = corr_wholeday 
    
    Metrics_t2_Day  [UrbNUrb[iurb]] = Metrics_t2_UrbNUrb_Day  
    Metrics_t2_Night[UrbNUrb[iurb]] = Metrics_t2_UrbNUrb_Night
    Metrics_t2_Whole[UrbNUrb[iurb]] = Metrics_t2_UrbNUrb_Whole


# Calculate RH2 RMSE, MB, Normalized SD, Correlation of Day, Night, whole day
Metrics_rh2_Day    = {}
Metrics_rh2_Night  = {}
Metrics_rh2_Whole  = {}


for iurb in range(len(UrbNUrb)): #range(1): #
    Metrics_rh2_UrbNUrb_Day    = pd.DataFrame(index = np.sort(csv_legend), columns = ['RMSE', 'MB', 'SDNORM', 'CORR'])
    Metrics_rh2_UrbNUrb_Night  = pd.DataFrame(index = np.sort(csv_legend), columns = ['RMSE', 'MB', 'SDNORM', 'CORR'])
    Metrics_rh2_UrbNUrb_Whole  = pd.DataFrame(index = np.sort(csv_legend), columns = ['RMSE', 'MB', 'SDNORM', 'CORR'])
    
    for ilegend in np.sort(csv_legend):
        Merge_data          = pd.concat([Mod_df_rh2_dict [UrbNUrb[iurb]][ilegend], Obs_df_rh2_dict [UrbNUrb[iurb]]], axis=1, keys=['modelled', 'observed'], names=['level_0', 'level_1', 'level_2'])
        # First interpolate the missing value (just for correaltion calculation)
        Merge_data['observed'] = ((Merge_data['observed'].interpolate(method='linear', limit_direction='both', axis=0).fillna(method='bfill', inplace=False) +\
                                   Merge_data['observed'].interpolate(method='linear', limit_direction='both', axis=0).fillna(method='ffill', inplace=False))/2).fillna(method='bfill', inplace=False)
        df_obs = Merge_data['observed'].dropna(axis=1, how='any')   # remove if all nan in columns
        df_obs = df_obs.loc[:, ~(df_obs == 100).all(axis=0)]        # remove if all == 100
        df_mod = Merge_data['modelled'][Merge_data['observed'].columns.to_list()]
        Merge_data          = pd.concat([df_mod, df_obs], axis=1, keys=['modelled', 'observed'], names=['level_0', 'level_1', 'level_2'])
        
        # RMSE      
        rmse_daytime        = np.sqrt(((Merge_data['modelled'] - Merge_data['observed']) ** 2).mean(axis = 1).iloc[daytime  ].mean())
        rmse_nighttime      = np.sqrt(((Merge_data['modelled'] - Merge_data['observed']) ** 2).mean(axis = 1).iloc[nighttime].mean())
        rmse_wholeday       = np.sqrt(((Merge_data['modelled'] - Merge_data['observed']) ** 2).mean(axis = 1).mean())
        # MB        
        mb_data             = (Merge_data['modelled'] - Merge_data['observed']).mean(axis = 1)
        mb_daytime          = mb_data[daytime  ].mean()
        mb_nighttime        = mb_data[nighttime].mean()
        mb_wholeday         = mb_data.mean()
        # Normalized SD
        sdnorm_daytime      =(Merge_data['modelled'].iloc[daytime  ].std() / Merge_data['observed'].iloc[daytime  ].std()).mean(axis = 0)
        sdnorm_nighttime    =(Merge_data['modelled'].iloc[nighttime].std() / Merge_data['observed'].iloc[nighttime].std()).mean(axis = 0)
        sdnorm_wholeday     =(Merge_data['modelled'].std() / Merge_data['observed'].std()).mean(axis = 0)
        # Correlation
        corr_daytime   = Merge_data['modelled'].iloc[daytime  ].corrwith(Merge_data['observed'].iloc[daytime  ], axis = 0).mean()
        corr_nighttime = Merge_data['modelled'].iloc[nighttime].corrwith(Merge_data['observed'].iloc[nighttime], axis = 0).mean()
        corr_wholeday  = Merge_data['modelled'].corrwith(Merge_data['observed'], axis = 0).mean()
        
        # Save metrics
        Metrics_rh2_UrbNUrb_Day.loc  [ilegend, 'RMSE'  ] = rmse_daytime  
        Metrics_rh2_UrbNUrb_Night.loc[ilegend, 'RMSE'  ] = rmse_nighttime
        Metrics_rh2_UrbNUrb_Whole.loc[ilegend, 'RMSE'  ] = rmse_wholeday 
        Metrics_rh2_UrbNUrb_Day.loc  [ilegend, 'MB'    ] = mb_daytime  
        Metrics_rh2_UrbNUrb_Night.loc[ilegend, 'MB'    ] = mb_nighttime
        Metrics_rh2_UrbNUrb_Whole.loc[ilegend, 'MB'    ] = mb_wholeday 
        Metrics_rh2_UrbNUrb_Day.loc  [ilegend, 'SDNORM'] = sdnorm_daytime  
        Metrics_rh2_UrbNUrb_Night.loc[ilegend, 'SDNORM'] = sdnorm_nighttime
        Metrics_rh2_UrbNUrb_Whole.loc[ilegend, 'SDNORM'] = sdnorm_wholeday 
        Metrics_rh2_UrbNUrb_Day.loc  [ilegend, 'CORR'  ] = corr_daytime  
        Metrics_rh2_UrbNUrb_Night.loc[ilegend, 'CORR'  ] = corr_nighttime
        Metrics_rh2_UrbNUrb_Whole.loc[ilegend, 'CORR'  ] = corr_wholeday 
    
    Metrics_rh2_Day  [UrbNUrb[iurb]] = Metrics_rh2_UrbNUrb_Day  
    Metrics_rh2_Night[UrbNUrb[iurb]] = Metrics_rh2_UrbNUrb_Night
    Metrics_rh2_Whole[UrbNUrb[iurb]] = Metrics_rh2_UrbNUrb_Whole



# Plot for manuscript
UrbNUrb = ['NUrb', 'Urb']
plt.figure(figsize=(16, 8))

for iplt, iurb in enumerate(UrbNUrb):
    ax1 = plt.subplot(1,2,iplt+1)
    #fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()  
    
    l1 = ax1.plot(Obs_df_t2_dict[iurb].mean(1)                , label = 'Obs (T2)'      , color = 'black', linewidth = 5)
    l2 = ax1.plot(Mod_df_t2_dict[iurb]['LCZ-'        ].mean(1), label = 'LCZ_D (T2)'    , color = 'red')
    l3 = ax1.plot(Mod_df_t2_dict[iurb]['LCZ+GLO-'    ].mean(1), label = 'LCZ_GLO (T2)'  , color = 'blue')
    
    l4 = ax2.plot(Obs_df_rh2_dict[iurb].mean(1)                , label = 'Obs (RH2)'    , color = 'black', linewidth = 5, linestyle = ':')
    l5 = ax2.plot(Mod_df_rh2_dict[iurb]['LCZ-'        ].mean(1), label = 'LCZ_D (RH2)'  , color = 'red', linestyle = ':')
    l6 = ax2.plot(Mod_df_rh2_dict[iurb]['LCZ+GLO-'    ].mean(1), label = 'LCZ_GLO (RH2)', color = 'blue', linestyle = ':')
    
    lns  = l1+l2+l3+l4+l5+l6
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, ncol = 2)
    
    ax2.set_ylim(40, 100 )
    ax1.set_ylim(297, 312 )
    ax1.set_xlim(0  , 24*4)
    ax1.set_ylabel('K', rotation=0)
    ax2.set_ylabel('%', rotation=0)
    ax1.set_xlabel('Hours')
    ax1.set_xticks(np.arange(0, 24*4+1, 12))
    ax1.axvspan(0 , 12, alpha=0.3, color='grey')
    ax1.axvspan(24, 36, alpha=0.3, color='grey')
    ax1.axvspan(48, 60, alpha=0.3, color='grey')
    ax1.axvspan(72, 84, alpha=0.3, color='grey')
    ax1.set_xticklabels(['Day0\n0Z', 'Day0\n12Z', 'Day1\n0Z', 'Day1\n12Z', \
                         'Day2\n0Z', 'Day2\n12Z', 'Day3\n0Z', 'Day3\n12Z', 'Day4\n0Z'])
    ax1.set_title(string.ascii_lowercase[iplt] +') '+iurb+' stations', loc = 'left')

plt.tight_layout()#pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig(opath+'T2m+RH2_allstation_NUrb+Urb'+'.png',dpi=dpi, bbox_inches='tight')
#plt.close('all')

plt.show()




