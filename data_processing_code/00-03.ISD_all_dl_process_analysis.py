#-------------------------------------------------------
# ISD_all_dl_process_analysis.py
# 
# Tailor made code for the ISD analysis with WRF results
# Incorporate with 
# - download_ISD.py
# - ISD_processing.py 
# - WRFandISD_UV_tocsv_SailorR.ncl
# - WRFandISD_T2RH2_tocsv_Taylor.ncl
# # Stations csv file created here https://www.ncei.noaa.gov/maps/hourly/
#  
# Kwun Yip Fung
# 10 Sept 2020
#
#--------------------------------------------------------

Data_dl_script_path = "/home1/05898/kf22523/Data_dl_script/"
import sys, os
from dateutil.parser import parse
from subprocess import call
sys.path.append(Data_dl_script_path)
import download_ISD as dlisd
import ISD_processing as isdpro


casetype_list       = ["HW1", "HW2", "HW3", "HW4", "HW5"]

for casetype in casetype_list:
    elif (casetype == "HW1"):
        STIME_TXT_list = ["2017-07-28 00:00:00"]
        ETIME_TXT_list = ["2017-08-01 00:00:00"]
    elif (casetype == 'HW2'):
        STIME_TXT_list = ["2018-07-22 00:00:00"]
        ETIME_TXT_list = ["2018-07-26 00:00:00"]
    elif (casetype == 'HW3'):
        STIME_TXT_list = ["2018-08-20 00:00:00"]
        ETIME_TXT_list = ["2018-08-24 00:00:00"]
    elif (casetype == 'HW4'):
        STIME_TXT_list = ["2019-08-13 00:00:00"]
        ETIME_TXT_list = ["2019-08-17 00:00:00"]
    elif (casetype == 'HW5'):
        STIME_TXT_list = ["2019-09-04 00:00:00"]
        ETIME_TXT_list = ["2019-09-08 00:00:00"]
    
    for (STIME_TXT, ETIME_TXT) in zip(STIME_TXT_list, ETIME_TXT_list):
        print(casetype)
        sdate               = parse(STIME_TXT)
        edate               = parse(ETIME_TXT)
        syyyymmdd           = sdate.strftime("%Y%m%d")
        eyyyymmdd           = edate.strftime("%Y%m%d")
        plotscript_path     = "/home1/05898/kf22523/graphs/scripts/"
        Expt_legend_stor_path = "/scratch/05898/kf22523/data/tmp/"
        cases_dir           = syyyymmdd + "-" + eyyyymmdd + "_" + casetype
        DL_ISD              = True #False#     # Set switch for downloading ISD 
        PROCESS_ISD         = True #False#     # Set switch for processing ISD
        WRITE_CSV_ISD       = True      # Set switch to run ncl script for 
        
        #"WRITE_CSV_ISD" has to be True before setting Below switch as True
        WRITE_CSV_T2RH2_ISD = True     # Set switch to run ncl script for T2RH2 txt  
        WRITE_CSV_UV_ISD    = True     # Set switch to run ncl script for UV txt  
        WRITE_CSV_RAIN_ISD  = False     # Set switch to run ncl script for RAIN txt  
                                        
        wrf_expt_pre_all    =  ['LCZ_D/E5_gr0tc0ar0ag0_d03',\
                                'LCZ_GLO/E5_gr0tc0ar0ag0_d03']
        
        legend              = ['LCZ', 'LCZ+GLO']
                
        
        file = open(Expt_legend_stor_path + "expt_pre.txt","w") 
        for i in wrf_expt_pre_all:
            file.write(i) 
            file.write("\n")  
        
        file.close() 
        
        file = open(Expt_legend_stor_path + "legend.txt","w") 
        for i in legend:
            file.write(i) 
            file.write("\n")  
        
        file.close()
        # User Input for download_ISD and ISD_processing
        station_list_csv_all    = ["Houston_stations_40miles.csv",  "Houston_stations_100-40miles.csv"]
        UrbNUrb_all             = ["Urb",                           "NUrb"]   # Whether the station lists is urban or nonurban (As they saved at different paths)
        # Loop over the Urban and NonUrban stations
        for station_list_csv, UrbNUrb in zip(station_list_csv_all, UrbNUrb_all):
            station_ID_file_input   = "/home1/05898/kf22523/graphs/txt/ISD_identified_cases/" + station_list_csv  #INPUT 3
            station_ID_file_output  = "/scratch/05898/kf22523/data/ISD/"+cases_dir+"/"+UrbNUrb+"/" + station_list_csv.replace('stations', 'valid_stations') # INPUT 4
            ISD_store_path          = "/scratch/05898/kf22523/data/ISD/"+cases_dir+"/"+UrbNUrb+"/"   # path you would like to store the dowload data #INPUT 5
            # download_ISD
            if DL_ISD:
                if not os.path.exists(ISD_store_path):
                    os.makedirs(ISD_store_path)
                
                dlisd.download_ISD(STIME_TXT, ETIME_TXT, station_ID_file_input, station_ID_file_output, ISD_store_path)
            # ISD_processing
            ISD_ipath   = ISD_store_path
            ISD_opath   = ISD_store_path
            if PROCESS_ISD:
                isdpro.ISD_processing (STIME_TXT, ETIME_TXT, ISD_ipath, ISD_opath)
        
        
        # User Input for ouputing the WRF files as csv
        if WRITE_CSV_ISD:
            os.environ["CASES_DIR"]         = cases_dir
            os.chdir(plotscript_path)
            WRITE_OBS                       = "True"    # Set switch for ouputing Obseravtion ISD station csv or not pass to NCL
            
            for EXPT_PRE, EXPT_PRE_LEGEND in zip(wrf_expt_pre_all, legend):
                os.environ["EXPT_PRE"]          = EXPT_PRE
                os.environ["EXPT_PRE_LEGEND"]   = EXPT_PRE_LEGEND
                os.environ["WRITE_OBS"]         = WRITE_OBS
                
                if WRITE_CSV_T2RH2_ISD:
                    call('ncl 02.WRFandISD_T2RH2_tocsv_Taylor.ncl', shell = True)
                if WRITE_CSV_UV_ISD:
                    call('ncl 03.WRFandISD_UV_tocsv_SailorR.ncl',   shell = True)
                
                WRITE_OBS = "False"   # After once, no need to save obs station again and again
        


