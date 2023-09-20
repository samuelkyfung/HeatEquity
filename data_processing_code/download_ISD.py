#-------------------------------------------------------
# download_ISD.py
#
# Kwun Yip Fung (Samuel)
# 29 Apr 2020
# 
# This script will download ISD Integrated Surface Data 
# (Global Hourly) data from the NOAA ftp site and unzip it
#
# Input1:   Start dates would like to download
# Input2:   End dates would like to download
# Input3:   ISD station list cover the desire location 
#           obtained from (https://gis.ncdc.noaa.gov/maps/ncei/cdo/hourly)
# Input4:   Output name of the valid ISD station list cover the desire location
# Input5:   Path to store the download data
#  
# Output1:  Valid ISD station list contain data for the 
# Output2:  Download ISD data for the particular year required
#-------------------------------------------------------
# Module load
import datetime as dt
from dateutil.parser import parse
#import sys
#python_Function_path = '/work/05898/kf22523/stampede2/python_functions/'
#sys.path.append(python_Function_path)
#import Functions as f1
from subprocess import call
import os
import pandas as pd


def download_ISD(STIME_TXT, ETIME_TXT, station_ID_file_input, station_ID_file_output, opath):
    
    url                     = "ftp://ftp.ncdc.noaa.gov/pub/data/noaa/"
    # Check the existence of the output directory
    if os.path.exists(opath):
        os.chdir(opath)
    else:
        print(opath)
        print("Do not exit, creating: " + opath)
        call('mkdir ' + opath, shell = True)
        os.chdir(opath)
    
    df              = pd.read_csv(station_ID_file_input)
    df_sdate        = df['BEGIN_DATE']
    df_edate        = df['END_DATE']
    df_ID           = df['STATION_ID']
    sdate           = parse(STIME_TXT)
    edate           = parse(ETIME_TXT)
    syr             = sdate.year
    eyr             = edate.year
    dlyr            = list(range(syr, eyr+1))
    
    valid_station_ind = []
    for i in range(len(df_sdate)):
        df_sdate_tmp = df_sdate[i]
        df_edate_tmp = df_edate[i]
        df_sdate_tmp_dt = parse(df_sdate_tmp)
        df_edate_tmp_dt = parse(df_edate_tmp)
        
        # Test if the desire date fall within the functioning dates of the stations
        test1           = (sdate - df_sdate_tmp_dt).days  # Both test should be >0 to be a valid station
        test2           = (df_edate_tmp_dt - edate).days  # Both test should be >0 to be a valid station
        
        # Capture the valid station index
        if (test1 > 0 and test2 > 0): 
            valid_station_ind.append(i)
            
            # Download the valid station data
            # First rename the station ID into the file names
            ID      = str(df_ID[i])
            FID_pre = ID[0:6] + "-" + ID[6:11]
            # Loop the years required to download
            for iyr in dlyr:
                FID = FID_pre + "-" + str(iyr) + ".gz"
                call('wget ' + url + str(iyr) + "/" + FID, shell = True) 
                call('gunzip ' + FID, shell = True)
    
    # Output the valid station as csv file
    df_out  = df.iloc[valid_station_ind]
    df_out.to_csv(station_ID_file_output, index = False)



## User Input
#STIME_TXT               = "2018-04-16 00:00:00"   INPUT 1
#ETIME_TXT               = "2018-04-22 00:00:00"   INPUT 2
##STIME_TXT               = "2017-09-05 00:00:00"
##ETIME_TXT               = "2017-09-13 00:00:00"
##STIME_TXT               = "2018-03-21 00:00:00"
##ETIME_TXT               = "2018-03-26 00:00:00"
#syyyymmdd               = parse(STIME_TXT).strftime("%Y%m%d")
#eyyyymmdd               = parse(ETIME_TXT).strftime("%Y%m%d")
#cases                   = syyyymmdd + "-" + eyyyymmdd + "_NoRain"
#station_list_csv        = "Houston_stations_40miles.csv" # or "Houston_stations_100-40miles.csv"
#UrbNUrb                 = "Urb"  # or NUrb   # Whether the station lists is urban or nonurban (As they saved at different paths)
#station_ID_file_input   = "/home1/05898/kf22523/graphs/txt/ISD_identified_cases/" + station_list_csv  #INPUT 3
#station_ID_file_output  = "/scratch/05898/kf22523/data/ISD/"+cases+"/"+UrbNUrb+"/" + station_list_csv.replace('stations', 'valid_stations') # INPUT 4
#opath                   = "/scratch/05898/kf22523/data/ISD/"+cases+"/"+UrbNUrb+"/"   # path you would like to store the dowload data #INPUT 5
#
#download_ISD(STIME_TXT, ETIME_TXT, station_ID_file_input, station_ID_file_output, opath)
#     
