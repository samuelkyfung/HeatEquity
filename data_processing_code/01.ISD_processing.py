#-------------------------------------------------------
# ISD_processing.py
#
# Kwun Yip Fung (Samuel)
# 29 Apr 2020
# 
# This script will process ISD Integrated Surface Data 
# (Global Hourly) data to csv file
#
# Input1:   Path contain the ISD data (ended with -YYYY)
# Input2:   Start and End dates would like to download
# Input3:   Path to store the download data
#  
# Output1:  csv files for stations contain sdate and edate 
#-------------------------------------------------------

# Module load
import pandas as pd
import csv
import numpy as np
from dateutil.parser import parse
import glob

def ISD_processing (STIME_TXT, ETIME_TXT, ipath, opath):
    # Preprocessing
    listfiles_tmp   = []
    sdate           = parse(STIME_TXT)
    edate           = parse(ETIME_TXT)
    syr             = sdate.year
    eyr             = edate.year
    dlyr            = list(range(syr, eyr+1))
    sYYYYMMDD       = sdate.strftime('%Y%m%d')
    eYYYYMMDD       = edate.strftime('%Y%m%d')
    
    # Get a list of station data files required (end with -YYYY)
    for iyr in dlyr:
        tmp_list = glob.glob(ipath + "*-" + str(iyr))
        listfiles_tmp = listfiles_tmp+tmp_list
    
    # Remove the paths in front of each name
    # it will look like this ['720594-00188-2019', '720637-00223-2019', '722427-12975-2019']
    listfiles = [sub.replace(ipath, '') for sub in listfiles_tmp] 
    
    # Processing
    for file_ind in range(len(listfiles)):
        ifile = listfiles[file_ind]
        ofile = ifile + "_" + sYYYYMMDD + "-" + eYYYYMMDD + ".csv"
        print("Writing " + ofile)
        
        # Open input file
        with open(ipath+ifile, newline='', encoding='ISO-8859-1') as f:
            reader = csv.reader(f)
            df = list(reader)
        
        dims = len(df)              # No. of data point
        df_new = pd.DataFrame({})   # Initialize DataFrame for storing data as table and export to csv
        
        TOTAL_CHAR      = []
        USAF_ID         = []
        WBAN_ID         = []
        YYYYMMDD        = []
        HHMM            = []
        YYYYMMDDHHMM    = []
        OBS_SOURCE      = []
        LAT             = []
        LON             = []
        REPORT          = []
        ELEV            = []
        IDENTIFY        = []
        OBS_QC          = []
        WIND_ANG        = []
        WIND_ANG_QC     = []
        WIND_OBS_TY     = []
        WIND_SPD        = []
        WIND_SPD_QC     = []
        LOW_CLO         = []
        LOW_CLO_QC      = []
        LOW_CLO_TY      = []
        LOW_CLO_RP      = []
        VISIBTY         = []
        VISIBTY_QC      = []
        VISIBTY_VAR     = []
        VISIBTY_VAR_QC  = []
        TC              = []
        TC_QC           = []
        DEWTC           = []
        DEWTC_QC        = []
        SLP             = []
        SLP_QC          = []
        ADD             = []
        oneHR_RAIN_MM   = []
        threeHR_RAIN_MM = []
        sixHR_RAIN_MM   = []
        
        # Loop for each data point (usually hourly or half hourly data)
        # The string location representaion according to the ISD metadata
        for i in range(dims):
            #print(str(i) + "/" + str(dims-1))
            string = str(df[i])[2:-2]
            TOTAL_CHAR.append      ( str  (string[0:4])         ) 
            USAF_ID.append         (   string[4:10]             ) 
            WBAN_ID.append         (   string[10:15]            ) 
            YYYYMMDD.append        (   string[15:23]            ) 
            HHMM.append            (   string[23:27]            ) 
            YYYYMMDDHHMM.append    (   string[15:27]            ) 
            OBS_SOURCE.append      (   string[27:28]            ) 
            LAT.append             ( str(  int(string[28:34]) / 1000) )
            LON.append             ( str(  int(string[34:41]) / 1000) )
            REPORT.append          (   string[41:46]            ) 
            ELEV.append            (   string[46:51]            ) 
            IDENTIFY.append        (   string[51:56]            ) 
            OBS_QC.append          (   string[56:60]            ) 
            WIND_ANG.append        (   (string[60:63])       ) 
            WIND_ANG_QC.append     (   (string[63:64])       ) 
            WIND_OBS_TY.append     (   string[64:65]            ) 
            WIND_SPD.append        ( str(  int(string[65:69]) /10   ) )                             #ms-1
            WIND_SPD_QC.append     (   (string[69:70])       )                      
            LOW_CLO.append         (   (string[70:75])       )                              #m max: 22000
            LOW_CLO_QC.append      (   (string[75:76])       )                      
            LOW_CLO_TY.append      (   string[76:77]            )                  
            LOW_CLO_RP.append      (   string[77:78]            )                              #if CAVOK reported
            VISIBTY.append         (   (string[78:84])       )                              #m
            VISIBTY_QC.append      (   (string[84:85])       )                              #m
            VISIBTY_VAR.append     (   string[85:86]            )                  
            VISIBTY_VAR_QC.append  (   string[86:87]            )                  
            TC.append              ( str(  int(string[87:92])/10    ) )                             #deg C
            TC_QC.append           (   string[92:93]            )                  
            DEWTC.append           ( str(  int(string[93:98])/10    ) )                             #deg CAVOK
            DEWTC_QC.append        (   string[98:99]            )                  
            SLP.append             ( str(  int(string[99:104])/10   ) )                             #hPa
            SLP_QC.append          (   string[104:105]          ) 
            ADD.append             (   string[105::]            ) 
            
            # Take a look the precipitaiton data
            oneHR_RAIN_MM_tmp   = str(9999)
            threeHR_RAIN_MM_tmp = str(9999)
            sixHR_RAIN_MM_tmp   = str(9999)
            ppt_tag = ["AA1", "AA2", "AA3", "AA4"]
            for itag in ppt_tag:
                Index = string.find(itag)  
                if (Index != -1):   # not missing
                    try:
                        wt_hr = int(string[Index+3:Index+5])
                        if (wt_hr == 1):
                            oneHR_RAIN_MM_tmp   = string[Index+5:Index+9]
                        elif (wt_hr == 3):
                            threeHR_RAIN_MM_tmp = string[Index+5:Index+9]
                        elif (wt_hr == 6):
                            sixHR_RAIN_MM_tmp   = string[Index+5:Index+9]
                    except:
                        print (string[Index:Index+3] + " is not valid here")
            
            oneHR_RAIN_MM.append    (oneHR_RAIN_MM_tmp  )
            threeHR_RAIN_MM.append  (threeHR_RAIN_MM_tmp)
            sixHR_RAIN_MM.append    (sixHR_RAIN_MM_tmp  )
        
        df_new['TOTAL_CHAR']     = TOTAL_CHAR    
        df_new['USAF_ID']        = USAF_ID       
        df_new['WBAN_ID']        = WBAN_ID       
        df_new['YYYYMMDD']       = YYYYMMDD      
        df_new['HHMM']           = HHMM    
        df_new['YYYYMMDDHHMM']   = YYYYMMDDHHMM
        df_new['OBS_SOURCE']     = OBS_SOURCE    
        df_new['LAT']            = LAT           
        df_new['LON']            = LON           
        df_new['REPORT']         = REPORT        
        df_new['ELEV']           = ELEV          
        df_new['IDENTIFY']       = IDENTIFY      
        df_new['OBS_QC']         = OBS_QC        
        df_new['WIND_ANG']       = WIND_ANG      
        df_new['WIND_ANG_QC']    = WIND_ANG_QC   
        df_new['WIND_OBS_TY']    = WIND_OBS_TY   
        df_new['WIND_SPD']       = WIND_SPD      
        df_new['WIND_SPD_QC']    = WIND_SPD_QC   
        df_new['LOW_CLO']        = LOW_CLO       
        df_new['LOW_CLO_QC']     = LOW_CLO_QC    
        df_new['LOW_CLO_TY']     = LOW_CLO_TY    
        df_new['LOW_CLO_RP']     = LOW_CLO_RP    
        df_new['VISIBTY']        = VISIBTY       
        df_new['VISIBTY_QC']     = VISIBTY_QC    
        df_new['VISIBTY_VAR']    = VISIBTY_VAR   
        df_new['VISIBTY_VAR_QC'] = VISIBTY_VAR_QC
        df_new['TC']             = TC            
        df_new['TC_QC']          = TC_QC         
        df_new['DEWTC']          = DEWTC         
        df_new['DEWTC_QC']       = DEWTC_QC      
        df_new['SLP']            = SLP           
        df_new['SLP_QC']         = SLP_QC        
        df_new['ADD']            = ADD           
        df_new['oneHR_RAIN_MM']  = oneHR_RAIN_MM
        df_new['threeHR_RAIN_MM']= threeHR_RAIN_MM
        df_new['sixHR_RAIN_MM']  = sixHR_RAIN_MM
        
        sind = [i for i, j in enumerate(YYYYMMDD) if j == sYYYYMMDD]
        eind = [i for i, j in enumerate(YYYYMMDD) if j == eYYYYMMDD]
        
        # Check if the data contain the starting and ending dates by checking if sind and eind are empty lists
        if (len(sind) > 0):
            if(len(eind) > 0):
                sind = sind[0]
                eind = eind[-1]
                
                df_out = df_new.iloc[sind:eind+1]
                # processed output
                df_out.to_csv(opath+ofile, index = False)
            else:
                print(ofile + " is not created, do not contain " + sYYYYMMDD + " or " + eYYYYMMDD)
        else:
            print(ofile + " is not created, do not contain " + sYYYYMMDD + " or " + eYYYYMMDD)


