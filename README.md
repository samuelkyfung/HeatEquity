# HeatEquity
Codes to produce the graphs 

./: Contains code for plotting graphs

data_processing_code/: Contains code for processing ISD and MODIS data for evaluation
  - download_ISD.py: download ISD data
  - 01.ISD_processing.py: for process ISD data
  - 02.WRFandISD_T2RH2_tocsv_Taylor.ncl: output of T2 and RH2 data from WRF simulation at the same location of the ISD stations as csv files
  - 03.WRFandISD_UV_tocsv_SailorR.ncl: output of U and V data from WRF simulation at the same location of the ISD stations as csv files
  - 00-03.ISD_all_dl_process_analysis.py: a script to run the download, 01., 02., and 03. codes.
  - 11.MODIS_regrid_to_WRFgrids_new_appeears.ncl: regrid MODIS data into WRF simulation grid for grid by grid comparison
