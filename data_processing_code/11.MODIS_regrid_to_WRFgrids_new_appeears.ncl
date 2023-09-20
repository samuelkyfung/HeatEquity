;-------------------------------------------------------
; MODIS_regrid_to_WRFgrids.ncl 
;
; Kwun Yip Fung
; 31 Aug 2022
; 
; ! Directly download the MODIS LST MOD21A1D, MOD21A1N, MYD21A1D, MYD21A1N
;   and store in 'MOD21A1DN_MYD21A1DN/' directory
; ! from https://appeears.earthdatacloud.nasa.gov/task/area
;--------------------------------------------------------
; Library load
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRF_contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"

;ipath_pre   = "/scratch/05898/kf22523/data/"
ipath_pre   = "/work/05898/kf22523/stampede2/data_for_analysis/Houston/"
ipaht       = "/scratch/05898/kf22523/data/WRF_out/Houston/Manuscript/"

; Select one set of the case each time
case                = "20170728-20170801_HW1"
aqua_YYYY           = 2017
aqua_DOY_start      = day_of_year(2017, 07, 28)

case                = "20180722-20180726_HW2"
aqua_YYYY           = 2018
aqua_DOY_start      = day_of_year(2018, 07, 22)

case                = "20180820-20180824_HW3"
aqua_YYYY           = 2018
aqua_DOY_start      = day_of_year(2018, 08, 20)

case                = "20190813-20190817_HW4"
aqua_YYYY           = 2019
aqua_DOY_start      = day_of_year(2019, 08, 13)

case                = "20190904-20190908_HW5"
aqua_YYYY           = 2019
aqua_DOY_start      = day_of_year(2019, 09, 04)
; Select one set of the case each time


aqua_DOY_start      = aqua_DOY_start-1 ; Since 01/01 is count as day 1, but we want day 0


aqua_path           = ipath_pre + "MOD21A1DN_MYD21A1DN/"+case+"/"
wrf_path            = ipath+case+"/LCZ_GLO/"
;plt_pre             = "WRFvs"+MOD_DATA+"_LST." + EXPTS1(iexpt1) + "." + EXPTS2(iexpt2)
regird_path         = "/scratch/05898/kf22523/data/tmp/"
;LU2018LCZ_path      = ipath_pre + "WRF_out/Houston/20180321-20180326_NoRain/MODIS_2018_Houston_3.9.1/"
;LU2018LCZ_path      = "/work/05898/kf22523/stampede2/WRF/WRFv3.9.1.1/geo_em_files/Houston/MODIS_2018_Houston_3.9.1/"

aqua_file   := systemfunc("ls "+ aqua_path   + "*aid0001.nc")
wrf_f0_file := systemfunc("ls "+ wrf_path    + "E5_gr0tc0ar0ag0_d03_" + "*_00_00_00")

nfiles      := dimsizes(aqua_file)

LOC         = -5
LOC_real    = -5

;-------------------------------
;           WRF file
;-------------------------------
tmp_wrf_df  = addfile (wrf_f0_file(0), "r")
wrf_lat2d   := wrf_user_getvar(tmp_wrf_df, "lat", 0)
wrf_lon2d   := wrf_user_getvar(tmp_wrf_df, "lon", 0)
wrf_minlat = min(wrf_lat2d)
wrf_maxlat = max(wrf_lat2d)
wrf_minlon = min(wrf_lon2d)
wrf_maxlon = max(wrf_lon2d)
dims       = dimsizes(wrf_lat2d)
nlat       = dims(0)
nlon       = dims(1)
wrf_t2m    = tmp_wrf_df->TSK (0,:,:)  

wrf_f0      = addfiles(wrf_f0_file, "r")
time        := wrf_user_getvar(wrf_f0, "XTIME", -1)
new_time_units = str_sub_str(time@units, "minutes", "hours")
time_new    := time_to_newtime(time, new_time_units)
;time_0      := wrf_times_c( tochar(time), 0 )      ; "hours since" initial time on file  (double)
;time_1      := wrf_times_c( tochar(time), 1 )      ; "hours since 1901-01-01 00:00:00"   (double)
time_LOC     := time_new + LOC_real
copy_VarMeta (time_new, time_LOC)
yyyymmddhh_LOC  := cd_calendar(time_LOC,-3)              ; yyyymmddhh_LOC
yyyymmdd_LOC    := cd_calendar(time_LOC,-2)
yyyymm_LOC      := cd_calendar(time_LOC,-1)
yyyy_LOC        := doubletointeger(yyyymm_LOC/100)
mmdd_LOC        := doubletointeger(yyyymmdd_LOC - yyyy_LOC*10000)
mm_LOC          := doubletointeger(mmdd_LOC/100)
dd_LOC          := doubletointeger(mmdd_LOC-mm_LOC*100)
hh_LOC          := doubletointeger(yyyymmddhh_LOC - yyyymmdd_LOC*100)

delete([/wrf_f0, tmp_wrf_df/])

;-------------------------------
;       Aqua + Terra file
;-------------------------------
aquaeostmp_df   = addfile (aqua_file (0), "r")
mod_lat1d       := aquaeostmp_df->lat
mod_lon1d       := aquaeostmp_df->lon
mod_lldims      = (/dimsizes(mod_lat1d), dimsizes(mod_lon1d)/)

; ---- Locate the area required for wrf ---- ;
; Find the index of lat lon which is closest to the lat lon in wrf
; Since in lat2d, every column gives the same lats, so we can just one of it
imax = closest_val(wrf_minlat, mod_lat1d) 
imin = closest_val(wrf_maxlat, mod_lat1d)
jmax = closest_val(wrf_maxlon, mod_lon1d) 
jmin = closest_val(wrf_minlon, mod_lon1d)

print("imin:" + imin + " imax:" + imax + " jmin:" + jmin + " jmax:" + jmax )
  
; Update new mod_lat2d, mod_lon2d
nx = imax-imin+1
ny = jmax-jmin+1
mod_lat2d := new((/nx, ny/), double)
mod_lon2d := new((/nx, ny/), double)

do j = 0, ny-1
mod_lat2d(:, j) = mod_lat1d(imin:imax)
end do 
do i = 0, nx-1
mod_lon2d(i, :) = mod_lon1d(jmin:jmax)
end do 

delete([/aquaeostmp_df/])

do ifile = 0, nfiles -1 ; Skip the first day for spin up
    ;-------------------------------
    ; Extract LST in MODIS and WRF
    ;-------------------------------
    ; ----  Aqua + Terra file ---- ; 
    aqua_file_tmp   = aqua_file  (ifile)
    odir           := str_sub_str(aqua_file_tmp, aqua_path, "")
    odir           := str_split  (odir, ".")
    odir           := odir(0)
    
    opath   = "/work/05898/kf22523/stampede2/data_for_analysis/Houston/" +odir+"/"+case+"/"
    aquatmp_df  = addfile (aqua_file_tmp, "r")
    aqua_LST_all    := aquatmp_df->LST_1KM
    aqua_time_all   := short2flt(aquatmp_df->View_Time)
    aqua_QC_all     := aquatmp_df->QC
    nt              := dimsizes(aqua_LST_all)
    nt              := nt(0)
    
    do it = 0, nt-1
    print("it: " + it)
    aqua_DOY  := aqua_DOY_start+it
    aqua_LST  := aqua_LST_all (it, :, :)
    aqua_time := aqua_time_all(it, :, :)
    aqua_QC   := aqua_QC_all  (it, :, :)
    ; QC variable process
    aqua_QC_bit         := getbitsone(aqua_QC)
    aqua_QC_bit!0       = "x"
    aqua_QC_bit!1       = "y"
    aqua_QC_bit!2       = "bit"
    aqua_QC_bit2        := aqua_QC_bit(bit|:, x|:, y|:)
    QCdims              := dimsizes(aqua_QC_bit)
    nbit                = QCdims(2)
    aqua_QC_bit2_2d     := reshape(aqua_QC_bit2, (/nbit, QCdims(0)*QCdims(1)/))
    aqua_QC_bit2_2d!0   = "bit"
    aqua_QC_bit2_2d!1   = "xy"
    aqua_QC_bit2_2d2    := aqua_QC_bit2_2d(xy|:, bit|:)
    ; bit 0 is the least significant bit (i.e. the last value)
    GoodInd             := ind(aqua_QC_bit2_2d2 (:, nbit-2) .eq. 0 )
    aqua_QC_update_1d   := new((QCdims(0)*QCdims(1)), integer)
    aqua_QC_update_1d(:) = 0
    aqua_QC_update_1d(GoodInd)   = 1
    aqua_QC_update      := reshape(aqua_QC_update_1d, (/QCdims(0), QCdims(1)/))
    copy_VarMeta(aqua_QC, aqua_QC_update)
    aqua_QC_update      := aqua_QC_update(imin:imax, jmin:jmax)
    aqua_QC_update@lat2d = mod_lat2d
    aqua_QC_update@lon2d = mod_lon2d
    ; QC variable process end

    aqua_LST   := aqua_LST  (imin:imax, jmin:jmax)
    aqua_time  := aqua_time (imin:imax, jmin:jmax)

    aqua_LST@lat2d = mod_lat2d
    aqua_LST@lon2d = mod_lon2d
    
    aqua_fname     = str_sub_str(aqua_file_tmp, aqua_path, "")
    aqua_DOY@units = "days since "+tostring(aqua_YYYY)+"-01-01 00:00:00"
    aqua_time_tmp  := time_to_newtime(aqua_DOY, new_time_units)
    aqua_time      := aqua_time + aqua_time_tmp
    aqua_time@units = new_time_units
    
    ; ----  WRF file ---- ;   
    wrf_aqua_frame   = closest_val(avg(aqua_time), time_LOC)
    ;wrf_aqua_f0_df   = addfile (wrf_f0_file(wrf_aqua_frame) , "r")
    ;wrf_aqua_f0_t2m  = wrf_aqua_f0_df->TSK (0,:,:)  
    wrf_aqua_f0_df   = addfile (wrf_f0_file , "r")
    wrf_aqua_f0_t2m  = wrf_aqua_f0_df->TSK (wrf_aqua_frame,:,:)  
    
    ;-------------------------------
    ; REGIRD MODIS into WRF
    ;-------------------------------
    print("Regriding")
    interp_method      = "bilinear" ; bilinear, neareststod, patch, conserve

    Opt                = True
    Opt@ForceOverwrite = True
    Opt@SrcRegional    = True      ; These two are important! Do not
    Opt@DstRegional    = True      ; set if you have a global lat/lon grid
    Opt@SrcGridLat     = mod_lat2d
    Opt@SrcGridLon     = mod_lon2d
    Opt@DstGridLat     = wrf_lat2d
    Opt@DstGridLon     = wrf_lon2d
    Opt@InterpMethod   = interp_method

    Opt@Debug          = True
    Opt@PrintTimings   = True

    Opt@SrcFileName    = regird_path + "s."+yyyymmdd_LOC(wrf_aqua_frame)+".MODIS_Grid.nc"
    Opt@DstFileName    = regird_path + "s."+yyyymmdd_LOC(wrf_aqua_frame)+".MODIS_WRF_Grid.nc"
    Opt@WgtFileName    = regird_path + "s."+yyyymmdd_LOC(wrf_aqua_frame)+".weight" + interp_method + ".nc"
    
    ;Opt@SrcGridMask    = where(.not.ismissing(aqua_LST),1,0)
    Opt@SrcGridMask     = aqua_QC_update
    ; if (fileexists(Opt@SrcFileName) .and. fileexists(Opt@DstFileName) .and. fileexists(Opt@WgtFileName))then 
    ; ; Uncomment any one of these if the grid file and/or weights file is alreaay generated.
    ; Opt@SkipSrcGrid     = True
    ; Opt@SkipDstGrid     = True
    ; Opt@SkipWgtGen      = True
    ; else
    Opt@SkipSrcGrid     = False
    Opt@SkipDstGrid     = False
    Opt@SkipWgtGen      = False
    ; end if
 
    aqua_LST_regridWRF  := ESMF_regrid(aqua_LST,Opt)   ; Regrid 
    if (all(ismissing(aqua_LST_regridWRF))) then
        aqua_LST_regridWRF = new((/nlat, nlon/), float)
        aqua_LST_regridWRF(:,:) = 0.0
    end if
    aqua_time_regridWRF := new((/nlat, nlon/), float)
    aqua_time_regridWRF(:,:) = tofloat(avg(aqua_time))
    copy_VarMeta  (wrf_t2m,   aqua_LST_regridWRF)
    copy_VarAtts  (aqua_time, aqua_time_regridWRF)
    copy_VarCoords(wrf_t2m,   aqua_time_regridWRF)
    
    print("==================================================")

    ;NCL cannot contour data that has a missing value (_FillValue attribute) equal to 0.0.
    aqua_LST_regridWRF@_FillValue  = -100000.0

    ;===================================================================
    ; Write the regrided file
    ;===================================================================
    aqua_oname     = str_sub_str(str_sub_str(aqua_file_tmp, ".nc", "")+"."+aqua_DOY+".WRFd04grid.nc", aqua_path, opath)
    ;aqua_oname     = str_sub_str(str_sub_str(aqua_file_tmp, ".nc", ".WRFd04grid.nc"), aqua_path, opath)
    system("/bin/rm -f " + aqua_oname) 
    fout  = addfile (aqua_oname, "c")  
    print("Writing: " + aqua_oname)
    
    ;===================================================================
    ; create global attributes of the file
    ;===================================================================
    fAtt               = True            ; assign file attributes
    fAtt@title         = "NCL processed MODIS LST regrided into WRF grid"  
    fAtt@source_file   = aqua_fname
    fAtt@Conventions   = "None"   
    fAtt@creation_date = systemfunc ("date")        
    fileattdef( fout, fAtt )            ; copy file attributes   
    
    ;===================================================================
    ; predefine the coordinate variables and their dimensionality
    ; Note: to get an UNLIMITED record dimension, we set the dimensionality
    ; to -1 (or the actual size) and set the dimension name to True.
    ;===================================================================
    dimNames = (/"south_north", "west_east"/)  
    dimSizes = dims
    dimUnlim = (/False, False/)   
    filedimdef(fout,dimNames,dimSizes,dimUnlim)
    
    ;===================================================================
    ; predefine the the dimensionality of the variables to be written out
    ;===================================================================
    ; Here we are using NCL functions to facilitate defining 
    ; each variable's dimension name(s) and type. 
    ; The following could be replaced with explicit, user defined dimension 
    ; names different from those associated with the variable in memory. 
    ; Say, PS(time,lat,lon) in the NCL script. They could be redefined for the file via: 
    ; filevardef(fout, "PS"   ,typeof(PS) ,(/"TIME","latitude","longitude"/)) 
    ;===================================================================
    filevardef(fout,   "T2_regird_WRF",   typeof(aqua_LST_regridWRF),  getvardims(aqua_LST_regridWRF ))
    filevardef(fout,   "time_regrid_WRF", typeof(aqua_time_regridWRF), getvardims(aqua_time_regridWRF))
    filevarattdef(fout,"T2_regird_WRF",   aqua_LST_regridWRF)                   
    filevarattdef(fout,"time_regrid_WRF", aqua_time_regridWRF)                   
    
    ;===================================================================
    ; output only the data values since the dimensionality and such have
    ; been predefined. The "(/", "/)" syntax tells NCL to only output the
    ; data values to the predefined locations on the file.
    ;====================================================================
    fout->T2_regird_WRF        = (/aqua_LST_regridWRF /)     
    fout->time_regrid_WRF      = (/aqua_time_regridWRF/)
    end do 
    
end do 
