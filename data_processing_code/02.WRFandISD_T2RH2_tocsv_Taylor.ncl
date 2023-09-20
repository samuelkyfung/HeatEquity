;-------------------------------------------------------
; WRFandISD_T2RH2_tocsv_Taylor.ncl
;
; Kwun Yip Fung
; 26 Aug 2020
; Export the T on the grid point to csv file
;--------------------------------------------------------
print("Plotting "+ "WRFandISD_T2RH2_tocsv_Taylor.ncl")
cases   = getenv("CASES_DIR")
expt_pre = getenv("EXPT_PRE")
legend   = getenv("EXPT_PRE_LEGEND")
legend1  := legend 
write_obs_sta = getenv("WRITE_OBS") ; Switch to write observation csv 

print("CASES_DIR:       " + cases)
print("EXPT_PRE:        " + expt_pre)
print("EXPT_PRE_LEGEND: " + legend)


UrbNUrb          = (/"Urb", "NUrb"/)
do icase = 0 , dimsizes(cases) -1
do iurb = 0 , dimsizes(UrbNUrb)-1
print(UrbNUrb(iurb) + " stations")
wrf_ipath        := "/scratch/05898/kf22523/data/WRF_out/Houston/"+cases(icase)+"/"
opath             = "/home1/05898/kf22523/graphs/txt/WRF_out/Houston/"+cases(icase)+"/"+UrbNUrb(iurb)+"/"
noaa_hrly_path    = "/scratch/05898/kf22523/data/ISD/"+cases(icase)+"/"+UrbNUrb(iurb)+"/"

csv_file            := systemfunc("ls " + noaa_hrly_path +"*-[10000000-30000000]*.csv")  ;file end with -[dates].csv  (e.g. -20170913.csv) will be listed
station_name_plt    := str_split_by_length(str_sub_str(csv_file, noaa_hrly_path, "ISD_"), 10) 
station_name_plt    := station_name_plt(:,0)

buffer  = 0

;file_pre          := wrf_ipath + expt_pre1
file_pre1         := wrf_ipath + expt_pre 
UTC               = -5 ; Summer time
UTC_real          = -5 ; Summer time

nexpts1 := dimsizes(file_pre1)
nexpts1 := nexpts1(0)

files_tmp   := systemfunc("ls "+ file_pre1(0) + "*_00_00")
tmp_f       := addfile(files_tmp, "r")
lat2d       := tmp_f->XLAT
lon2d       := tmp_f->XLONG
lat2d       := lat2d(0,:,:)      
lon2d       := lon2d(0,:,:)      
lat         := lat2d(:,0)
lon         := lon2d(0,:)
time        := wrf_user_getvar(tmp_f, "XTIME", -1)
new_time_units = str_sub_str(time@units, "minutes", "hours")
time_new    = time_to_newtime(time, new_time_units)
time_LT      := time_new + UTC_real
copy_VarMeta (time_new, time_LT )
if (time_LT(0) .lt. 0) then
    print("update the starting day for time unit one day earlier!")
    print("time_to_newtime(time, new_time_units)")
    tmp := cd_calendar(time_LT ,-2)
    tmp := tostring(tmp(0))
    new_time_units = "hours since " + str_join(str_split_by_length(tmp, (/4,2,2/)), "-") + " 00:00:00"
    time_new    = time_to_newtime(time, new_time_units)
    time_LT      := time_new + UTC_real  
    copy_VarMeta (time_new, time_LT )
end if

yyyymmddhh  := cd_calendar(time_LT ,-3)              ; yyyymmddhh
yyyymmdd    := cd_calendar(time_LT ,-2)
yyyymm      := cd_calendar(time_LT ,-1)
yyyy        := doubletointeger(yyyymm/100)
mmdd        := doubletointeger(yyyymmdd - yyyy*10000)
mm          := doubletointeger(mmdd/100)
dd          := doubletointeger(mmdd-mm*100)
hh          := doubletointeger(yyyymmddhh - yyyymmdd*100)
    
;tk_f0   := wrf_user_getvar(nc_plt(0), "tk",    -1)

;dims    = dimsizes(tk_f0)
nlat    = dimsizes(lat)
nlon    = dimsizes(lon)
ntime   = dimsizes(time)
;nlev    = dims(1)


c1 = -42.379
c2 = 2.04901523
c3 = 10.14333127
c4 = -0.22475541 
c5 = -6.83783 * 10^-3
c6 = -5.481717 * 10^-2
c7 = 1.22874 * 10^-3
c8 = 8.5282 * 10^-4
c9 = -1.99 * 10^-6


nstations    = dimsizes(csv_file)
;field_req    := (/"DATE", "HourlyDryBulbTemperature", "HourlyPrecipitation"/)
field_req    :=(/"YYYYMMDDHHMM", "TC", "DEWTC"/) ; Dont remove the first two elements
station_lat  := new(nstations, float)
station_lon  := new(nstations, float)
nfields      = dimsizes(field_req)
vars_stations := new((/nstations, nfields, ntime/), string)
print("First read obs")

; Extract index that can have 
first_ind = 0 
do istation  = 0, nstations-1
    print(istation + "/" + (nstations-1))
    lines  := asciiread(csv_file(istation),-1,"string")
    nlines := dimsizes(lines)-1   ; First line is a header

    ;---This file has quotes around everything, which we don't want.
    dq    := str_get_dq()
    lines := str_sub_str(lines,dq,"")   ; Replace quotes with empty string
    lines := str_sub_str(lines,",,",",NaN,") ; To make sure that the empty fields between the consecutive delimiters get treated as real values, use str_sub_str to insert a number between these delimiters. We use NaN
    lines := str_sub_str(lines,",,",",NaN,")

    ;---First line is name of each field
    delim        := ","
    field_names  := str_split(lines(0),delim)

    ;---We're going to read the whole file as a 2D array of strings
    fields := new((/nfields,nlines/),string)
    do nf=0,nfields-1
        index        := ind(field_names .eq. field_req(nf))
        fields(nf,:) = str_get_field(lines(1:),index+1,delim)
    end do

    index                   := ind(field_names .eq. "LAT")
    tmp                     := str_get_field(lines(1:),index+1,delim)
    station_lat(istation)    = tofloat(tmp(0))
    index                   := ind(field_names .eq. "LON")
    tmp                     := str_get_field(lines(1:),index+1,delim)
    station_lon(istation)    = tofloat(tmp(0))
    
    ; Remove missing
    index1 := ind(tofloat(fields(1,:)) .lt. 100)
    if .not. (any(ismissing (index1))) then
        if (first_ind .eq. 0) then
            csv_ind := istation
            first_ind = 1
        else 
            csv_ind := array_append_record(csv_ind, istation, 0)
        end if
    else
        print("Remove file wil all missing T2m data: " + csv_file(istation))
    end if
end do 
csv_file := csv_file(csv_ind)
station_name_plt    := str_split_by_length(str_sub_str(csv_file, noaa_hrly_path, "ISD_"), 10) 
station_name_plt    := station_name_plt(:,0)
station_lon         := station_lon(csv_ind)
station_lat         := station_lat(csv_ind)
lljjres = True
lljjres@returnInt = True ; True will return integer, False will return interpolated index
istation_ij := wrf_user_ll_to_ij(tmp_f, station_lon, station_lat, lljjres)
okind       := ind(ind(istation_ij(1,:) .lt. nlat) .and. ind(istation_ij(0,:) .lt. nlon))
nookind     := ind(ind(istation_ij(1,:) .ge. nlat) .and. ind(istation_ij(0,:) .ge. nlon))
if .not. (any(ismissing(nookind))) then
print("Remove file with stations beyond the simulation area: " + csv_file(nookind))
end if
csv_file    := csv_file(okind)
station_name_plt    := str_split_by_length(str_sub_str(csv_file, noaa_hrly_path, "ISD_"), 10) 
station_name_plt    := station_name_plt(:,0)
;----------------ISD preprocessing end----------------

print("Real read obs")
nstations    = dimsizes(csv_file)
;field_req    := (/"DATE", "HourlyDryBulbTemperature", "HourlyPrecipitation"/)
field_req    :=(/"YYYYMMDDHHMM", "TC", "DEWTC"/) ; Dont remove the first two elements
station_lat  := new(nstations, float)
station_lon  := new(nstations, float)
nfields      = dimsizes(field_req)
vars_stations := new((/nstations, nfields, ntime/), string)
do istation  = 0, nstations-1
    print(istation + "/" + (nstations-1))
    lines  := asciiread(csv_file(istation),-1,"string")
    nlines := dimsizes(lines)-1   ; First line is a header

    ;---This file has quotes around everything, which we don't want.
    dq    := str_get_dq()
    lines := str_sub_str(lines,dq,"")   ; Replace quotes with empty string
    lines := str_sub_str(lines,",,",",NaN,") ; To make sure that the empty fields between the consecutive delimiters get treated as real values, use str_sub_str to insert a number between these delimiters. We use NaN
    lines := str_sub_str(lines,",,",",NaN,")
    count := str_fields_count(lines,",")
    ;print(count)

    ;---First line is name of each field
    delim        := ","
    field_names  := str_split(lines(0),delim)

    ;---We're going to read the whole file as a 2D array of strings
    fields := new((/nfields,nlines/),string)
    do nf=0,nfields-1
        index        := ind(field_names .eq. field_req(nf))
        fields(nf,:) = str_get_field(lines(1:),index+1,delim)
    end do
    
    ; READ LAT LON
    index                   := ind(field_names .eq. "LAT")
    tmp                     := str_get_field(lines(1:),index+1,delim)
    station_lat(istation)    = tofloat(tmp(0))
    index                   := ind(field_names .eq. "LON")
    tmp                     := str_get_field(lines(1:),index+1,delim)
    station_lon(istation)    = tofloat(tmp(0))
    
    
    ; Remove missing
    index1 := ind(tofloat(fields(1,:)) .lt. 100)
    fields := fields(:, index1)
    
    units                = time_LT@units
    time_station_string := fields(0,:)
    t_split := str_split_by_length(time_station_string, (/4,2,2,2,2/))
    t_split := toint(t_split(:,:))
    zeros   := new(dimsizes(t_split(:,0)), integer)
    zeros(:) = 0
    
    time_station  := cd_inv_calendar(t_split(:,0), t_split(:,1), t_split(:,2), t_split(:,3), t_split(:,4), zeros, units, 0)
    time_station  = time_station+UTC
    index2        := ind(time_station .ge. 0)
    fields        := fields(:, index2)
    time_station  := time_station(index2)
    ;print(get1Dindex(time_station, time_2))
    time_ind   := new(ntime, integer)
    do it = 0, ntime-1
        time_ind(it) = closest_val(time_LT (it), time_station)
    end do 
    
    vars_stations(istation, :, :) = fields(:, time_ind)
    t2mC_sta    = tofloat(vars_stations(istation,1,:))
    t2mF_sta    = (t2mC_sta * 9/5) + 32
    td2mC_sta   = tofloat(vars_stations(istation,2,:))
    rh2_sta     = tofloat(100*(exp((17.625*td2mC_sta)/(243.04+td2mC_sta))/exp((17.625*t2mC_sta)/(243.04+t2mC_sta))))
    missingInd := ind(rh2_sta .gt. 100)
    if (any (.not. ismissing(ind(rh2_sta .gt. 100)))) then
        rh2_sta(ind(rh2_sta .gt. 100)) = -1.0
    end if
    HI_sta      = c1 + c2*t2mF_sta + c3*t2mF_sta + c4*t2mF_sta*rh2_sta + c5*t2mF_sta^2 + c6*rh2_sta^2 + c7*t2mF_sta^2*rh2_sta + c8*t2mF_sta*rh2_sta^2 + c9*t2mF_sta^2*rh2_sta^2
    vars_stations(istation,1,:)   = t2mC_sta + 273.15
    vars_stations(istation,2,:)   = rh2_sta
    
end do 

lljjres = True
lljjres@returnInt = True ; True will return integer, False will return interpolated index
istation_ij  := wrf_user_ll_to_ij(tmp_f, station_lon, station_lat, lljjres)


;----- Write Observation CSV -----;
obs_sta_name   := str_sub_str(csv_file, noaa_hrly_path, "")
obs_sta_name   := str_split_by_length(obs_sta_name, 12)
obs_sta_name   := obs_sta_name(:,0)
obs_ofile_name := obs_sta_name + "-ISD-T2RH2-WRFtime.csv"

t2k_obs := decimalPlaces(tofloat(vars_stations(:,1,:)), 3, True)
rh2_obs := decimalPlaces(tofloat(vars_stations(:,2,:)), 3, True)
t_obs   := vars_stations(:,0,:)

if (write_obs_sta .eq. "True") then
do iofile =  0 , nstations-1
    t_obs_wri       = t_obs  (iofile,:)
    t2k_obs_wri     = t2k_obs(iofile,:)
    rh2_obs_wri     = rh2_obs(iofile,:)
    oname_wri   = obs_ofile_name(iofile)
    print("Writing: " + oname_wri)
    alist   = [/t_obs_wri, t2k_obs_wri, rh2_obs_wri/]

    header  = (/"time, T2, RH2"/)
    hlist   = [/header/]

    fname   = opath + oname_wri
    write_table(fname, "w", hlist, "%s")
    write_table(fname, "a", alist, "%s,%6.3f,%6.3f")
end do 
end if





;----- Write WRF CSV -----;
;t2k_wrf_all     := new((/nexpts, ntime, nlat, nlon/), float)
t2k_wrf_tmp     := new((/nexpts1,ntime, nlat, nlon/), float)
;rh2_wrf_all     := new((/nexpts, ntime, nlat, nlon/), float)
rh2_wrf_tmp     := new((/nexpts1,ntime, nlat, nlon/), float)


print("read WRF")
do i = 0, nexpts1-1
    print(i + "/" + (nexpts1-1))
    filename            := systemfunc("ls "+ file_pre1(i) + "*_00_00")
    tmp                 := addfile(filename, "r")
    t2k_tmp             := tmp->T2
    rh2_tmp             := wrf_user_getvar(tmp,"rh2",-1) 
    t2k_wrf_tmp(i,:,:,:)   = t2k_tmp
    rh2_wrf_tmp(i,:,:,:)   = rh2_tmp
end do 

;t2k_wrf_all     := array_append_record(t2k_wrf_all, t2k_wrf_tmp, 0)
;rh2_wrf_all     := array_append_record(rh2_wrf_all, rh2_wrf_tmp, 0)
t2k_wrf_all     := t2k_wrf_tmp
rh2_wrf_all     := rh2_wrf_tmp

;--- extract station location data ---;
;t2k_wrf_gridavg := new((/nexpts+nexpts1, nstations, ntime/), float)
;rh2_wrf_gridavg := new((/nexpts+nexpts1, nstations, ntime/), float)
t2k_wrf_gridavg := new((/nexpts1, nstations, ntime/), float)
rh2_wrf_gridavg := new((/nexpts1, nstations, ntime/), float)

do ista = 0, nstations-1
    ii = istation_ij(1,ista) 
    jj = istation_ij(0,ista)
    t2k_wrf_gridavg(:,ista,:) = dim_avg_n_Wrap(dim_avg_n_Wrap (t2k_wrf_all(:,:,ii-buffer:ii+buffer, jj-buffer:jj+buffer),2),2)
    rh2_wrf_gridavg(:,ista,:) = dim_avg_n_Wrap(dim_avg_n_Wrap (rh2_wrf_all(:,:,ii-buffer:ii+buffer, jj-buffer:jj+buffer),2),2)
end do 

;----- Write WRF CSV -----;
t_wrf := tostring(cd_calendar(time_new, -3)) + "00"

;do ij = 0, nexpts+nexpts1-1
do ij = 0, nexpts1-1
    do ista = 0, nstations-1
        t_wrf_wri       = t_wrf(:)
        t2k_wrf_wri     = t2k_wrf_gridavg(ij,ista,:)
        rh2_wrf_wri     = rh2_wrf_gridavg(ij,ista,:)
        oname_wri       = legend1(ij) +"-" + obs_sta_name(ista) + "-WRF-T2RH2-WRFtime.csv"
        print("Writing: " + oname_wri)

        alist   = [/t_wrf_wri, t2k_wrf_wri, rh2_wrf_wri/]

        header  = (/"time, T2, RH2"/)
        hlist   = [/header/]

        fname   = opath + oname_wri
        write_table(fname, "w", hlist, "%s")
        write_table(fname, "a", alist, "%s,%6.3f,%6.3f")


end do 
end do 
end do 
end do




















