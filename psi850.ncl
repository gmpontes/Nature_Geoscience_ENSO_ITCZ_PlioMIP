begin
  model=new(14,string) 
  model=(/"CCSM4-1deg.","CCSM4-UofT.","CESM2.","COSMOS.","EC-Earth3.3.","GISS-E2-1-G.","HadCM3.","IPSL-CM6A_LR.","IPSLCM5A2.","IPSLCM5A.","MIROC4m.","MRI.","NorESM1-F.","NorESM-L."/)
  dim=new(2,string)
  sim=(/"E280","Eoi400"/)
  nmodel=dimsizes(model)
  nsim=dimsizes(sim)

  do s=0,nsim-1
  do m=0,nmodel-1
  ; specify output file
  fout = "psi850_2/"+model(m)+sim(s)+".psi850.clim.nc"
 
  ; open input file
  fin1 = addfile(model(m)+sim(s)+".uv850.clim.nc","r")
 
  v = fin1->v850

  nlon = dimsizes(v&lon)
  nlat = dimsizes(v&lat)
  ntime = dimsizes(v&time)

  u = fin1->u850
  
  sf = new ( dimsizes(u), typeof(u))
  vp = new ( dimsizes(u), typeof(u))
  uv2sfvpg(u,v,sf,vp)

  sf!0 = "time"
  sf&time = u&time
  sf!1 = "lat"
  sf&lat = u&lat
  sf!2 = "lon"
  sf&lon = u&lon
  
  ;;;;;create netCDF data;;;;;;;;;;;;;
  system("rm " + fout)      ; remove any pre-existing file
  fo = addfile(fout,"c")   ; open netCDF file

  ; create global attributes of the file
  ;===================================================================
  fAtt                = True
  fAtt@author         = "Johnny"
  ;fAtt@creation_date = systemfunc ("date")
  fileattdef(fo, fAtt)

  dimNames = (/ "time", "lat", "lon" /)
  dimSizes = (/ ntime, nlat, nlon /)
  dimUnlim = (/ True, False, False /)
  filedimdef(fo, dimNames, dimSizes, dimUnlim)

  filevardef(fo, "lon", "double", "lon")
  lonatts = True
  lonatts@standard_name = "longitude"
  lonatts@long_name = "longitude"
  lonatts@units = "degrees_east"
  lonatts@axis = "x"
  filevarattdef(fo,"lon", lonatts)                     ; copy lon attributes
  fo->lon    = (/ u&lon /)

  filevardef(fo, "lat", "double", "lat")
  latatts = True
  latatts@standard_name = "latitude"
  latatts@long_name = "latitude"
  latatts@units = "degrees_north"
  latatts@axis = "y"
  filevarattdef(fo,"lat", latatts)                     ; copy lon attributes
  fo->lat    = (/ u&lat /)

  filevardef(fo, "time", "double", "time")
  tatts = True
  tatts@calendar = "standard"
  tatts@standard_name = "time"
  tatts@long_name = "time"
  tatts@units = "days since 0000-01-01 00:00:00"
  filevarattdef(fo,"time", tatts)                   ; copy time attributes
  fo->time   = (/ u&time /)

  filevardef(fo, "sf", "double", (/"time", "lat", "lon"/))
  zopsiatt = True
  zopsiatt@standard_name = "stream function" 
  zopsiatt@units = "kgs-1"
  filevarattdef(fo,"sf", zopsiatt)
  fo->sf      = (/ sf /)
  
  delete(u)
  delete(v)
  delete(sf)
  delete(vp)
  end do
  end do

end

