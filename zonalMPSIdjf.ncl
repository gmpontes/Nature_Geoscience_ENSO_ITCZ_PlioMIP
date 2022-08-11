begin

    sim="F_sst_sic_plio.0"

  do e=1,5

  fn="/aurora/wk3/CESM_gabriel/"+sim+e+"/atm/hist/"+sim+e+".cam.h0.V.PSL.djf.nc"

  data=addfile(fn,"r")
  psl=data->PSL
  v=data->V

  p=data->lev
  lat=data->lat

  nlat = dimsizes(v&lat)
  nlev = dimsizes(v&lev)

  psi=zonal_mpsi_Wrap(v(1:,:,:),lat,p(1:)*100,psl*100)

 ;;;;;create netCDF data;;;;;;;;;;;;;

   fout="/aurora/wk3/CESM_gabriel/"+sim+e+"/atm/hist/"+sim+e+".cam.h0.zmpsi.djf.nc"
   system("rm " + fout)      ; remove any pre-existing file
   fo = addfile(fout,"c")   ; open netCDF file

  ; create global attributes of the file
  ;===================================================================
  fAtt                = True
  fAtt@author         = "Gabriel"
  ;fAtt@creation_date = systemfunc ("date")
  fileattdef(fo, fAtt)

  dimNames = (/ "lev", "lat"/)
  dimSizes = (/ nlev-1, nlat /)
  dimUnlim = (/ False, False /)
  filedimdef(fo, dimNames, dimSizes,dimUnlim)

  filevardef(fo, "lev", "double", "lev")
  levatts = True
  levatts@standard_name = "air_pressure"
  levatts@long_name = "pressure"
  levatts@positive = "down"
  levatts@units = "Pa"
  levatts@axis = "y"
  filevarattdef(fo, "lev", levatts)  ; copy lev attributes
  fo->lev    = (/ v&lev(1:) /)

  filevardef(fo, "lat", "double", "lat")
  latatts = True
  latatts@standard_name = "lat"
  latatts@long_name = "latitude"
  latatts@units = "degrees_north"
  latatts@axis = "y"
  filevarattdef(fo,"lat", latatts)                     ; copy lon attributes
  fo->lat    = (/ v&lat /)

  filevardef(fo, "zmpsi", "double", (/ "lev", "lat"/))
  zmpsiatt = True
  zmpsiatt@standard_name = "meridional stream function"
  zmpsiatt@long_name = zmpsiatt@standard_name
  zmpsiatt@units = "kgs-1"
  filevarattdef(fo,"zmpsi", zmpsiatt)
  fo->zmpsi      = (/ psi /)
 
  delete(v) 
  delete(psl)
  end do
end
