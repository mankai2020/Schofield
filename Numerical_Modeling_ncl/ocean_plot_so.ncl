begin

	sfile=systemfunc("ls /work1/mankai/ORAS/so*");
	f=addfiles(sfile,"r");
	ListSetType(f, "join");
	
	so=f[:]->so(:,7,0:36,:,327:339);
	so_zon=dim_avg(dim_avg_n(so,0));
	so_zon!0="depth";
	so_zon!1="lat";
	so_zon&depth=so&depth(0:36);
	so_zon&lat=so&lat;
	; printVarSummary(so);
	; printVarSummary(so_zon);
	
	;===========================
	; plot parameters
	;===========================
	
	 wks_type = "png"
	 wks   = gsn_open_wks (wks_type, "so_zon_Aug" )         ; send graphics to PNG file

	 res               = True            ; Plot mods desired
	 res@vpWidthF = 3
	 res@vpHeightF = 1.8
	 
	 res@gsnMaximize   = True            ; Maximize plot in frame
	 res@cnFillOn      = True            ; Turn on contour fill
	 res@cnFillPalette = "BlAqGrYeOrRe"
	 res@tiMainString  = "Salinity_Aug"   ; Main title
	 res@gsnRightString   = "psu" 
	 
	 res@gsnYAxisIrregular2Linear = True ; converts irreg depth to linear
	 ;res@gsnYAxisIrregular2Log = True    ; Convert Y axis to logarithmic
	 res@trYReverse    		      = True ; reverse y axis
	 
	 res@tmYLMode              = "Explicit"
	 res@tmYLValues            = (/10,1000,2000,3000/)
	 res@tmYLLabels            = "" + res@tmYLValues + "m"

	 plot = gsn_csm_contour(wks, so_zon, res ) 

end