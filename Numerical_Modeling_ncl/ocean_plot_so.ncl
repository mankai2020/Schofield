begin

	sfile=systemfunc("ls /work1/mankai/ORAS/so*");
	f=addfiles(sfile,"r");
	
	so=f[:]->so(50,0:36,:,327:339);
	;so=f[:]->so(0,:,:,:);
	so_zon=dim_avg(so);
	so_zon!0="depth";
	so_zon!1="lat";
	so_zon&depth=so&depth(0:36);
	so_zon&lat=so&lat;
	print(so&lon)
	printVarSummary(so);
	printVarSummary(so_zon);
	
	so1=f[1]->so(7,0:36,:,327:339);
	printVarSummary(so1);
	
	;===========================
	; plot parameters
	;===========================
	
	 wks_type = "png"
	 ;wks_type@wkWidth = 800
	 ;wks_type@wkHeight = 500

	 wks   = gsn_open_wks (wks_type, "so_zon" )         ; send graphics to PNG file

	 res               = True            ; Plot mods desired
	 
	 res@vpWidthF = 3
	 res@vpHeightF = 1.8
	 
	 res@gsnMaximize   = True            ; Maximize plot in frame
	 res@cnFillOn      = True            ; Turn on contour fill
	 res@cnFillPalette = "BlAqGrYeOrRe"
	 res@tiMainString  = "Salinity_Aug"   ; Main title
	 
	 ;res@gsnYAxisIrregular2Linear = True ; converts irreg depth to linear
	 res@gsnYAxisIrregular2Log = True    ; Convert Y axis to logarithmic
	 res@trYReverse    		      = True ; reverse y axis
	 
	 res@tmYLMode              = "Explicit"
	 res@tmYLValues            = (/10,100,1000,3000/)
	 res@tmYLLabels            = "" + res@tmYLValues

	 plot = gsn_csm_contour(wks, so_zon, res ) 
	 
	 ;res@tiMainString          = "Logarithmic axis"
	 ;res@gsnYAxisIrregular2Log = True    ; Convert Y axis to logarithmic
	 ;res@gsnXAxisIrregular2Log = True    ; Convert X axis to logarithmic

	;---Set some axis labels, otherwise all we'll get is "10^3".
	 ;res@tmYLMode              = "Explicit"
	 ;res@tmYLValues            = (/10,20,30,40/)
	 ;res@tmYLLabels            = "" + res@tmYLValues

	 ;plot = gsn_csm_contour(wks, so_zon, res ) 

end