begin

	sfile=systemfunc("ls /work1/mankai/ORAS/thetao*");
	f=addfiles(sfile,"r");
	ListSetType(f, "join");
	
	thetao=f[:]->thetao(:,7,0:36,:,327:339);
	thetao_zon=dim_avg(dim_avg_n(thetao,0));
	thetao_zon!0="depth";
	thetao_zon!1="lat";
	thetao_zon&depth=thetao&depth(0:36);
	thetao_zon&lat=thetao&lat;
	; printVarSummary(thetao);
	; printVarSummary(thetao_zon);
	
	;===========================
	; plot parameters
	;===========================
	
	 wks_type = "png"
	 wks   = gsn_open_wks (wks_type, "thetao_zon_Aug" )         ; send graphics to PNG file

	 res               = True            ; Plot mods desired
	 res@vpWidthF = 3
	 res@vpHeightF = 1.8
	 
	 res@gsnMaximize   = True            ; Maximize plot in frame
	 res@cnFillOn      = True            ; Turn on contour fill
	 res@cnFillPalette = "BlAqGrYeOrRe"
	 res@tiMainString  = "Temperature_Aug"   ; Main title
	 
	 res@gsnYAxisIrregular2Linear = True ; converts irreg depth to linear
	 ;res@gsnYAxisIrregular2Log = True    ; Convert Y axis to logarithmic
	 res@trYReverse    		      = True ; reverse y axis
	 
	 res@tmYLMode              = "Explicit"
	 res@tmYLValues            = (/10,1000,2000,3000/)
	 res@tmYLLabels            = "" + res@tmYLValues + "m"

	 plot = gsn_csm_contour(wks, thetao_zon, res ) 
	 
	 ;res@tiMainString          = "Logarithmic axis"
	 ;res@gsnYAxisIrregular2Log = True    ; Convert Y axis to logarithmic
	 ;res@gsnXAxisIrregular2Log = True    ; Convert X axis to logarithmic

	;---Set some axis labels, otherwise all we'll get is "10^3".
	 ;res@tmYLMode              = "Explicit"
	 ;res@tmYLValues            = (/10,20,30,40/)
	 ;res@tmYLLabels            = "" + res@tmYLValues

	 ;plot = gsn_csm_contour(wks, thetao_zon, res ) 

end