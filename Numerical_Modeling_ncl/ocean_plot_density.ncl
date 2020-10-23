begin

	;===========================
	;input temp&salinity data
	;===========================
	
	sfile=systemfunc("ls /work1/mankai/ORAS/so*");
	f=addfiles(sfile,"r");
	ListSetType(f, "join");
	so=f[:]->so(:,7,0:36,:,327:339);
	so_avg=dim_avg_n(so,0);
	
	sfile2=systemfunc("ls /work1/mankai/ORAS/thetao*");
	f2=addfiles(sfile2,"r");
	ListSetType(f2, "join");
	thetao=f2[:]->thetao(:,7,0:36,:,327:339);
	thetao_avg=dim_avg_n(thetao,0);
	
	;===========================
	;calculate density
	;===========================
	
	level=0;
	depth=0.0;
	density=new((/37,180,13/),float);
	do level=0,36
		;print(so&depth(level));
		depth=so&depth(level);
		density(level,:,:)=rho_mwjf(thetao_avg(level,:,:),so_avg(level,:,:),depth)
	end do
	density_zon=dim_avg(density);
	density_zon!0="depth";
	density_zon!1="lat";
	density_zon&depth=so&depth(0:36);
	density_zon&lat=so&lat;
	;printVarSummary(density_zon);
	
	;===========================
	; plot parameters
	;===========================
	
	 wks_type = "png"
	 wks   = gsn_open_wks (wks_type, "density_zon_Aug" )         ; send graphics to PNG file

	 res               = True            ; Plot mods desired
	 res@vpWidthF = 3
	 res@vpHeightF = 1.8
	 
	 res@gsnMaximize   = True            ; Maximize plot in frame
	 res@cnFillOn      = True            ; Turn on contour fill
	 res@cnFillPalette = "BlAqGrYeOrRe"
	 res@tiMainString  = "density_Aug"   ; Main title
	 res@gsnRightString   = "g/cm^3" 
	 
	 res@gsnYAxisIrregular2Linear = True ; converts irreg depth to linear
	 ;res@gsnYAxisIrregular2Log = True    ; Convert Y axis to logarithmic
	 res@trYReverse    		      = True ; reverse y axis
	 
	 res@tmYLMode              = "Explicit"
	 res@tmYLValues            = (/10,1000,2000,3000/)
	 res@tmYLLabels            = "" + res@tmYLValues + "m" 

	 plot = gsn_csm_contour(wks, density_zon, res ) 

end