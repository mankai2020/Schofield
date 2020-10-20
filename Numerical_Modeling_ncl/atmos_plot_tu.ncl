begin

	;sfile=systemfunc("ls /work1/mankai/ERA_interim/t_u*");
	;f=addfiles(sfile,"r");

	newfile="/work1/mankai/ERA_interim/t_u_2000.nc";
	f1=addfile(newfile,"r");
	t=f1->t(:,:,:,:)
	
	;t=f[:]->t(1,1,:,1);
	;t_zon=dim_avg(t);
	;t_zon!0="level";
	;t_zon!1="latitude";
	;t_zon&level=t&level;
	;t_zon&latitude=t&latitude;
	;print(t)
	printVarSummary(t);
	;printVarSummary(t_zon);

	u=f[:]->u(0,:,:,:);
	u_zon=dim_avg(dim_avg(u));
	u_zon!0="level";
	u_zon!1="latitude";
	u_zon&level=u&level;
	u_zon&latitude=u&latitude;
	print(u_zon(1,1))
	;printVarSummary(u);
	;printVarSummary(u_zon);
	
	;===========================
	; plot parameters
	;===========================
	
	 wks_type = "png"
	 ;wks_type@wkWidth = 800
	 ;wks_type@wkHeight = 500

	 wks   = gsn_open_wks (wks_type, "t_u_zon" )         ; send graphics to PNG file

	 res               = True            ; Plot mods desired
	 
	 ;res@vpWidthF = 3
	 ;res@vpHeightF = 1.8
	 
	 res@gsnMaximize   = True            ; Maximize plot in frame
	 res@cnFillOn      = True            ; Turn on contour fill
	 res@cnFillPalette = "BlAqGrYeOrRe"
	 res@tiMainString  = "Salinity_Aug"  ; Main title
	 
	 ;res@gsnYAxisIrregular2Linear = True ; converts irreg depth to linear
	 ;res@gsnYAxisIrregular2Log = True    ; Convert Y axis to logarithmic
	 res@trYReverse    		      = True ; reverse y axis
	 
	 ;res@tmYLMode              = "Explicit"
	 ;res@tmYLValues            = (/10,100,1000,3000/)
	 ;res@tmYLLabels            = "" + res@tmYLValues

	 plotA = gsn_csm_contour(wks, t_zon, res ) 
	 
	 res@cnFillOn      = False           ; Turn on contour fill

	 plotB = gsn_csm_contour(wks, u_zon, res ) 
	 
	 overlay(plotA,plotB)                            ; Overlay plots; plot1 now
	 draw(plotA)                                     ; Draw base plot.

end