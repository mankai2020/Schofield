begin

	sfile=systemfunc("ls /work1/mankai/ERA_interim/t_u*");
	f=addfiles(sfile,"r");
	ListSetType(f, "join");
	
	t=f[:]->t(:,0,:,:,:)
	t_zon=dim_avg(dim_avg_n(t,0));
	t_zon!0="level";
	t_zon!1="latitude";
	t_zon&level=t&level;
	t_zon&latitude=t&latitude;
	t_zon=t_zon*0.0021+248.78
	;print(t)
	;printVarSummary(t);
	;printVarSummary(t_zon);

	u=f[:]->u(:,0,:,:,:);
	u_zon=dim_avg(dim_avg_n(u,0));
	u_zon!0="level";
	u_zon!1="latitude";
	u_zon&level=u&level;
	u_zon&latitude=u&latitude;
	u_zon=u_zon*0.0029+30.36
	;print(u_zon(1,1))
	;printVarSummary(u);
	;printVarSummary(u_zon);
	
	level = f[0]->level;
	printVarSummary(level)
	pressure = (/level/)*100;
	printVarSummary(pressure)
	theta_zon = t_zon(5:36,:);
	theta_zon = pot_temp(pressure(5:36),t_zon(5:36,:),0,False);
	
	;===========================
	; plot parameters
	;===========================
	
	 wks_type = "png"
	 wks   = gsn_open_wks (wks_type, "t_u_zon_Jan" )

	 res               = True            ; Plot mods desired
	 res@vpWidthF = 3
	 res@vpHeightF = 1.8
	 
	 res@gsnDraw      = False                        ; don't draw yet
	 res@gsnFrame     = False                        ; don't advance frame yet
	 
	 res@gsnMaximize   = True            ; Maximize plot in frame
	 res@cnFillOn      = True            ; Turn on contour fill
	 res@cnFillPalette = "BlAqGrYeOrRe"
	 res@tiMainString  = "Temperature & U Wind (Jan)"  ; Main title
	 res@gsnRightString   = "K"    
	 res@gsnLeftString    = "m/s"
	 res@gsnCenterString  = ""
	 res@cnLineThicknessF = 0.                       ; line thickness
	 
	 res@gsnYAxisIrregular2Linear = True ; converts irreg depth to linear
	 ;res@gsnYAxisIrregular2Log = True    ; Convert Y axis to logarithmic
	 res@trYReverse    		      = True ; reverse y axis
	 
	 ;res@tmYLMode              = "Explicit"
	 ;res@tmYLValues            = (/10,100,1000,3000/)
	 ;res@tmYLLabels            = "" + res@tmYLValues

	 plotA = gsn_csm_contour(wks, t_zon, res ) 
	 
	 res@cnFillOn      = False           ; Turn off contour fill
	 res@cnLineThicknessF = 2.                       ; line thickness

	 plotB = gsn_csm_contour(wks, u_zon, res ) 
	 overlay(plotA,plotB)                            ; Overlay plots; plot1 now
	 draw(plotA)                                     ; Draw base plot.
	 frame(wks)
	 
	 	;===========================
	; plot parameters
	;===========================

	 wks2_type = "png"
	 wks2   = gsn_open_wks (wks2_type, "pt_zon_Jan" )
	
	 res2               = True            ; Plot mods desired
	 res2@vpWidthF = 3
	 res2@vpHeightF = 1.8
	 	 
	 res2@gsnMaximize   = True            ; Maximize plot in frame
	 res2@cnFillOn      = True            ; Turn on contour fill
	 res2@cnFillPalette = "BlAqGrYeOrRe"
	 res2@tiMainString  = "Potential Temperature (Jan)"  ; Main title
	 res2@gsnRightString   = "K"    
	 res2@gsnLeftString    = ""
	 res2@gsnCenterString  = ""
	 res2@cnLineThicknessF = 1.                       ; line thickness
	 
	 res2@gsnYAxisIrregular2Linear = True ; converts irreg depth to linear
	 ;res2@gsnYAxisIrregular2Log = True    ; Convert Y axis to logarithmic
	 res2@trYReverse    		      = True ; reverse y axis
	 
	 plot = gsn_csm_contour(wks2, theta_zon, res2) 
	 
end