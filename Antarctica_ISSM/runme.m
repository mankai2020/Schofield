% create domain

% domain = ['./DomainOutline.exp'];
domain = ['./coastline_10ma_100.exp'];
hinit = 50000;
hmax = 200000;
hmin = 10000;
gradation = 100;
err = 8;

% Gnerate an inital mesh
md = bamg(model,'domain',domain,'hmax',hinit);

% Load lndmask
antarData = './Antarctica_new_10ma.nc';

x1    = ncread(antarData,'x1');
y1    = ncread(antarData,'y1');
topg  = ncread(antarData,'topg_smooth');
topg(find(topg==0))=nan;
thkmask=ncread(antarData,'mask');
thkmask(find(thkmask==0))=nan;

md.geometry.base    = InterpFromGridToMesh(x1,y1,topg,md.mesh.x,md.mesh.y,0);

% Adapt the mesh to minimize error in velocity interpolation
md=bamg(md,'hmax',hmax,'hmin',hmin,'gradation',gradation,'field',md.geometry.base,'err',err);

md.geometry.base    = InterpFromGridToMesh(x1,y1,topg,md.mesh.x,md.mesh.y,0);

clear topg;

%interpolate onto our mesh vertices
groundedice=InterpFromGridToMesh(x1,y1,thkmask,md.mesh.x,md.mesh.y,0);
groundedice(groundedice>0)=1;
groundedice(groundedice<=0)=-1;
clear thkmask;

%fill in the md.mask structure
md.mask.groundedice_levelset=groundedice; %ice is grounded for mask equal one
md.mask.ice_levelset=-1*ones(md.mesh.numberofvertices,1);%ice is present when negatvie

% Parameterization
md = parameterize(md,'./Antarctica.par');

% Extrude
md = extrude(md,3,1);

% Set flow equation
md = setflowequation(md,'HO','all');
% md = setflowequation(md,'SSA','all');

% Transient Run
md.inversion.iscontrol=1;
md.transient.ismasstransport=1;
md.transient.isstressbalance=1;
md.transient.isgroundingline=0;
md.transient.ismovingfront=0;
md.transient.isthermal=1;

md.timestepping.time_step=10;
%md.timestepping.start_time=100;
md.settings.output_frequency=50;
md.timestepping.final_time=50000;
md.transient.requested_outputs={'default','IceVolume'};

md.cluster=generic('name',oshostname,'np',48);
md=solve(md,'Transient');

% Save model
save ./Model/Antarctica_Transient md;

