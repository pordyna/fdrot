tic;
clc;
clear;
set(0,'defaulttextinterpreter','none');
load FireColormap; %colormap for density and intensity distributions
load JetWithGreyColormap; %colormap for magnetic fields etc. with grey for zero values
load HighContrastColormap; %colormap for magnetic fields etc. with grey for zero values with higher contrast

%% Numerical parameters
gridsize=200; %grid size number (radius)
cellsize=0.1; % cell size in micron
coord=linspace(-gridsize*cellsize,gridsize*cellsize,2*gridsize+1);%define coordinates in micron

load filaments_100_a.mat; % Current filament position (indexed) and amplitude (normalized)

Scale=5; % to play with B-field amplitude, gives ~ "Scale" kT

%% Physical settings
A=27; % mass number
Z=13; %ionization state
rho=2.7; %mass density in g/cc
T_sample=0.6;

% A=63.5; % mass number
% Z=20; %ionization state
% rho=8.9; %mass density in g/cc

Obs=6.457; %Photon energy of observation in keV
Analyzer=2; % Orientation of analyzer in mrad
Extinction=1e-6; %Extinction of analyzer

N_0=5e12; %number of photons in total
T_opt=0.045; % transmission of all optical elements
A_field=(20)^2; % field size in microns, homogeneous illumination

M=13; %magnification
Imageblur=1; %spatial resolution in micron
Pixelsize=20; %pixel size in micron 

N_phot_p_px = N_0 * Pixelsize^2/M^2/A_field*T_opt*T_sample; %number of incident photons per detector pixel
A_px=Pixelsize/M; %what one pixel corresponds to in real space, in micron

K_conf = 1; %confidence factor

%% First general checks and output
if (Analyzer/1000)^2 > 10*Extinction,
    disp('Analyzer setting is very good.');
elseif (Analyzer/1000)^2 < Extinction,
    disp('Analyzer setting should be larger');
else disp('Analyzer setting is sufficient');
end;

Phi_min=sqrt(( Extinction + (Analyzer/1000)^2)*K_conf^2/ (4*N_phot_p_px))/Analyzer*1e6; % minimum detectable rotation in mrad
disp(['Minimum detectable rotation is ', num2str(Phi_min),32,'mrad.']);
%% Show current distribution
Currentfield = zeros(2*gridsize+1);
for i=1:length(Filaments)
    Currentfield(Filaments(i,1),Filaments(i,2))=Filaments(i,3);
end;
% figure(1);
% imagesc(Currentfield); axis equal;

%% Create Biot-Savart-Kernel
resolution=69; % radial distance
KernelX = zeros(2*resolution+1); % x-component
KernelY = zeros(2*resolution+1); % y-component
for i = 1 : 2*resolution+1
    for j =  1 : 2*resolution+1
        if ~(i==resolution && j == resolution),
            KernelX(i,j)=-(i-resolution)/((i-resolution)^2+(j-resolution)^2);
            KernelY(i,j)=(j-resolution)/((i-resolution)^2+(j-resolution)^2);
        else
            KernelX(i,j)=1;
            KernelY(i,j)=1;
        end;
    end;
end;
%figure(2);
%imagesc(sqrt(KernelX.^2+KernelY.^2));axis equal;

%% Create global B-Field distribution
Bx = zeros(2*gridsize+1);
By = zeros(2*gridsize+1);

for i=1:length(Filaments)
    x=Filaments(i,1);
    y=Filaments(i,2);
    val=Scale*Filaments(i,3);
    
    Bx(x-resolution:x+resolution,y-resolution:y+resolution)=val*KernelX+Bx(x-resolution:x+resolution,y-resolution:y+resolution);
    By(x-resolution:x+resolution,y-resolution:y+resolution)=val*KernelY+By(x-resolution:x+resolution,y-resolution:y+resolution);
end;
clear x y val;
%fields are in kT from now on (numerical values are reasonable)

figure(3)
imagesc(coord,coord,sqrt(Bx.^2+By.^2));axis equal;
title('Magnetic field amplitude in kT');
colormap(FireColormap); colorbar;

figure(4)
imagesc(coord,coord,Bx);axis equal;
title('Magnetic field x component in kT');
caxis([-Scale, Scale]);colormap(HighContrastColormap); colorbar;

figure(5)
imagesc(coord,coord,By);axis equal;
title('Magnetic field y component in kT');
caxis([-Scale, Scale]);colormap(HighContrastColormap); colorbar;

%% Compute Faraday rotation
FaradayX=sum(Bx,2);
FaradayY=sum(By,1);
%making the units
density_unit=rho/1.66e-24/A*Z; %electron density in cm^-3
nc_x=7.25215814834367507619e26*Obs^2; %critical density of x-ray probe beam in cm^-3
const=293.35194015652920913458; %e_0/2/c_0/m_e
RotationX=const*cellsize*density_unit/nc_x*FaradayX; %Faraday rotation in mrad
RotationY=const*cellsize*density_unit/nc_x*FaradayY; %Faraday rotation in mrad

figure(6);
plot(coord,RotationX,'-xr',coord,RotationY,'-xg'); title('Faraday rotation along X (red) and Y (green) axes in mrad');

%% Compute transmission
TransX = 1 - (1 - Extinction)*cos((RotationX+Analyzer)/1000).^2; %note that angles were in mrad -> factor 1000!
TransY = 1 - (1 - Extinction)*cos((RotationY+Analyzer)/1000).^2; %note that angles were in mrad -> factor 1000!

figure(9);
plot(coord,TransX,'-xr',coord,TransY,'-xg');
title({strcat('Transmission through polarizer at ',32,int2str(Analyzer),' mrad with',32,num2str(Extinction),' extinction.');'for a probe beam along X (red) and Y (green) axes.'});

%% Emulate image lineout with binning into pixels

tmp1 = TransX;
tmp2 = TransY;

Pixelbinning = round(A_px / cellsize); %number of cells which are eaten by one pixel
Pixelcount = floor((2*gridsize+1)*cellsize/A_px); %number of pixels along lineout
LineoutX=zeros(1,Pixelcount);
LineoutY=zeros(1,Pixelcount);

for i=1:Pixelcount
    LineoutX(i)=round(sum(tmp1((i-1)*Pixelbinning+1 : i*Pixelbinning))/Pixelbinning*N_phot_p_px);
    LineoutY(i)=round(sum(tmp2((i-1)*Pixelbinning+1 : i*Pixelbinning))/Pixelbinning*N_phot_p_px);
end;
clear tmp1 tmp2;

Imagecoord=linspace(-gridsize*cellsize,gridsize*cellsize,Pixelcount);%define detector coordinates in micron

figure(11);
clf;
hold on;
stairs(Imagecoord,LineoutX,'-xr');
stairs(Imagecoord,LineoutY,'-xg'); 
title({'Detector signal for a probe beam along X (red) and Y (green) axes in photons per pixel.';strcat('Imaging resolution is',32,num2str(Imageblur),' micron, one pixel corresponds to',32,num2str(A_px),' micron on the object.');strcat('Initially',32,num2str(N_0,'%g'),' photons in beam.')});
hold off;

XErr=K_conf*sqrt(LineoutX);
YErr=K_conf*sqrt(LineoutY);
%XOffset=gridsize*cellsize/(Pixelcount-1);

figure(12);
clf;
hold on;
errorbar(Imagecoord,LineoutX,XErr,'sr');
errorbar(Imagecoord,LineoutY,YErr,'sg');
%stairs(Imagecoord,LineoutX,'-xr');
%stairs(Imagecoord,LineoutY,'-xg'); 
title({'Detector signal for a probe beam along X (red) and Y (green) axes in photons per pixel.';strcat('Imaging resolution is',32,num2str(Imageblur),' micron, one pixel corresponds to',32,num2str(A_px),' micron on the object.');strcat('Initially',32,num2str(N_0,'%g'),' photons in beam. Error bars of',32,num2str(K_conf),' sigma.')});
hold off;

Noiselevel=sqrt(N_phot_p_px*(Extinction+(Analyzer/1000)^2));
NoisyLineoutX= LineoutX + Noiselevel*randn(1,Pixelcount);
NoisyLineoutY= LineoutY + Noiselevel*randn(1,Pixelcount);

figure(13);
clf;
hold on;
stairs(Imagecoord,NoisyLineoutX,'-xr');
stairs(Imagecoord,NoisyLineoutY,'-xg'); 
title({'Detector signal for a probe beam along X (red) and Y (green) axes in photons per pixel.';strcat('Imaging resolution is',32,num2str(Imageblur),' micron, one pixel corresponds to',32,num2str(A_px),' micron on the object.');strcat('Initially',32,num2str(N_0,'%g'),' photons in beam. !! WITH NOISE !!')});
hold off;

%% Play with FFT
% L = length(coord); %number of sampling points in lineout
% NFFT = 2^nextpow2(L); %number of points to compute FFT
% FFTX = fft(TransX,NFFT)/L; %compute fft
% f = 1/cellsize/2*linspace(0,1,NFFT/2+1); %frequency vector
% 
% figure(10);
% semilogy(f,2*abs(FFTX(1:NFFT/2+1)),'-o') ;

toc;