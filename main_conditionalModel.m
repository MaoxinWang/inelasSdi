function main_conditionalModel

RotType = 'RotD50';  % RotD50 or RotD100
T = 0.3;  % period (in second) between 0.03 and 5 s
Cy = 0.05;  % yeild strength coefficient between 0.01 and 3
I_hys = 0;  % 0 for bilinear and 1 for Takeda
damping = 0.05;  % damping ratio between 0.025 and 0.05
Sde = 1.9;  % median of elastic spectral displacement at T (in cm)
PGV = 38.0;  % median of peak ground velocity (in cm/s)
sigma_lnSde = 0.60;  % logarithmic Std. Dev. of elastic spectral displacement at T
sigma_lnPGV = 0.60;  % logarithmic Std. Dev. of PGV
rho_SdePGV = 0.65;  % correlation coefficient between lnSde(T) and lnPGV

% predict Sdi conditioned on 'exact' Sde and PGV (sigma_lnSde=sigma_lnPGV=0)
[medianSdi,stdLnSdi] = MNN1_Model(Cy,I_hys,damping,Sde,PGV,T,RotType);

% estimate total uncertainty using linear approximation (Eq. 13)
[derivLnSde,derivLnPGV] = MNN2_Model(Cy,I_hys,damping,Sde,PGV,T,RotType);
sigma_linear = stdLnSdi.^2+sigma_lnSde.^2.*derivLnSde.^2+...
    sigma_lnPGV.^2.*derivLnPGV.^2+2*rho_SdePGV.*sigma_lnSde.*sigma_lnPGV.*derivLnSde.*derivLnPGV;

% estimate ratio between MCS and linear based standard deviations
lambdaSTD = MNN3_Model(Cy,I_hys,damping,Sde,PGV,sigma_lnSde,rho_SdePGV,T,RotType);

% estimate final total standard deviation
sigma_total = sigma_linear.*lambdaSTD;

