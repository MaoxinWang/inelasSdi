function [medianSdi,sigmaLnSdi,tauLnSdi,phiLnSdi] = MNN4_Model(X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,T,RotType)

% Created by Mao-Xin Wang (dr.maoxin.wang@gmail.com or maoxinwang@ust.hk)
% June 2025
% This code implements the MNN-IV model to estimate the median and
% logarithmic standard deviation of inelastic spectral displacement
% given seismological and site parameters
%
% If you use this code in your work, it is requested that you cite the following article:
% Wang, M.X., Wang, G., and Tian, Y. (2025). "Conditional and non-conditional predictive models for inelastic spectral displacement demands based on machine learning."
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT
%
%   X1      =  scalar or matrix of Cy
%   X2      =  scalar or matrix of I_hys
%   X3      =  scalar or matrix of damping ratio
%   X4      =  scalar or matrix of earthquake magnitude
%   X5      =  scalar or matrix of Ztor (in km)
%   X6      =  scalar or matrix of Frv
%   X7      =  scalar or matrix of Fnm
%   X8      =  scalar or matrix of Rrup (in km)
%   X9      =  scalar or matrix of Vs30 (in m/s)
%   X10     =  scalar or matrix of dZ25 (km)
%   (Note: the above inputs must be in the same dimension)
%   T       =  scalar of spectral period (in s)
%   RotType =  'RotD50' or 'RotD100'
%
% OUTPUT
%
%   medianSdi  =  median inelastic displacement (in cm)
%   sigmaLnSdi =  total standard deviation of logarithmic displacement
%   tauLnSdi   =  between-event standard deviation of logarithmic displacement
%   phiLnSdi   =  within-event standard deviation of logarithmic displacement
%
%   (Note: the outputs will automatically have same dimension as the predictors)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Periods = [0.030 0.050 0.075 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.75 1.0 1.5 2.0 3.0 4.0 5.0];

load paraALL_MNN4
load para_stdNonCond

if mode(X2)==0
    HysType = 'Bilinear';
elseif mode(X2)==1
    HysType = 'Takeda';
end
if strcmp(RotType,'RotD50')
    idRot = 1;
elseif strcmp(RotType,'RotD100')
    idRot = 2;
end
[n_row,n_col] = size(X1);

medianSdi_allT = MNN4_Model_sub(X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,paraALL{idRot});
if (isempty(find(abs(Periods-T) < 0.0001, 1)))
    T_low = max(Periods(Periods < T));
    T_high = min(Periods(Periods > T));
    ip_low  = find(Periods==T_low);
    ip_high = find(Periods==T_high);
    x = [log(T_low) log(T_high)];
    Y = log(medianSdi_allT(:,[ip_low,ip_high]));
    medianSdi = exp(Y(:,1)+(log(T)-x(1))./(x(2)-x(1)).*(Y(:,2)-Y(:,1)));
else
    ip_T = find(abs((Periods-T)) < 0.0001);
    medianSdi = medianSdi_allT(:,ip_T);
end
[tauLnSdi,phiLnSdi] = interp_STDnonCond(T,para_stdNonCond,RotType,HysType,X1,X3);

medianSdi = reshape(medianSdi,n_row,n_col);
tauLnSdi = reshape(tauLnSdi,n_row,n_col);
phiLnSdi = reshape(phiLnSdi,n_row,n_col);
sigmaLnSdi = (tauLnSdi.^2+phiLnSdi.^2).^0.5;

%% sub-function 1
function medianSdi = MNN4_Model_sub(X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,paraMNN)

X_min = [log(0.01) 	0 	log(0.025) 	3 	paraMNN.X5min 	0 	0 	0.05 	log(0.05) 	paraMNN.X10min 	paraMNN.X11min];
X_max = [log(3) 	1 	log(0.05) 	8 	paraMNN.X5max 	1 	1 	80 	log(80) 	paraMNN.X10max 	paraMNN.X11max];
[n_row,n_col] = size(X1);
n_data = n_row*n_col;
X1_temp = reshape(log(X1),n_data,1);
X2_temp = reshape(X2,n_data,1);
X3_temp = reshape(log(X3),n_data,1);
X4_temp = reshape(X4,n_data,1);
X5_temp = reshape(X5,n_data,1);
X6_temp = reshape(X6,n_data,1);
X7_temp = reshape(X7,n_data,1);
X8_temp = reshape(X8,n_data,1);
X9_temp = reshape(log(X8),n_data,1);
X10_temp = reshape(log(X9),n_data,1);
X11_temp = reshape(X10,n_data,1);

% NaN is specified for inputs outside their ranges
% X1_temp(X1_temp<X_min(1) | X1_temp>X_max(1)) = nan;
% X2_temp(X2_temp<X_min(2) | X2_temp>X_max(2)) = nan;
% X3_temp(X3_temp<X_min(3) | X3_temp>X_max(3)) = nan;
% X4_temp(X4_temp<X_min(4) | X4_temp>X_max(4)) = nan;
% X5_temp(X5_temp<X_min(5) | X5_temp>X_max(5)) = nan;
% X6_temp(X6_temp<X_min(6) | X6_temp>X_max(6)) = nan;
% X7_temp(X7_temp<X_min(7) | X7_temp>X_max(7)) = nan;
% X8_temp(X8_temp<X_min(8) | X8_temp>X_max(8)) = nan;
% X9_temp(X9_temp<X_min(9) | X9_temp>X_max(9)) = nan;
% X10_temp(X10_temp<X_min(10) | X10_temp>X_max(10)) = nan;
% X11_temp(X11_temp<X_min(11) | X11_temp>X_max(11)) = nan;

A = ([X1_temp,X2_temp,X3_temp,X4_temp,X5_temp,X6_temp,X7_temp,X8_temp,X9_temp,X10_temp,X11_temp]-...
    repmat(X_min,[n_data,1]))./(repmat(X_max-X_min,[n_data,1]));

W1 = paraMNN.W1;
W2 = paraMNN.W2;
B1 = paraMNN.B1;
B2 = paraMNN.B2;

medianSdi = exp(sigmoid(A*W1+B1)*W2+B2);


%% sub-function 2
function y = sigmoid(x)
y = 1./(1+exp(-x));

