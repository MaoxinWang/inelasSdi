function [medianSdi,stdLnSdi] = MNN1_Model(X1,X2,X3,X4,X5,T,RotType)

% Created by Mao-Xin Wang (dr.maoxin.wang@gmail.com or maoxinwang@ust.hk)
% June 2025
% This code implements the MNN-I model to estimate the median and
% logarithmic standard deviation of inelastic spectral displacement
% conditioned on PGV and elastic spectral displacement
%
% If you use this code in your work, it is requested that you cite the following article:
% Wang, M.X., Wang, G., and Tian, Y. (2025). "Conditional and non-conditional predictive models for inelastic spectral displacement demands based on machine learning."
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT
%
%   X1      =  scalar or matrix of Cy
%   X2      =  scalar or matrix of I_hys (0 for bilinear and 1 for Takeda)
%   X3      =  scalar or matrix of damping ratio
%   X4      =  scalar or matrix of RotD50 Sde (in cm)
%   X5      =  scalar or matrix of RotD50 PGV (in cm/s)
%   (Note: the above inputs must be in the same dimension)
%   T       =  scalar of spectral period (in s)
%   RotType =  'RotD50' or 'RotD100'
%
% OUTPUT
%
%   medianSdi  =  median inelastic displacement (in cm)
%   stdLnSdi   =  conditional standard deviation of logarithmic displacement (sigma_cond)
%
%   (Note: the outputs will automatically have same dimension as the predictors)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Periods = [0.030 0.050 0.075 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.75 1.0 1.5 2.0 3.0 4.0 5.0];

load paraALL_MNN1
load para_stdCond

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

if (isempty(find(abs(Periods-T) < 0.0001, 1)))
    T_low = max(Periods(Periods < T));
    T_high = min(Periods(Periods > T));
    ip_low  = find(Periods==T_low);
    ip_high = find(Periods==T_high);
    
    id_row = contains(para_stdCond.Component, RotType) &...
        contains(para_stdCond.Hysteresis, HysType) & abs(para_stdCond.Period-T_low)<0.0001;
    [Sdi_low,stdLnSdi_low] = MNN1_Model_sub(X1,X2,X3,X4,X5,T_low,paraALL{ip_low,idRot},para_stdCond(id_row,:));
    id_row = contains(para_stdCond.Component, RotType) &...
        contains(para_stdCond.Hysteresis, HysType) & abs(para_stdCond.Period-T_high)<0.0001;
    [Sdi_high,stdLnSdi_high] = MNN1_Model_sub(X1,X2,X3,X4,X5,T_high,paraALL{ip_high,idRot},para_stdCond(id_row,:));
    x = [log(T_low) log(T_high)];
    Y_Sdi = [log(Sdi_low) log(Sdi_high)];
    Y_STD = [stdLnSdi_low stdLnSdi_high];
    medianSdi = exp(Y_Sdi(:,1)+(log(T)-x(1))./(x(2)-x(1)).*(Y_Sdi(:,2)-Y_Sdi(:,1)));
    stdLnSdi = Y_STD(:,1)+(log(T)-x(1))./(x(2)-x(1)).*(Y_STD(:,2)-Y_STD(:,1));
else
    ip_T = find(abs((Periods-T)) < 0.0001);
    id_row = contains(para_stdCond.Component, RotType) &...
        contains(para_stdCond.Hysteresis, HysType) & abs(para_stdCond.Period-T)<0.0001;
    [medianSdi,stdLnSdi] = MNN1_Model_sub(X1,X2,X3,X4,X5,T,paraALL{ip_T,idRot},para_stdCond(id_row,:));
end
medianSdi = reshape(medianSdi,n_row,n_col);
stdLnSdi = reshape(stdLnSdi,n_row,n_col);


%% sub-function
function [medianSdi,stdLnSdi] = MNN1_Model_sub(X1,X2,X3,X4,X5,T,paraMNN,paraSTD)

X_min = [log(0.01) 	0 	log(0.025) 	paraMNN.X4min 	paraMNN.X5min];
X_max = [log(3) 	1 	log(0.05) 	paraMNN.X4max 	paraMNN.X5max];
[n_row,n_col] = size(X1);
n_data = n_row*n_col;
X1_temp = reshape(log(X1),n_data,1);
X2_temp = reshape(X2,n_data,1);
X3_temp = reshape(log(X3),n_data,1);
X4_temp = reshape(log(X4),n_data,1);
X5_temp = reshape(log(X5),n_data,1);

% NaN is specified for inputs outside their ranges
% X1_temp(X1_temp<X_min(1) | X1_temp>X_max(1)) = nan;
% X2_temp(X2_temp<X_min(2) | X2_temp>X_max(2)) = nan;
% X3_temp(X3_temp<X_min(3) | X3_temp>X_max(3)) = nan;
% X4_temp(X4_temp<X_min(4) | X4_temp>X_max(4)) = nan;
% X5_temp(X5_temp<X_min(5) | X5_temp>X_max(5)) = nan;

%% median
n = paraMNN.n;
w = paraMNN.w;
w0 = paraMNN.w0;
b = paraMNN.b;
b0 = paraMNN.b0;

Y = b0;
for i = 1:n
    Y = Y+w0(1,i)./( 1+exp(-b(i)-w(1,i)*(X1_temp-X_min(1))/(X_max(1)-X_min(1))...
        -w(2,i)*(X2_temp-X_min(2))/(X_max(2)-X_min(2))...
        -w(3,i)*(X3_temp-X_min(3))/(X_max(3)-X_min(3))...
        -w(4,i)*(X4_temp-X_min(4))/(X_max(4)-X_min(4))...
        -w(5,i)*(X5_temp-X_min(5))/(X_max(5)-X_min(5))) );
end
medianSdi = exp(Y);

%% conditional standard deviation
Sdy = 100.*9.81.*exp(X1_temp).*T.^2./(4*pi^2);  % in cm
rx = X4_temp-log(Sdy);

% damping ratio of 2.5%
id = 1;
c0 = paraSTD.c0(id);
c1 = paraSTD.c1(id);
c2 = paraSTD.c2(id);
c3 = paraSTD.c3(id);
c4 = paraSTD.c4(id);
c5 = paraSTD.c5(id);
c6 = paraSTD.c6(id);
c7 = paraSTD.c7(id);
Y1 = (c0+c3*rx+c4*rx.^2)./(1+c5*rx+c6*rx.^2);
indRx = rx<0;
Y1(indRx) = (c0+c1*max(rx(indRx),-3))./(1+c2*max(rx(indRx),-3));
indRx = rx>c7;
Y1(indRx) = (c0+c3*c7+c4*c7.^2)./(1+c5*c7+c6*c7.^2);

% damping ratio of 5%
id = 2;
c0 = paraSTD.c0(id);
c1 = paraSTD.c1(id);
c2 = paraSTD.c2(id);
c3 = paraSTD.c3(id);
c4 = paraSTD.c4(id);
c5 = paraSTD.c5(id);
c6 = paraSTD.c6(id);
c7 = paraSTD.c7(id);
Y2 = (c0+c3*rx+c4*rx.^2)./(1+c5*rx+c6*rx.^2);
indRx = rx<0;
Y2(indRx) = (c0+c1*max(rx(indRx),-3))./(1+c2*max(rx(indRx),-3));
indRx = rx>c7;
Y2(indRx) = (c0+c3*c7+c4*c7.^2)./(1+c5*c7+c6*c7.^2);

% interpolation for damping ratio
Y12 = Y1+(Y2-Y1).*(X3_temp-X_min(3))./(X_max(3)-X_min(3));
stdLnSdi = Y12;

