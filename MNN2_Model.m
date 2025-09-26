function [derivLnSde,derivLnPGV] = MNN2_Model(X1,X2,X3,X4,X5,T,RotType)

% Created by Mao-Xin Wang (dr.maoxin.wang@gmail.com or maoxinwang@ust.hk)
% June 2025
% This code implements the MNN-II model to estimate the partial derivatives
% of logarithmic inelastic displacement with respect to lnSde and lnPGV
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
%   X4      =  scalar or matrix of RotD50 Sde (in cm)
%   X5      =  scalar or matrix of RotD50 PGV (in cm/s)
%   (Note: the above inputs must be in the same dimension)
%   T       =  scalar of spectral period (in s)
%   RotType =  'RotD50' or 'RotD100'
%
% OUTPUT
%
%   derivLnSde  =  partial derivative of lnSdi with respect to lnSde
%   derivLnPGV  =  partial derivative of lnSdi with respect to lnPGV
%
%   (Note: the outputs will automatically have same dimension as the predictors)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Periods = [0.030 0.050 0.075 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.75 1.0 1.5 2.0 3.0 4.0 5.0];

load paraALL_MNN2

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
    
    [derivLnSde_low,derivLnPGV_low] = MNN2_Model_sub(X1,X2,X3,X4,X5,paraALL{ip_low,idRot});
    [derivLnSde_high,derivLnPGV_high] = MNN2_Model_sub(X1,X2,X3,X4,X5,paraALL{ip_high,idRot});
    x = [log(T_low) log(T_high)];
    Y_Sde = [derivLnSde_low derivLnSde_high];
    Y_PGV = [derivLnPGV_low derivLnPGV_high];
    derivLnSde = Y_Sde(:,1)+(log(T)-x(1))./(x(2)-x(1)).*(Y_Sde(:,2)-Y_Sde(:,1));
    derivLnPGV = Y_PGV(:,1)+(log(T)-x(1))./(x(2)-x(1)).*(Y_PGV(:,2)-Y_PGV(:,1));
else
    ip_T = find(abs((Periods-T)) < 0.0001);
    [derivLnSde,derivLnPGV] = MNN2_Model_sub(X1,X2,X3,X4,X5,paraALL{ip_T,idRot});
end

derivLnSde = reshape(derivLnSde,n_row,n_col);
derivLnPGV = reshape(derivLnPGV,n_row,n_col);

%% sub-function
function [derivLnSde,derivLnPGV] = MNN2_Model_sub(X1,X2,X3,X4,X5,paraMNN)

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

n = paraMNN.n;
w = paraMNN.w;
b = paraMNN.b;

%% derivative with respect to lnSde
w0_Sde = paraMNN.w0_Sde;
b0_Sde = paraMNN.b0_Sde;
Y = b0_Sde;
for i = 1:n
    Y = Y+w0_Sde(1,i)./( 1+exp(-b(i)-w(1,i)*(X1_temp-X_min(1))/(X_max(1)-X_min(1))...
        -w(2,i)*(X2_temp-X_min(2))/(X_max(2)-X_min(2))...
        -w(3,i)*(X3_temp-X_min(3))/(X_max(3)-X_min(3))...
        -w(4,i)*(X4_temp-X_min(4))/(X_max(4)-X_min(4))...
        -w(5,i)*(X5_temp-X_min(5))/(X_max(5)-X_min(5))) );
end
derivLnSde = Y;

%% derivative with respect to lnPGV
w0_PGV = paraMNN.w0_PGV;
b0_PGV = paraMNN.b0_PGV;
Y = b0_PGV;
for i = 1:n
    Y = Y+w0_PGV(1,i)./( 1+exp(-b(i)-w(1,i)*(X1_temp-X_min(1))/(X_max(1)-X_min(1))...
        -w(2,i)*(X2_temp-X_min(2))/(X_max(2)-X_min(2))...
        -w(3,i)*(X3_temp-X_min(3))/(X_max(3)-X_min(3))...
        -w(4,i)*(X4_temp-X_min(4))/(X_max(4)-X_min(4))...
        -w(5,i)*(X5_temp-X_min(5))/(X_max(5)-X_min(5))) );
end
derivLnPGV = Y;
