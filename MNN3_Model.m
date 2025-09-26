function lambdaSTD = MNN3_Model(X1,X2,X3,X4,X5,X6,X7,T,RotType)

% Created by Mao-Xin Wang (dr.maoxin.wang@gmail.com or maoxinwang@ust.hk)
% June 2025
% This code implements the MNN-III model to estimate the ratio between
% MCS-based and linear-based total standard deviations
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
%   X6      =  scalar or matrix of standard deviation of lnSde
%   X7      =  scalar or matrix of standard deviation of lnPGV
%   (Note: the above inputs must be in the same dimension)
%   T       =  scalar of spectral period (in s)
%   RotType =  'RotD50' or 'RotD100'
%
% OUTPUT
%
%   lambdaSTD  =  ratio between MCS- and linear-based total standard deviations
%
%   (Note: the output will automatically have same dimension as the predictors)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Periods = [0.030 0.050 0.075 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.75 1.0 1.5 2.0 3.0 4.0 5.0];

load paraALL_MNN3

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
    
    LNlambdaSTD_low = MNN3_Model_sub(X1,X2,X3,X4,X5,X6,X7,paraALL{ip_low,idRot});
    LNlambdaSTD_high = MNN3_Model_sub(X1,X2,X3,X4,X5,X6,X7,paraALL{ip_high,idRot});
    x = [log(T_low) log(T_high)];
    Y_LNlambdaSTD = [LNlambdaSTD_low LNlambdaSTD_high];
    LNlambdaSTD = Y_LNlambdaSTD(:,1)+(log(T)-x(1))./(x(2)-x(1)).*(Y_LNlambdaSTD(:,2)-Y_LNlambdaSTD(:,1));
else
    ip_T = find(abs((Periods-T)) < 0.0001);
    LNlambdaSTD = MNN3_Model_sub(X1,X2,X3,X4,X5,X6,X7,paraALL{ip_T,idRot});
end
lambdaSTD = reshape(exp(LNlambdaSTD),n_row,n_col);

%% sub-function
function LNlambdaSTD = MNN3_Model_sub(X1,X2,X3,X4,X5,X6,X7,paraMNN)

X_min = [log(0.01) 	0 	log(0.025) 	paraMNN.X4min 	paraMNN.X5min 	paraMNN.X6min 	paraMNN.X7min];
X_max = [log(3) 	1 	log(0.05) 	paraMNN.X4max 	paraMNN.X5max 	paraMNN.X6max 	paraMNN.X7max];
[n_row,n_col] = size(X1);
n_data = n_row*n_col;
X1_temp = reshape(log(X1),n_data,1);
X2_temp = reshape(X2,n_data,1);
X3_temp = reshape(log(X3),n_data,1);
X4_temp = reshape(log(X4),n_data,1);
X5_temp = reshape(log(X5),n_data,1);
X6_temp = reshape(X6,n_data,1);
X7_temp = reshape(X7,n_data,1);

% NaN is specified for inputs outside their ranges
% X1_temp(X1_temp<X_min(1) | X1_temp>X_max(1)) = nan;
% X2_temp(X2_temp<X_min(2) | X2_temp>X_max(2)) = nan;
% X3_temp(X3_temp<X_min(3) | X3_temp>X_max(3)) = nan;
% X4_temp(X4_temp<X_min(4) | X4_temp>X_max(4)) = nan;
% X5_temp(X5_temp<X_min(5) | X5_temp>X_max(5)) = nan;
% X6_temp(X6_temp<X_min(6) | X6_temp>X_max(6)) = nan;
% X7_temp(X7_temp<X_min(7) | X7_temp>X_max(7)) = nan;

A = ([X1_temp,X2_temp,X3_temp,X4_temp,X5_temp,X6_temp,X7_temp]-repmat(X_min,[n_data,1]))./(repmat(X_max-X_min,[n_data,1]));

W1 = paraMNN.W1;
W2 = paraMNN.W2;
W3 = paraMNN.W3;
B1 = paraMNN.B1;
B2 = paraMNN.B2;
B3 = paraMNN.B3;

LNlambdaSTD = sigmoid(sigmoid(A*W1+B1)*W2+B2)*W3+B3;


function y = sigmoid(x)
y = 1./(1+exp(-x));
