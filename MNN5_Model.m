function [medianRatio,stdLnRatio] = MNN5_Model(T,ratio_SdeSdy,damp)

% Created by Mao-Xin Wang (dr.maoxin.wang@gmail.com or maoxinwang@ust.hk)
% June 2025
% This code implements the MNN-V model to estimate the median and
% logarithmic standard deviation of the RotD50/RotD100 ratio of inelastic 
% spectral displacement
%
% If you use this code in your work, it is requested that you cite the following article:
% Wang, M.X., Wang, G., and Tian, Y. (2025). "Conditional and non-conditional predictive models for inelastic spectral displacement demands based on machine learning."
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT
%
%   T             =  scalar of spectral period (in s)
%   ratio_SdeSdy  =  scalar or matrix of the Sde/Sdy ratio
%   damp          =  scalar or matrix of damping ratio (within [0.025, 0.05] interval)
%
% OUTPUT
%
%   medianRatio  =  median RotD50/RotD100 ratio of inelastic displacement
%   stdLnRatio   =  standard deviation of logarithmic RotD50/RotD100 ratio
%
%   (Note: the outputs will automatically have same dimension as the predictors)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Periods = [0.030 0.050 0.075 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.75 1.0 1.5 2.0 3.0 4.0 5.0];

load paraALL_MNN5

[n_row,n_col] = size(ratio_SdeSdy);

if (isempty(find(abs(Periods-T) < 0.0001, 1)))
    T_low = max(Periods(Periods < T));
    T_high = min(Periods(Periods > T));
    
    ip_low =  abs(paraMedian_MNN5.Period-T_low)<0.0001;
    [medianRatio_low,stdLnRatio_low] = MNN5_Model_sub(ratio_SdeSdy,damp,paraMedian_MNN5(ip_low,:),paraSTD_MNN5(ip_low,:));

    ip_high =  abs(paraMedian_MNN5.Period-T_high)<0.0001;
    [medianRatio_high,stdLnRatio_high] = MNN5_Model_sub(ratio_SdeSdy,damp,paraMedian_MNN5(ip_high,:),paraSTD_MNN5(ip_high,:));

    x = [log(T_low), log(T_high)];
    Y_Ratio = [log(medianRatio_low), log(medianRatio_high)];
    Y_STD = [stdLnRatio_low, stdLnRatio_high];
    medianRatio = exp(Y_Ratio(:,1)+(log(T)-x(1))./(x(2)-x(1)).*(Y_Ratio(:,2)-Y_Ratio(:,1)));
    stdLnRatio = Y_STD(:,1)+(log(T)-x(1))./(x(2)-x(1)).*(Y_STD(:,2)-Y_STD(:,1));
else
    ip_T =  abs(paraMedian_MNN5.Period-T)<0.0001;
    [medianRatio,stdLnRatio] = MNN5_Model_sub(ratio_SdeSdy,damp,paraMedian_MNN5(ip_T,:),paraSTD_MNN5(ip_T,:));
end
medianRatio = reshape(medianRatio,n_row,n_col);
stdLnRatio = reshape(stdLnRatio,n_row,n_col);

%% sub-function
function [medianRatio,stdLnRatio] = MNN5_Model_sub(ratio_SdeSdy,damp,paraMedian,paraSTD)

[n_row,n_col] = size(ratio_SdeSdy);
n_data = n_row*n_col;
X1_temp = reshape(log(ratio_SdeSdy),n_data,1);
Rd_min = -2;
Rd_max = paraMedian.Rd_max(1);

% NaN is specified for inputs outside their ranges
X1_temp(X1_temp<Rd_min | X1_temp>Rd_max) = nan;

%% median
n = 6;
q = [paraMedian.q1,paraMedian.q2,paraMedian.q3,paraMedian.q4,paraMedian.q5,paraMedian.q6];
u = [paraMedian.u1,paraMedian.u2,paraMedian.u3,paraMedian.u4,paraMedian.u5,paraMedian.u6];
d = [paraMedian.d1,paraMedian.d2,paraMedian.d3,paraMedian.d4,paraMedian.d5,paraMedian.d6];
d0 = paraMedian.d0;

% damping ratio of 2.5% 
Y1 = d0(1);
for i = 1:n
    Y1 = Y1+u(1,i)./( 1+exp(-d(1,i)-q(1,i)*(X1_temp-Rd_min)/(Rd_max-Rd_min)) );
end
% damping ratio of 5% 
Y2 = d0(2);
for i = 1:n
    Y2 = Y2+u(2,i)./( 1+exp(-d(2,i)-q(2,i)*(X1_temp-Rd_min)/(Rd_max-Rd_min)) );
end
Y = Y1+(Y2-Y1)./(log(0.05)-log(0.025)).*(log(damp)-log(0.025));
medianRatio = exp(Y);

%% standard deviation
n = 6;
q = [paraSTD.q1,paraSTD.q2,paraSTD.q3,paraSTD.q4,paraSTD.q5,paraSTD.q6];
u = [paraSTD.u1,paraSTD.u2,paraSTD.u3,paraSTD.u4,paraSTD.u5,paraSTD.u6];
d = [paraSTD.d1,paraSTD.d2,paraSTD.d3,paraSTD.d4,paraSTD.d5,paraSTD.d6];
d0 = paraSTD.d0;

% damping ratio of 2.5% 
Y1 = d0(1);
for i = 1:n
    Y1 = Y1+u(1,i)./( 1+exp(-d(1,i)-q(1,i)*(X1_temp-Rd_min)/(Rd_max-Rd_min)) );
end
% damping ratio of 5% 
Y2 = d0(2);
for i = 1:n
    Y2 = Y2+u(2,i)./( 1+exp(-d(2,i)-q(2,i)*(X1_temp-Rd_min)/(Rd_max-Rd_min)) );
end
Y = Y1+(Y2-Y1)./(log(0.05)-log(0.025)).*(log(damp)-log(0.025));
stdLnRatio = Y;

