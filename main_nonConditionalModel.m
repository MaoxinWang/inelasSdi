function main_nonConditionalModel

RotType = 'RotD50';  % RotD50 or RotD100
T = 0.3;  % period (in second) between 0.03 and 5 s
Cy = 0.05;  % yeild strength coefficient between 0.01 and 3
I_hys = 0;  % 0 for bilinear and 1 for Takeda
damping = 0.05;  % damping ratio between 0.025 and 0.05
Mag = 7;  % earthquake magnitude
Ztor = 0.135;  % depth to the rupture top (in km)
Frv = 0;  % indicator taken as 1 for reverse fault and 0 otherwise
Fnm = 0;  % indicator taken as 1 for normal fault and 0 otherwise
Rrup = 10;  % rupture distance (in km)
Vs30 = 350;  % shear-wave velocity (in m/s)
dZ25 = 0;  %  difference between measured and predicted depths to the 2.5 km/s shear-wave velocity (in km)

[medianSdi,sigma_total] = MNN4_Model(Cy,I_hys,damping,Mag,Ztor,Frv,Fnm,Rrup,Vs30,dZ25,T,RotType);

