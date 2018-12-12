clear;
%ANN
load('ANN regression.mat')
ANNPrediction = RMS_errors;

%Regression linear
load('RL.mat')
SVMLinear = rms_errors;

%Regression rbf
load('RG.mat')
SVMRBF = rms_errors;

%Regression polynomial
load('RP.mat')
SVMPolynomial = rms_errors;

H_ttest = zeros(1,3);
P_ttest = zeros(1,3);

[H_ttest(1), P_ttest(1)] = ttest2(ANNPrediction,SVMLinear);
[H_ttest(2), P_ttest(2)] = ttest2(ANNPrediction,SVMRBF);
[H_ttest(3), P_ttest(3)] = ttest2(ANNPrediction,SVMPolynomial);