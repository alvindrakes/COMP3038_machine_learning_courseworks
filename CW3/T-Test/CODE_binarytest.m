clear;
%ANN
load('ANN binary.mat')
ANNPrediction = f1m;

%Decision Tree
load('DT.mat')
DTPrediction = f1m;

%Classification linear
load('CL.mat')
SVMLinear = f1m;

%Classification rbf
load('CG.mat')
SVMRBF = f1m;

%Classification polynomial
load('CP.mat')
SVMPolynomial = f1m;

H_ttest = zeros(1,7);
P_ttest = zeros(1,7);

[H_ttest(1), P_ttest(1)] = ttest2(ANNPrediction,SVMLinear);
[H_ttest(2), P_ttest(2)] = ttest2(ANNPrediction,SVMRBF);
[H_ttest(3), P_ttest(3)] = ttest2(ANNPrediction,SVMPolynomial);
[H_ttest(4), P_ttest(4)] = ttest2(ANNPrediction,DTPrediction);
[H_ttest(5), P_ttest(5)] = ttest2(DTPrediction,SVMLinear);
[H_ttest(6), P_ttest(6)] = ttest2(DTPrediction,SVMRBF);
[H_ttest(7), P_ttest(7)] = ttest2(DTPrediction,SVMPolynomial);
