load('facialPoints.mat');
load('headpose.mat');

X2 = reshape(points, [132 8955])';
Y2 = pose(:,6);

%Find the sample size
[a,b] = size(Y2);

iteration = 1;
k = 10;
C_value = [0.5, 1.0, 5.0, 10.0, 50.0];
KernelScale =[0.25, 0.5, 2.0, 5.0, 10.0]; 
epsilon = [0.5, 1.0, 3.0, 5.0, 10.0];

for i = 1:length(C_value)
    for j =1:length(KernelScale)
        for l =1:length(epsilon)
            
            rms_errors = zeros(1, 10);

            %Loop for 10 folds to find the average F1 value
            for m =1:k

                %get training and testing input and target
                [training_input, training_target, testing_input, testing_target] = doKsplit(X2,Y2,randperm(a),m);

                SVM = fitrsvm(training_input, training_target,'KernelFunction', 'rbf', 'KernelScale', KernelScale(j),...
                'BoxConstraint', C_value(i), 'Epsilon', epsilon(l),'Standardize', true );

                predicted_result = predict(SVM, testing_input);
                rms =(1/(2*length(testing_target)))*sum(power((predicted_result - testing_target),2));
                rms_errors(m)=rms;
            end

            RMS = mean(rms_errors);

            % Store the hyperparameters in structure array
            hyper_data.C_value = C_value(i);
            hyper_data.KernelScale = KernelScale(j);
            hyper_data.epsilon = epsilon(l);
            hyper_data.SupportVector = sum(SVM.IsSupportVector);
            hyper_data.SV_percentage = hyper_data.SupportVector/size(training_input,1);
            hyper_data.RMS = RMS;
            hyperparameter(iteration) = hyper_data;
            iteration = iteration +1;
        end
    end
end

bestRMS = hyperparameter(1).RMS;
%Look for best hyperparameter
for i = 1:length(hyperparameter)
    
    currentRMS = hyperparameter(i).RMS;
    
    if currentRMS < bestRMS 
         bestRMS = currentRMS;
         bestHP = hyperparameter(i);
    end
end

rms_errors = zeros(1, 10);
SV = zeros(1,10);
percentage_SV = zeros(1,10);

%Do K Fold and find comparison results using optimised hyperparameter
for i =1:k
    [training_input, training_target, testing_input, testing_target] = doKsplit(X2,Y2,randperm(a),i);
    
    SVM = fitrsvm(training_input, training_target,'KernelFunction', 'rbf', 'KernelScale', bestHP.KernelScale,...
                'BoxConstraint', bestHP.C_value, 'Epsilon', bestHP.epsilon,'Standardize', true );

    
    predicted_result = predict(SVM, testing_input);
    rms =(1/(2*length(testing_target)))*sum(power((predicted_result - testing_target),2));
    rms_errors(i)=rms;
    %Store no of support vectors selected
    SV(i) = size(SVM.SupportVectors,1);
    percentage_SV(i) = size(SVM.SupportVectors,1)./size(training_input,1);
end

%Function to do split into K fold
function [training_input, training_target, testing_input, testing_target] = doKsplit(input, target, rand, current_fold)
    
    %One fold is 896 samples while last fold is 891 samples
    lower_bound = 1+896*(current_fold-1);
    if current_fold == 10
        upper_bound =8955;
    else
        upper_bound = 896*current_fold;
    end
    
    input = input(rand,:);
    target = target(rand,:);
    testing_input = input((lower_bound : upper_bound),:);
    testing_target = target((lower_bound : upper_bound),:);
    training_input = input([1:lower_bound-1 upper_bound+1:end],:);
    training_target = target([1:lower_bound-1 upper_bound+1:end],:);

end
