load('facialPoints.mat');
load('labels.mat');

X1 = reshape(points, [132 150])';
Y1 = labels;

%Find the sample size
[a,b] = size(Y1);

k =10; % no of folds

%Parameter values for inner cross validation
C_value = [0.05, 0.5, 1.0, 5.0, 10.0, 50.0,100.0, 250.0];
poly_order = [2,3,4,5];

iteration = 1;
bestF1 = 0;

%Inner cross validation to find best hyperparameter
for i = 1 : length(C_value)
    for j =1: length(poly_order)
        recall = zeros(1, 10);
        precision = zeros(1, 10);
        f1m = zeros(1, 10);
        
        %Loop for 10 folds to find the average F1 value
        for m =1:k
            
            %get training and testing input and target
            [training_input, training_target, testing_input, testing_target] = doKsplit(X1,Y1,randperm(a),m);
            
            SVM = fitcsvm(training_input,training_target, 'KernelFunction','polynomial','BoxConstraint',C_value(i),...
            'PolynomialOrder',poly_order(j),'Standardize', true);
            
            predicted_result = predict(SVM, testing_input);
            cm = conf_mat(predicted_result, testing_target);
            [recall(m), precision(m), f1m(m) ] = calculate_result(cm);
        end
        
        f1 = mean(f1m);
        
        % Store the hyperparameters in structure array
        hyper_data.C_value = C_value(i);
        hyper_data.poly_order = poly_order(j);
        hyper_data.SupportVector = sum(SVM.IsSupportVector);
        hyper_data.SV_percentage = hyper_data.SupportVector/size(training_input,1);
        hyper_data.f1 = f1;
        hyperparameter(iteration) = hyper_data;
        iteration = iteration +1;
    end
end

%Look for best hyperparameter
for i = 1:length(hyperparameter)
    
    currentF1 = hyperparameter(i).f1;
    
    if currentF1 > bestF1 && ~isnan(currentF1)
         bestF1 = currentF1;
         bestHP = hyperparameter(i);
    end
end

recall = zeros(1, 10);
precision = zeros(1, 10);
f1m = zeros(1, 10);
SV = zeros(1,10);
percentage_SV = zeros(1,10);

%Do K Fold and find comparison results using optimised hyperparameter
for i =1:k
    [training_input, training_target, testing_input, testing_target] = doKsplit(X1,Y1,randperm(a),i);
    
    SVM = fitcsvm(training_input,training_target, 'KernelFunction','polynomial','BoxConstraint',bestHP.C_value,...
            'PolynomialOrder',bestHP.poly_order,'Standardize', true);
    
    predicted_result = predict(SVM, testing_input);
    cm = conf_mat(predicted_result, testing_target);
    [recall(i), precision(i), f1m(i) ] = calculate_result(cm);
    %Store no of support vectors selected
    SV(i) = size(SVM.SupportVectors,1);
    percentage_SV(i) = size(SVM.SupportVectors,1)./size(training_input,1);
end

function cm = conf_mat(outputs, targets)
% to count number of TP, FP, TN, FN
    tp=0; tn=0; fp=0; fn= 0; 
    
    for i=1:length(outputs)
        if (outputs(i)==1) && (targets(i)==1)
            tp = tp+1;
        elseif (outputs(i)==1) && (targets(i)==0)
            fp = fp+1;
        elseif (outputs(i)==0) && (targets(i)==0)
            tn = tn+1; 
        elseif (outputs(i)==0) && (targets(i)==1)
            fn = fn+1; 
        end
    end
    
    cm = [tp, fn; fp, tn];
end

%Function to do split into K fold
function [training_input, training_target, testing_input, testing_target] = doKsplit(input, target, rand, current_fold)
    
    lower_bound = 1+15*(current_fold-1);
    upper_bound = 15*current_fold;
    
    input = input(rand,:);
    target = target(rand,:);
    testing_input = input((lower_bound : upper_bound),:);
    testing_target = target((lower_bound : upper_bound),:);
    training_input = input([1:lower_bound-1 upper_bound+1:end],:);
    training_target = target([1:lower_bound-1 upper_bound+1:end],:);

end

%Function to calculate result for recall, precision and f1
function [recall, precision, f1m ] = calculate_result(cm)
        recall = cm(1,1)/(cm(1,1)+cm(1,2)); % tp/tp+fn
        precision = cm(1,1)/(cm(1,1)+cm(2,1)); % tp/tp+fp
        f1m = 2*((precision*recall)/(precision+recall));
end


