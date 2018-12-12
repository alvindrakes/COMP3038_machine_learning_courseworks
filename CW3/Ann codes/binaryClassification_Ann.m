load('facialPoints.mat');
load('labels.mat');

input = reshape(points, [66*2,150]);
target = labels'; 

k = 10; % no. of folds
k_net = cell(1, 10); % cell of all nets

[a, b] = size(target);

for i=1:k
    
    [training_input, training_target, testing_input, testing_target] = doKsplit(input, target, randperm(b), i)
    
    % network creation
    net = newff(inTrain, trainTarget, [15] , '','trainlm', 'learngd');
    net.trainParam.epochs=100;
    
    [k_net{i}, tr] = train(net, inTrain, trainTarget);
    
    % obtain the values
    predicted_result = predict(SVM, testing_input);
    cm = conf_mat(predicted_result, testing_target);
   [recall(m), precision(m), f1m(m) ] = calculate_result(cm);
    
end

f1 = mean(f1m);



% calculate the confusion matrix
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

% k-fold split 
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

% obtain recall, precision, f1 value 
function [recall, precision, f1m ] = calculate_result(cm)
        recall = cm(1,1)/(cm(1,1)+cm(1,2)); % tp/tp+fn
        precision = cm(1,1)/(cm(1,1)+cm(2,1)); % tp/tp+fp
        f1m = 2*((precision*recall)/(precision+recall));
end


    
    
    

