load('C:\Users\hp\Desktop\Coursework\Machine Learning\data\assessment\binary smile\facialPoints.mat');
load('C:\Users\hp\Desktop\Coursework\Machine Learning\data\assessment\binary smile\labels.mat');

inputs = reshape(points, [132 150])';
targets = labels;

% Do K-Fold
[k_tree, outputs, accuracies, recall, precision, f1m, cm] = k_fold(inputs, targets);


% Function to do K-fold cross validation
function [k_tree, outputs, accuracies, recall, precision, f1m, cm] = k_fold(inputs, targets)
    
    k_tree = cell(1, 10); % cell of all trees
    outputs = zeros(1, 15);
    accuracies = zeros(1, 10);
    recall = zeros(1, 10);
    precision = zeros(1, 10);
    f1m = zeros(1, 10);
    k =10;
    [a,b] = size(targets);
    
    for i=1:k
        
        [inTrain, trainTarget, inTest, testTarget] = doKsplit(inputs, targets, randperm(a), i);

        % Training 10 trees
        k_tree{i} = decisionTreeLearning(inTrain, trainTarget);
       
        % Loop through all in test fold
        for j = 1:15
            outputs(j) = outp_Tree(k_tree{i}, inTest(j,:)); 
        end
        
        abs_differences = abs(outputs' - testTarget);
        accuracies(i) = 1 - sum(abs_differences)/15;
        
        cm = conf_mat(outputs, testTarget);
        recall(i) = cm(1,1)/(cm(1,1)+cm(1,2)); % tp/tp+fn
        precision(i) = cm(1,1)/(cm(1,1)+cm(2,1)); % tp/tp+fp
        
        f1m(i) = 2*((precision(i)*recall(i))/(precision(i)+recall(i)));
        
    end
end

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

%Function to output a tree based on the inputs and targets
function tree = decisionTreeLearning(inputs, targets)

    %If the node consist of all the labels from the same class
    % return as leaf node
    if sum(targets)==size(targets,1) || sum(targets) == 0
        tree.op = '';
        tree.kids = [];
        tree.threshold = '';
        tree.class = get_majority_value(targets);

    % else choose the best attribute and threshold and do the splitting
    else
        [best_feature, best_threshold, p ,n] = choose_Attribute(inputs, targets);
        % p and n is the total number of positive and negative samples
        % splitted at the node
        tree.op = [best_feature, best_threshold];
        tree.kids = cell(1,2);
        tree.class = '';
        tree.attribute = best_feature;
        tree.threshold = best_threshold;

        %Get input index with threshold
        [leftTreeIndex, rightTreeIndex] = split_data(inputs, best_threshold, best_feature);
        %Get input values with index given
        leftTreeInputs = inputs(leftTreeIndex,:);
        leftTreeTargets = targets(leftTreeIndex);
        rightTreeInputs = inputs(rightTreeIndex,:);
        rightTreeTargets = targets(rightTreeIndex);

        %Create left subtree and recurse the same function with left inputs
        tree.kids{1,1} = decisionTreeLearning(leftTreeInputs, leftTreeTargets);
        % Right subtree and recurse the same function with right inputs
        tree.kids{1,2} = decisionTreeLearning(rightTreeInputs, rightTreeTargets);
    end
end

%Function that return left and right subtree index to split the inputs 
% using the best threshold and features selected
function [leftTreeIndex, rightTreeIndex] = split_data(inputs, threshold, best_feature)
    leftTreeIndex = [];
    rightTreeIndex = [];
    inputSize = size(inputs,1);

    for i = 1:inputSize
        if(inputs(i, best_feature) > threshold)
            leftTreeIndex = [leftTreeIndex, i];
        else
            rightTreeIndex = [rightTreeIndex, i];
        end
    end
end

%Choose the best feature and threshold to do the spliting of the tree
function [best_feature, best_threshold, num_pos, num_neg] = choose_Attribute(features, targets)

    [sampleSize, attributeSize] = size(features);
    
    [p, n] = calculate_ratio(targets);
    num_pos = p;
    num_neg = n;
    
    %Initialise variables
    threshold =0;
    best_Attribute = 0;
    best_Threshold = 0;
    bestGain = 0;
    entropy = calculate_Entropy(p,n);
    
    % By trying every features in data samples as threshold and
    % choose the attribute and threshold that gives the highest 
    % information gain
    for i = 1:attributeSize
        for j = 1:sampleSize
        
            threshold = features(j,i);
            leftChildIndex = [];
            rightChildIndex = [];
            
            %Split attribute index
            for y = 1:sampleSize
               
                if features(y,i) > threshold
                    leftChildIndex = [leftChildIndex,y];
                    
                else
                    rightChildIndex = [rightChildIndex,y];
                end
            end
            % Calculate total positive and negative samples at each leaf node
            [left_pos, left_neg] = calculate_ratio(getTargets(leftChildIndex, targets));
            [right_pos,right_neg] = calculate_ratio(getTargets(rightChildIndex, targets));
            
            % Calculate the remainder and information gain
            remainder = calculate_Remainder(left_pos, left_neg, right_pos, right_neg);
            info_gain = entropy - remainder;
            
            % Set current information gain to best gain if it is bigger
            % than previous information gain
            if info_gain > bestGain
                bestGain = info_gain;
                best_Attribute = i;
                best_Threshold = threshold;
            end
        end
    end
    
    best_feature = best_Attribute;
    best_threshold = best_Threshold;

end

% function to get the target values with the given index
function new_target = getTargets(index, targets)

    new_target = [];
    for i=1:length(index)
        new_target = [new_target, targets(index(i))];
    end
end

% Function to calculate the total positive and negative samples from the input
function [positive, negative] = calculate_ratio(target_input)
    positive = 0;
    negative = 0;
    for i=1:length(target_input)
        if target_input(i) == 0
            negative = negative +1;
            
        else
            positive = positive +1;
        end
    end
end

% FUnction to calculate entropy
function Entropy = calculate_Entropy(positive, negative)
    pos_probability = positive / (positive + negative);
    neg_probability = negative / (positive + negative);
    
    if (pos_probability == 0)
        entropy_pos = 0;
    else
        entropy_pos = -pos_probability*log2(pos_probability);
    end
    
    if (neg_probability == 0)
        entropy_neg = 0;
    else
        entropy_neg = -neg_probability*log2(neg_probability);
    end
    
    Entropy = entropy_pos + entropy_neg; 
end

% Function that return the majority value from the input
function value = get_majority_value(targets)
    
    p = 0;
    n = 0;
    for i=1:length(targets)
        if targets(i) == 1
            p = p + 1;
        else
            n = n + 1;
        end
    end
    
    if p > n
        value = 1;
    else
        value = 0;
    end       
end

%Function to calculate remainder to find information gain
function Remainder = calculate_Remainder(left_pos, left_neg, right_pos, right_neg)

    total = left_pos + left_neg + right_pos + right_neg;

    Remainder = (left_pos+ left_neg)*calculate_Entropy(left_pos, left_neg) / total +(right_pos+ right_neg)*calculate_Entropy(right_pos, right_neg)/total ;
end

function output = outp_Tree(tree, input)
% get output of tree with test set in k_fold

    if isempty(tree.kids) % if leaf node
        output = tree.class;
        return
    elseif input(tree.attribute) > tree.threshold
        output = outp_Tree(tree.kids{1}, input);
    else
        output = outp_Tree(tree.kids{2}, input);
    end
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

