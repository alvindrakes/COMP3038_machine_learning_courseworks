load('facialPoints.mat');
load('headpose.mat');
labels = pose(:,6);
target = labels';
input = reshape(points, [66*2,8955]);

%Find the sample size
[a,b] = size(target);

% number of folds
k = 10;
numNeurons = 20;

%create the net
net = newff(input, target, numNeurons, '','trainlm', 'learngd');
net.trainParam.epochs = 1000;


%create a cell of 10 neural nets
numNN = 10;
nets = cell(1,numNN);

RMS_errors = zeros(1,k);
RMSE_array = [];

for i=1:k 
   
    [trainingInputs, trainingTargets, testingInputs, testingTargets] = doKsplit(input, target, randperm(b), i);
    
    % Set up learning rate & Division of Data for Training, Validation, Testing 
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 20/100;
    net.divideParam.testRatio = 0/100;
    net.divideParam.lr = 0.01;
    
    % Create a neural network, train it on the training data for this fold
    [nets{i},tr] = train(net,trainingInputs,trainingTargets);
    
    fprintf('Training completed: %d/%d\n', i, k)
    
    % Test the networks performance on test data
    outputs = nets{i}(testingInputs);
    
    % Calculate the rms error between predictions and targets and store it
    % inside an array 
    rms =(1/(2*length(testingTargets)))*sum(power((outputs - testingTargets),2));
    RMS_errors(i) = rms  
    RMSE_array = [RMSE_array RMS_errors(i)];  
    
    % Get the lowest Rms Error from the array 
    bestRmsError = min(RMSE_array(:)); 
   
    % Calculate and store network performance
    performance = perform(nets{i},testingTargets,outputs);
end

fprintf("Lowest RMSE: %f", bestRmsError)


% own created k-fold function 
function [trainingInputs, trainingTargets, testingInputs, testingTargets] = doKsplit(input, target, rand, current_fold)
    
    %One fold is 896 samples while last fold is 891 samples
    lower_bound = 1+896*(current_fold-1);
    if current_fold == 10
        upper_bound =8955;
    else
        upper_bound = 896*current_fold;
    end
    
    
    input = input(:, rand);
    target = target(:, rand);
    testingInputs = input(:, (lower_bound : upper_bound));
    testingTargets = target(:, (lower_bound : upper_bound));
    trainingInputs = input(:, [1:lower_bound-1 upper_bound+1:end]);
    trainingTargets = target(:, [1:lower_bound-1 upper_bound+1:end]);

end

