load('facialPoints.mat');
load('headpose.mat');
labels = pose(:,6);
target = labels';
input = reshape(points, [66*2,8955]);

% number of folds
k = 10;
numNeurons = 20;

c = cvpartition(length(input),'KFold', k); %split the data into 10 folds
perf = zeros(c.NumTestSets,1);

%create the net
net = newff(input, target, numNeurons, '','trainlm', 'learngd');
net.trainParam.epochs = 1000;


%create a cell of 10 neural nets
numNN = 10;
nets = cell(1,numNN);

RMS_errors = zeros(1,k);
RMSE_array = [];

for i=1:c.NumTestSets
    % Obtain indexes for the train/test split
    train_index = training(c,i);
    test_index = test(c,i);
    
    % Obtain training inputs and associated labels
    trainingInputs = input(:,train_index);
    trainingTargets = target(:,train_index);
    
    % Obtain test inputs and associated labels
    testingInputs = input(:,test_index);
    testingTargets = target(:,test_index);
    
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

