load('facialPoints.mat');
load('labels.mat');

input = reshape(points, [66*2,150]);
target = labels'; 

% K-fold cross validation
k_part = cvpartition(length(input), 'KFold', 10);

perf = zeros(1, k_part.NumTestSets); % performance array for each net
k_net = cell(1, 10); % cell of all nets

for i=1:k_part.NumTestSets
    % cvpartition index
    iTrain = training(k_part, i);
    iTest = test(k_part, i);
    
    % Train data
    inTrain = input(:, iTrain);
    trainTarget = target(:, iTrain);
    
    % network creation
    net = newff(inTrain, trainTarget, [15] , '','trainlm', 'learngd');
    net.trainParam.epochs=100;
    
    [k_net{i}, tr] = train(net, inTrain, trainTarget);
    
    % Test data
    inTest = input(:, iTest);
    testTarget = target(:, iTest);
    
    % net simulation
    output = k_net{i}(inTest);
    outRound = round(output);
    perf(i) = perform(k_net{i}, testTarget, output);
    
    differences = outRound - testTarget;
    abs_differences = abs(differences); % absolute value of the difference
    
    % accuracy of each network
    accuracies(:,i) = (1-(sum(abs_differences)/k_part.TestSize(i)))*100;
    
end

% Average accuracy
average_accuracy = mean(accuracies);


    
    
    

