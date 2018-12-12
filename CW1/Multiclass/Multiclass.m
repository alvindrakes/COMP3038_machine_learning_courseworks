
%Multiclass emotions
load('C:\Users\hp\Desktop\Coursework\Machine Learning\data\assessment\multiclass emotion\emotions_data.mat');
input = x';
target = y';
total_correct_classify = 0;

%change matrix form
new_target = [0;0;0;0;0;0];
for k = 1:612
    if (target(k) == 1)
        new_target(:,k) = [1; 0; 0; 0; 0; 0];
        
    elseif (target(k) == 2)
        new_target(:,k) = [0; 1; 0; 0; 0; 0];
        
    elseif (target(k) == 3)
        new_target(:,k) = [0; 0; 1; 0; 0; 0];

    elseif (target(k) == 4)
        new_target(:,k) = [0; 0; 0; 1; 0; 0];

    elseif (target(k) == 5)
        new_target(:,k) = [0; 0; 0; 0; 1; 0];
    else 
        new_target(:,k) = [0; 0; 0; 0; 0; 1];  
    end   
end

target = new_target;

%randomize the input and target
[a,b] = size(target);
Shuffle = randperm(b);
target = target(:,Shuffle);
input = input(:,Shuffle);

%Split data and store them into 3D matrix
input_fold = input(:,(1:61));
input_fold(:,:,2) = input(:,(62:122));
input_fold(:,:,3) = input(:,(123:183));
input_fold(:,:,4) = input(:,(184:244));
input_fold(:,:,5) = input(:,(245:305));
input_fold(:,:,6) = input(:,(306:366));
input_fold(:,:,7) = input(:,(367:427));
input_fold(:,:,8) = input(:,(428:488));
input_fold(:,:,9) = input(:,(489:549));
input_fold_last = input(:,(550:end));

target_fold = target(:,(1:61));
target_fold(:,:,2) = target(:,(62:122));
target_fold(:,:,3) = target(:,(123:183));
target_fold(:,:,4) = target(:,(184:244));
target_fold(:,:,5) = target(:,(245:305));
target_fold(:,:,6) = target(:,(306:366));
target_fold(:,:,7) = target(:,(367:427));
target_fold(:,:,8) = target(:,(428:488));
target_fold(:,:,9) = target(:,(489:549));
target_fold_last = target(:,(550:end));

%Create network
neuron =[20 10];
net = newff(input, target, neuron,'','traingdx'); 
net.trainParam.max_fail = 50;
net.trainParam.epochs=300;
net.layers{length(neuron)+1}.transferFcn = 'softmax';
net.performFcn = 'crossentropy';

%K-fold validation start here
for i = 1:10
    fold_correct_classify = 0;

    %lower and upper bound for the folds
    lower_bound = 1+61*(i-1);
    upper_bound = 61*i;
    
    %fold no. 10
    if i == 10
        test = input_fold_last;
        training = input(:,(1:lower_bound - 1));        %training data
        tr_target = target(:,(1:lower_bound-1));        %target data
        
    %fold no. 1 to 9
    else 
        test = input_fold(:,:,i);
        %training data
        training = input(:,[1:lower_bound-1 upper_bound+1:end]);
        %target data
        tr_target = target(:,[1:lower_bound-1 upper_bound+1:end]);
    end
    
    
    net2 = train(net, training, tr_target);             %train the network
    raw_predicted_output = net2(test);                  %simulation and get predicted output
    predicted_output= roundoff(raw_predicted_output);   %round the values so that it gets value 1 or 0
    
    %Get the whole predicted set for evaluation in the end
    if(i == 1)
        whole_predicted_set = predicted_output;
    else
        whole_predicted_set = horzcat(whole_predicted_set , predicted_output);
    end
    
    %Compute classification result
    if (i==10)
        output_data = target_fold_last;
    else
        output_data = target_fold(:,:,i);
    end
    
    [m, n] = size(predicted_output);
    
    %Compare predicted data and output data(from target data)
    for y = 1:n
        if(predicted_output(:,y) == output_data(:,y))
            fold_correct_classify = fold_correct_classify + 1;
            total_correct_classify = total_correct_classify + 1;
        end
    end
    
    %calulate the accuracy for this fold and store it in an array
    accuracy = (fold_correct_classify*100) / n; 
    accuracy_array(:,i) = accuracy;
end

% Calculate the overall accuracy for the whole network
Total_accuracy = (total_correct_classify *100) / 612;

%Plot confusion matrix
plotconfusion(target, whole_predicted_set);
