% Step1: reading Data from the file
load data.mat;
for i=1:length(data)
    if((data(i,7)>=0.02||data(i,8)>=0.02||data(i,9)>=0.015||data(i,10)>=0.1||data(i,11)>=0.002))%Pb Cd As Cr Hg
        data(i,12)=1;
    else
        data(i,12)=0;
    end
end
%data(find(data(:,7)>=0.02||data(:,8)>=0.02||data(:,9)>=0.015||data(:,10)>=0.1||data(:,11)>=0.002),12)=1;%cd 是否超标放到最后一列

Data = data(:,1:end-6)';
Labels = data(:, end)';%最后一列为真实分类结果
Labels = Labels*2 - 1;

MaxIter = 100; % boosting iterations

% Step2: splitting data to training and control set
TrainData   = Data(:,1:2:end);
TrainLabels = Labels(1:2:end);

ControlData   = Data(:,2:2:end);
ControlLabels = Labels(2:2:end);

% Step3: constructing weak learner
weak_learner = tree_node_w(3); % pass the number of tree splits to the constructor

% Step4: training with Gentle AdaBoost
[RLearners RWeights] = RealAdaBoost(weak_learner, TrainData, TrainLabels, MaxIter);

% Step5: training with Modest AdaBoost
[MLearners MWeights] = ModestAdaBoost(weak_learner, TrainData, TrainLabels, MaxIter);

% Step6: evaluating on control set
ResultR = sign(Classify(RLearners, RWeights, ControlData));

ResultM = sign(Classify(MLearners, MWeights, ControlData));

% Step7: calculating error
ErrorR  = sum(ControlLabels ~= ResultR) / length(ControlLabels)

ErrorM  = sum(ControlLabels ~= ResultM) / length(ControlLabels)