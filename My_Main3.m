%与传统方法对比
% Step1: reading Data from the file
load data.mat;
for i=1:length(data)
    if((data(i,7)>=0.02||data(i,8)>=0.02||data(i,9)>=0.015||data(i,10)>=0.1||data(i,11)>=0.002))%Pb Cd As Cr Hg
        data(i,12)=1;
    else
        data(i,12)=0;
    end
end
soil=[];
for i=1:length(data)
    if((data(i,1)>=35||data(i,2)>=0.2||data(i,3)>=15||data(i,4)>=90||data(i,5)>=0.15))%土壤Pb Cd As Cr Hg
        soil(i,1)=1;
    else
        soil(i,1)=0;
    end
end
Data = data(:,1:end-6)';
Labels = data(:, end)';%最后一列为真实分类结果
Labels = Labels*2 - 1;

MaxIter = 100; % boosting iterations

% Step2: splitting data to training and control set
TrainData   = Data(:,1:2:end);
TrainLabels = Labels(1:2:end);

ControlData   = Data(:,2:2:end);
ControlLabels = Labels(2:2:end);

% and initializing matrices for storing step error
RAB_control_error = zeros(1, MaxIter);
MAB_control_error = zeros(1, MaxIter);
GAB_control_error = zeros(1, MaxIter);
T_control_error = zeros(1, MaxIter);

% Step3: constructing weak learner
weak_learner = tree_node_w(3); % pass the number of tree splits to the constructor

% and initializing learners and weights matices
GLearners = [];
GWeights = [];
RLearners = [];
RWeights = [];
NuLearners = [];
NuWeights = [];

% Step4: iterativly running the training

for lrn_num = 1 : MaxIter

    clc;
    disp(strcat('Boosting step: ', num2str(lrn_num),'/', num2str(MaxIter)));

    %training gentle adaboost
    [GLearners GWeights] = GentleAdaBoost(weak_learner, TrainData, TrainLabels, 1, GWeights, GLearners);

    %evaluating control error
    GControl = sign(Classify(GLearners, GWeights, ControlData));

    GAB_control_error(lrn_num) = GAB_control_error(lrn_num) + sum(GControl ~= ControlLabels) / length(ControlLabels);

    %training real adaboost
    [RLearners RWeights] = RealAdaBoost(weak_learner, TrainData, TrainLabels, 1, RWeights, RLearners);

    %evaluating control error
    RControl = sign(Classify(RLearners, RWeights, ControlData));

    RAB_control_error(lrn_num) = RAB_control_error(lrn_num) + sum(RControl ~= ControlLabels) / length(ControlLabels);

    %training modest adaboost
    [NuLearners NuWeights] = ModestAdaBoost(weak_learner, TrainData, TrainLabels, 1, NuWeights, NuLearners);

    %evaluating control error
    NuControl = sign(Classify(NuLearners, NuWeights, ControlData));

    MAB_control_error(lrn_num) = MAB_control_error(lrn_num) + sum(NuControl ~= ControlLabels) / length(ControlLabels);
    
    T_control_error(lrn_num) = T_control_error(lrn_num) + sum(soil(2:2:end,1) ~= data(2:2:end,12)) / length(soil(2:2:end,1));

end

% Step4: displaying graphs
figure, plot(GAB_control_error);
hold on;
plot(MAB_control_error, 'r');

plot(RAB_control_error, 'g');

plot(T_control_error, 'k');
hold off;

legend('Gentle AdaBoost', 'Modest AdaBoost', 'Real AdaBoost','Non-AdaBoost');
xlabel('Iterations');
ylabel('Test Error');