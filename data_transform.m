% Step1: reading Data from the file
load data_all.mat;
for i=1:length(data)
    if((data(i,7)>=0.02||data(i,8)>=0.02||data(i,9)>=0.015||data(i,10)>=0.1||data(i,11)>=0.002))%Pb Cd As Cr Hg
        data(i,12)=1;
    else
        data(i,12)=0;
    end
end
%data(find(data(:,7)>=0.02||data(:,8)>=0.02||data(:,9)>=0.015||data(:,10)>=0.1||data(:,11)>=0.002),12)=1;%cd 是否超标放到最后一列

features = data(:,1:end-6)';
targets=data(:, end)';%最后一列为真实分类结果