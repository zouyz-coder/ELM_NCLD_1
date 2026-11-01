%% Load the data
clear;clc;
restoredefaultpath;
userpath('clear');
currentFolder = pwd; 
addpath(genpath(currentFolder));

name = 'arts';
load([name '.mat']);
if exist('train_data','var')==1
    X=[train_data;test_data]';
    G=[train_target,test_target];
    clear train_data test_data train_target test_target
end
if exist('dataset','var')==1
    X=dataset';
    G=class;
    clear dataset class
end  
if exist('data','var')==1
    X=data';
    G=target;
    clear data target
end  
G(G==0)=-1;

%% Normalize the data 
X     = double(X);
temp_data = X + eps;
num_data = size(X,2);

temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,1)),size(temp_data,1),1);
if sum(sum(isnan(temp_data)))>0
    temp_data = temp_data+eps;
    temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,1)),size(temp_data,1),1);
end
X = temp_data;

para = ParaInit(name);
num = para.num;

%% Inject noisy labels randomly
[y_noisy, ~, Rho] = CCN_Gen5(G',para);%
Y = y_noisy;[q,T] = size(Y);

%% Run ELM_NCLD
X0 = X(:,1:num);Y0 = Y(:,1:num);G = G(:,(num+1):end);
ResultAll = ELM_NCLD(X0',Y0',X',Y',G',Rho,para);



