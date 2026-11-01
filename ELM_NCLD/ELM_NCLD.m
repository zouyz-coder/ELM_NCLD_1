function [ResultAll,time,W,yt_hat,Outputs,BlockNum] = ELM_NCLD(X0,Y0,X,Y,G,Rho,para)
%X0:num*d Y0: num*q
%X:(T-num)*d Y:(T-num)*q with noisy labels G:(T-num)*q with ground-truth labels
%Rho:The noisy rates  para:The model parameters
%% preTrain
start = tic;
rng(0);
alpha = para.alpha;
beta = para.beta;
%gamma = para.gamma;
BlockSize = para.BlockSize;
k = para.k;
 
[num,d] = size(X0);
X = X(num+1:end,:);Y = Y(num+1:end,:);
if para.useELM
    IW = rand(para.NumHiddenNeuron,d)*2-1;
    switch lower(para.Type)
        case{'rbf'}
            Bias = rand(para.NumHiddenNeuron,1);
            TH = RBFun([X0;X]',IW,Bias);
        case{'sig'}
            Bias =rand(para.NumHiddenNeuron,1)*2-1;
            TH = SigActFun([X0;X]',IW,Bias);
        case{'sin'}
            Bias =rand(para.NumHiddenNeuron,1)*2-1;
            TH = SinActFun([X0;X]',IW,Bias);
        case{'hardlim'}
            Bias = rand(para.NumHiddenNeuron,1)*2-1;
            TH = HardlimActFun([X0;X]',IW,Bias);
            TH = double(TH);
    end
else
    TH = [X0',X'];% not use ELM
end
 
H0 = TH(:,1:num)';H = TH(:,num+1:end)';
mdl = (H0'*H0+alpha*eye(size(H0,2)))\(H0'*Y0);
 
R = calcS(X0,k,beta);
if para.useELM
    M0 = pinv(alpha*eye(para.NumHiddenNeuron)+H0'*R*H0);% use ELM
else
    M0 = pinv(alpha*eye(d)+H0'*R*H0);% not use ELM
end
 
M = M0;
[~,~,O_0] = calcO(H0,Y0,Rho,mdl,para);
W = M0*(H0'*O_0);
thr = 0;
 
i = 1;
dataBlock = [];
T = size(X,1);
while i<=T
    if i + 2*BlockSize - 1 > T
        dataBlock = [dataBlock [i;T]];
        break;
    end
    dataBlock = [dataBlock [i;i+BlockSize-1]];
    i = i + BlockSize;
end
 
BlockNum = size(dataBlock,2);
r = 0;
                             
%% start online learning
for i = 1:BlockNum
    Hi = H(dataBlock(1,i):dataBlock(2,i),:);
    Xi = X(dataBlock(1,i):dataBlock(2,i),:);
    Yi = Y(dataBlock(1,i):dataBlock(2,i),:);
    Ni = size(Hi,1);
 
    Outputs(r+1:r+Ni,:) = Hi*W;
    yt_hat(r+1:r+Ni,:) = sign(Outputs(r+1:r+Ni,:)-thr-eps);
    
    mdl = (Hi'*Hi+para.alpha*eye(size(Hi,2)))\(Hi'*Yi);
    [~,~,Oi] = calcO(Hi,Yi,Rho,mdl,para);
    r = r + Ni; 
    [R,C] = calcS(Xi,k,beta);
    
    M = M - M *Hi'*pinv(C+Hi*M*Hi')*Hi*M;   
    W = W - M * (Hi' *R * Hi * W - Hi'*Oi);
   
end
    ResultAll = EvaluationAll(yt_hat',Outputs',G');   
    time = toc(start);
end
 
function [Yi,Ci,Oi] = calcO(Hi,Yi,Rho,mdl,para)
    beta = para.beta;gamma = para.gamma;
    [n,q] = size(Yi);
    Yi(Yi==-1)=0;
 
    Ci = zeros(n,q);
 
    for i = 1:n
        for j = 1:q
            for k = j+1:q
                if Yi(i,j)==1&&Yi(i,k)==0
                    p_j_hat = sigmoid(Hi(i,:)*mdl(:,j));                                                                           
                    p_k_hat = 1 - sigmoid(Hi(i,:)*mdl(:,k));     
                    weight_j = max(0,1-Rho(j,2)/p_j_hat)/(1-Rho(j,1)-Rho(j,2));
                    weight_k = max(0,1-Rho(k,1)/p_k_hat)/(1-Rho(k,1)-Rho(k,2));
                    c = weight_j*weight_k;
                    Ci(i,j) =  Ci(i,j) - c;
                    Ci(i,k) =  Ci(i,k) + c;
                elseif Yi(i,j)==0&&Yi(i,k)==1
                    p_j_hat = 1 - sigmoid(Hi(i,:)*mdl(:,j));                                                                           
                    p_k_hat = sigmoid(Hi(i,:)*mdl(:,k));     
                    weight_j = max(0,1-Rho(j,1)/p_j_hat)/(1-Rho(j,1)-Rho(j,2));
                    weight_k = max(0,1-Rho(k,2)/p_k_hat)/(1-Rho(k,1)-Rho(k,2));
                    c = weight_j*weight_k;
                    Ci(i,k) =  Ci(i,k) - c;
                    Ci(i,j) =  Ci(i,j) + c;
                end
            end
        end
    end
    
    Yi(Yi==0)=-1;Yi = beta*Yi;   
    Oi = Yi - gamma*Ci;
end
 
function [R,C,S] = calcS(Xi,k,beta)
    [dist,neighbor] = pdist2(Xi,Xi,'euclidean','Smallest',k+1);
    n = size(Xi,1);
    if k > n - 1
        S = eye(n);
        return;
    end
    dist=dist';
    neighbor=neighbor';
    neighbor = neighbor(:,2:k+1);
    dist = dist(:,2:k+1);
 
    rows = repmat((1:n)',1,k);
    datas = zeros(n,k);
    for t=1:n
        neighborIns = Xi(neighbor(t,:),:)';
        w = lsqnonneg(neighborIns,Xi(t,:)');
        datas(t,:) = w;
    end
 
    trans = sparse(rows,neighbor,datas,n,n);
    sumW = full(sum(trans,2));
    sumW(sumW==0)=1;
    S = bsxfun(@rdivide,trans,sumW);
    S = sparse(S);
    
    R = beta*eye(n)+(1-beta)*(-S-S'+S'*S);
    C = pinv(R);
end
 
function p = sigmoid(a)
    for i = 1:size(a,1)
        for j = 1:size(a,2)
            p(i,j) =1/(1+exp(-a(i,j)));
        end
    end
end

