 function [modelparameter] = ParaInit(name)%optparameter,

    modelparameter.alpha = 1;% The regularization parameter
    modelparameter.beta = 0.7;% Balance parameter of noisy learning part and reconstruction part
    modelparameter.gamma = 0.05;% Balance parameter of label thresholding term and label scoring term   
    modelparameter.k = 10; % The number of neighbours 
    
    modelparameter.NumHiddenNeuron = 20;
    modelparameter.useELM = 1; % Use ELM to transform or just use original features
    modelparameter.Type = 'sig'; % Use sigmoid as the activation function
    
    modelparameter.BlockSize = 500; % The data chunk size at online learning stage
    modelparameter.num = modelparameter.BlockSize; % The intialization chunk size 

    modelparameter.A = 0.2;
    modelparameter.B = 0.4;
    modelparameter.sep = 0;
    modelparameter.A1 = 0;
    modelparameter.B1 = 0;
    modelparameter.A2 = 0;
    modelparameter.B2 = 0;
   
    if nargin > 0
        switch name
            case 'arts'
                modelparameter.beta = 0.5;modelparameter.gamma = 2^-4;        
            case 'recreation'
                modelparameter.beta = 0.5;modelparameter.gamma = 2^-4;
            case 'enron' 
                modelparameter.beta = 0.5;modelparameter.gamma = 2^-7;
            case 'medical' 
                modelparameter.beta = 0.55;modelparameter.gamma = 2^-7;         
            case 'social' 
                modelparameter.beta = 0.55;modelparameter.gamma = 2^-6;
            case 'rcv1subset5_top944' 
                modelparameter.beta = 0.5;modelparameter.gamma = 2^-7;                   
            case 'mediamill'
                modelparameter.beta = 1;modelparameter.gamma = 2^-8; 
            case 'languagelog'
                modelparameter.beta = 0.5;modelparameter.gamma = 2^-5; 
            case 'health'
                modelparameter.beta = 0.55;modelparameter.gamma = 2^-5;    
            case 'education'
                modelparameter.beta = 0.65;modelparameter.gamma = 2^-6;
            case 'corel16k001'
                modelparameter.beta = 0.55;modelparameter.gamma = 2^-6;
            case 'core15k'
                modelparameter.beta = 0.55;modelparameter.gamma = 2^-8;
            case 'bibtex'
                modelparameter.beta = 0.65;modelparameter.gamma = 2^-6;
            case 'tmc2007'
                modelparameter.beta = 0.75;modelparameter.gamma = 2^-4;         
            case 'imdb' 
                modelparameter.beta = 0.7;modelparameter.gamma = 0.05; 
            case 'bookmarks' 
                modelparameter.beta = 0.7;modelparameter.gamma = 0.05; 
            case 'eurlex-sm' 
                modelparameter.beta = 0.7;modelparameter.gamma = 0.05;       
        end
    end
end




