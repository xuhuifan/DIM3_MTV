clear all;
% %% Dynamic Infinite Mixed-Membership Relational Model Sampling
% %% initialization

load('enrondata.mat');
% Niteration = 100000;
Niteration = 5;
dim3 = dim3Ini(datas,numClass, dataNum, tTime);
dim3.datas = datas;
dim3.kappa = [0 0.01];
dim3.indexLabel = 1:numClass;
dim3.gamma = 0.3;
dim3.alpha = 1;

%% Gibbs sampling loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ite_numc = zeros(1, Niteration);
deviance_numc = zeros(1, Niteration);

st_like = -inf*ones(1, 5);
like_seL = dim3.seLabel;
like_reL = dim3.reLabel;
selec_like = zeros(1, 5);
cu_like = zeros(1, Niteration);
st_dims = cell(1, 5);
tic;
for n_ite = 1:Niteration
    % sampling \beta value
    %    dim3.m_val = dim3.m_val/(sum(dim3.m_val)/numClass);
    dim3.betas = dirrnd([dim3.m_val dim3.gamma], 1);
    % sampling se_Labels re_Labels value
    %   dim3=sere_sample_nolik(dim3);
    dim3=label_gibbs(dim3);
    % sampling m value
    dim3 = m_stick(dim3);
    [dim3.alpha, dim3.gamma, dim3.kappa(2)]= Hyperpara( dim3.nohat_m, dim3.m_val, dim3.numClass, dim3.dataNum,  dim3.tTime, dim3.alpha, dim3.gamma, dim3.kappa(2));

    [dim3.deviance, cu_jp, cu_likes]= gibbs_dev(dim3);

    cu_like(n_ite) = cu_likes;
    
    spe_k = ceil(n_ite/Niteration*5);

    if cu_likes > st_like(spe_k)
        st_like(spe_k) = cu_likes;
        like_seL = dim3.seLabel;
        like_reL = dim3.reLabel;
        selec_like(spe_k) = n_ite;
        st_dims{spe_k} = dim3;
    end
    
    deviance_numc(n_ite) = dim3.deviance;
    ite_numc(n_ite) = dim3.numClass;
    if mod(n_ite, 100)==0
        toc;
        fprintf('the iteration time is %d\n', n_ite);
        fprintf('num of class is %d\n', dim3.numClass);
        tic;
    end
end




