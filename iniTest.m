clear all;
%% Dynamic Infinite Mixed-Membership Relational Model Sampling
Niteration = 20000;
%% data simulation
dataNum = 20;
tTime = 1;
numClass = 4;
[datas, ith_sela, ith_rela] = dataGens_1(dataNum, tTime);
% E = load('senator.mat');
% datas = E.E;
% [dataNum, ds, tTime] = size(datas);
%% initialization


dim3 = dim3Ini(numClass, dataNum, tTime);
dim3.datas = datas;
dim3.kappa = [0 0.03];
dim3.indexLabel = 1:numClass;
dim3.gamma = 0.6;
dim3.alpha = 1;
% whole_se = zeros(dataNum, dataNum, tTime, Niteration/2);
% whole_re = zeros(dataNum, dataNum, tTime, Niteration/2);
% whole_in = zeros(1, Niteration/2);

%% Gibbs sampling loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ite_numc = zeros(1, Niteration);
deviance_numc = zeros(1, Niteration);
st_jp = -inf;
st_seL = dim3.seLabel;
st_reL = dim3.reLabel;
selec_nite = 1;
cu_jps = zeros(1, Niteration);

st_like = -inf;
like_seL = dim3.seLabel;
like_reL = dim3.reLabel;
selec_like = 1;
cu_like = zeros(1, Niteration);
for n_ite = 1:Niteration
    % sampling \beta value
    %    dim3.m_val = dim3.m_val/(sum(dim3.m_val)/numClass);
    dim3.betas = dirrnd([dim3.m_val dim3.gamma], 1);
    % sampling se_Labels re_Labels value
    %   dim3=sere_sample_nolik(dim3);
    dim3=sere_sample(dim3);
    % sampling m value
    dim3 = m_stick(dim3);
    [dim3.deviance, cu_jp, cu_likes]= gibbs_dev(dim3);
    cu_jps(n_ite) = cu_jp;
    cu_like(n_ite) = cu_likes;
    if cu_jp > st_jp
        st_jp = cu_jp;
        st_seL = dim3.seLabel;
        st_reL = dim3.reLabel;
        selec_nite = n_ite;
    end
    if cu_likes > st_like
        st_like = cu_likes;
        like_seL = dim3.seLabel;
        like_reL = dim3.reLabel;
        selec_like = n_ite;
    end
    
    deviance_numc(n_ite) = dim3.deviance;
    ite_numc(n_ite) = dim3.numClass;
    if mod(n_ite, 100)==0
        fprintf('the iteration time is %d\n', n_ite);
        fprintf('num of class is %d\n', dim3.numClass);
    end
end

%% block-slice sampling loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for n_ite = 1:Niteration
% %     dim3.betas = inibetas(1:(end-1));
% %     dim3 = slice_samp(dim3);
%     dim3.betas = dirrnd([dim3.m_val dim3.gamma], 1);
%     dim3 = slice_revise(dim3);
%     dim3 = m_sample(dim3);
%     dim3.betas
%     dim3.numClass
%     dim3.m_val
%     disp(['the iteration time is ', num2str(n_ite)]);
% end

%% test mixing rate
% %% post dealing
% var_estimation = auto_cov(whole_in);

