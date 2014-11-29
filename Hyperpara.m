function [alphas, gammas, kappas]= Hyperpara( nohat_m, m_val, numClass, dataNum,  tTime, pre_alpha, pre_gamma, kappas)
% sampling the hyper-parameters \gamma, \alpha

% input:
%  m_val: K-length vector denoting each component's table number
%  nums: N-length vector denoting each restaurant's dish number
% dataNum: the data points number
% cpi : the \pi's value
% seLabel : \{s_{ij}\}_{n\times n}
% reLabel : \{r_{ij}\}_{n\times n}
% ctheta : the currrent copula function parameter

% output:
% alphas: the required alpha value
% gammas: the required gamma value
% cthetas: the new copula function parameter theta's value
%
%
% sampling \gamma
% 
%  m_total = sum(m_val); % m_{\cdot\cdot} value in \gamma's posterior distribution
% K = max(nums); % K in \gamma's posterior distribution
% 
% m_plier = (K-1/2)*0-exp(-0)/2-sum(log(exp(0)+[0:(m_total-1)]));
% % fprintf('m_total is %d\n', m_total);
% %  fprintf('gamma m_plier is %f\n', m_plier);
% func   = @(gx) exp(-m_plier+(K-1/2)*gx-exp(-gx)/2-sum(log(repmat(exp(gx),1,m_total)+repmat([0:(m_total-1)], length(gx), 1)), 2));  % the proportional pdf function
% log_func = @(gx) -m_plier+(K-1/2)*gx-exp(-gx)/2-sum(log(repmat(exp(gx),1,m_total)+repmat([0:(m_total-1)], length(gx), 1)), 2);  % the log pdf function
% log_dev = @(gx) (K-1/2)+exp(-gx)/2-sum(repmat(exp(gx),1,m_total)./(repmat(exp(gx),1,m_total)+repmat([0:(m_total-1)], length(gx), 1)), 2);  % the first derivative of log pdf function
% 
% init_points = [-2 ; -0.75; 1; 2];
% bounds = [-inf, inf];
% MAX_ACCEPTED = 1;
% IS_PLOT = 0;
% plot_bounds = [-10, 10];
% %  fprintf('gamma sampling\n');
% [samples num_sampled] = ARS(func, log_func, log_dev, init_points, MAX_ACCEPTED, IS_PLOT, bounds, plot_bounds);
% % samples = samples';
% gammas = exp(samples);
% 
% % sampling \alpha
% %
% % % using Adaptive Rejection Sampling to sample \alpha cause some problem because the log_func varies too dramastic
% % %
% % m_total = m_total/dataNum;
% % m_plier = (m_total-1/2)*0-exp(-0)/2-sum(log(exp(0)+[0:((2*dataNum)-1)]));
% % 
% % %  fprintf('alpha m_plier is %f\n', m_plier);
% % % func   = @(ax) exp(-m_plier+(m_total-1/2)*ax-exp(-ax)/2-dataNum*(sum(log(repmat(exp(ax),1,2*dataNum)+repmat([0:(2*dataNum-1)], length(ax), 1)), 2)));  % the proportional pdf function
% % % log_func = @(ax) -m_plier+(m_total-1/2)*ax-exp(-ax)/2-dataNum*(sum(log(repmat(exp(ax),1,2*dataNum)+repmat([0:(2*dataNum-1)], length(ax), 1)), 2));  % the log pdf function
% % % log_dev = @(ax) (m_total-1/2)+exp(-ax)/2-dataNum*sum((repmat(exp(ax),1,2*dataNum)./(repmat(exp(ax),1,2*dataNum)+repmat([0:(2*dataNum-1)], length(ax), 1))), 2);  % the first derivative of log pdf function
% % func   = @(gx) exp(-m_plier+(m_total-1/2)*gx-exp(-gx)/2-sum(log(repmat(exp(gx),1,(2*dataNum))+repmat([0:((2*dataNum)-1)], length(gx), 1)), 2));  % the proportional pdf function
%  log_func = @(gx) (m_total-1/2)*gx-exp(-gx)/2-dataNum*sum(log(repmat(exp(gx),1,(2*dataNum))+repmat([0:((2*dataNum)-1)], length(gx), 1)), 2);  % the log pdf function
% % log_dev = @(gx) (m_total-1/2)+exp(-gx)/2-sum(repmat(exp(gx),1,(dataNum*2))./(repmat(exp(gx),1,(2*dataNum))+repmat([0:((2*dataNum)-1)], length(gx), 1)), 2);  % the first derivative of log pdf function
% % 
% % init_points = [2 ; -0.75; 1; -1.5];
% % bounds = [-10, 10];
% % MAX_ACCEPTED = 5;
% % IS_PLOT = 0; 
% % plot_bounds = [-10, 10];
% % %  fprintf('alpha sampling\n');
% % [samples num_sampled] = ARS(func, log_func, log_dev, init_points, MAX_ACCEPTED, IS_PLOT, bounds, plot_bounds);
% % alphas = exp(samples(end));
% % %
% %
% % Instead, we use the auxiliary variable method in Teh.(Hierarchical Dirichlet Process 2006) to sample \alpha
% %
% %

alpha_kappa = randconparam((pre_alpha+kappas*2*dataNum), 2*dataNum*ones(1, dataNum*tTime), sum(nohat_m), 1, 1 , 10);

ratio = betarnd(1+sum(m_val), 1+sum(nohat_m)-sum(m_val));
alphas = alpha_kappa*ratio;
kappas = (alpha_kappa - alphas)/(2*dataNum);

gammas = randconparam(pre_gamma, sum(m_val), numClass, 1, 1, 10);


end

