function [deviance, cu_jointp, li_jps] = gibbs_dev(dim3)
deviance = 0;
cu_jointp = 0;
se_Labels = dim3.seLabel;
re_Labels = dim3.reLabel;
numClass = dim3.numClass;
indexLabel = dim3.indexLabel;
alphas = dim3.alpha;
betas = dim3.betas;
li_jps = 0;
% ratio_time = 0;
% ratio_u = 1;
% fprintf('numClass is %d\n', numClass);
% for i=1:numClass
%     fprintf('length of indexLabel is %d\n', (indexLabel(i)));
% end


for k_dev = 1:(numClass-1)
    if k_dev > 1
        k_beta = betas(k_dev)/(1-sum(betas(1:(k_dev-1))));
    else
        k_beta = betas(1);
    end
    cu_jointp = cu_jointp+log(betapdf(k_beta, 1, dim3.gamma));
    %     if (log(betapdf(k_beta, 1, dim3.gamma))==inf)
    %                 fprintf('value of beta leave is %f\n', 1-sum(betas));
    %         fprintf('value of k_dev is %f\n', k_dev);
    %         fprintf('value of k_beta is %f\n', k_beta);
    %         fprintf('oh my god\n');
    %     end
end

pre_Nikt = zeros(dim3.dataNum, numClass);

for t_dev = 1:dim3.tTime
    tau_kl = zeros(numClass, numClass);
    tau1_kl = zeros(numClass, numClass);
    
    for k = 1:numClass
        for l=1:numClass
            [x_loc, y_loc]=find((se_Labels(:,:,t_dev)==indexLabel(k))&(re_Labels(:,:,t_dev)==indexLabel(l)));
            tau1_kl(k,l)=sum(diag(dim3.datas(x_loc, y_loc, t_dev)));
            tau_kl(k,l) =length(x_loc);
        end
    end
    
    Nikt = zeros(dim3.dataNum, numClass);
    for i=1:(dim3.dataNum)
        Nikt(i, :) = N_count(se_Labels(i,:,t_dev), (re_Labels(:,i,t_dev))', indexLabel);
    end

    
    for i_dev = 1:dim3.dataNum
        for j_dev = 1:dim3.dataNum
            seL = find(indexLabel==se_Labels(i_dev, j_dev, t_dev));
            reL = find(indexLabel==re_Labels(i_dev, j_dev, t_dev));
            
            tau_kl(seL, reL)=tau_kl(seL, reL)-1;
            tau1_kl(seL, reL)= tau1_kl(seL, reL)-dim3.datas(i_dev,j_dev,t_dev);
            tau0_kl = tau_kl-tau1_kl;
            
            %             if sum(sum(tau_kl < 0)) >0
            %                 fprintf('here here \n');
            %             end
            %             if sum(sum(tau1_kl < 0)) >0
            %                 fprintf('here here \n');
            %             end
            % deviance's calculation
            if (dim3.datas(i_dev,j_dev,t_dev)==1)
                like_wei = (tau1_kl+dim3.lam1)./(tau_kl+dim3.lam1+dim3.lam2);
            else
                like_wei = (tau0_kl+dim3.lam2)./(tau_kl+dim3.lam1+dim3.lam2);
            end
            wei_ij = (diag(Nikt(i_dev, :))*like_wei)*diag(Nikt(j_dev, :))/(4*(dim3.dataNum)^2*dim3.tTime);
            deviance = deviance+log(sum(sum(wei_ij)));
            % end
            
            % log joint probability's calculation
            Nikt(i_dev, seL) = Nikt(i_dev, seL)-1;
            Nikt(j_dev, reL) = Nikt(j_dev, reL)-1;
           
            if (dim3.datas(i_dev,j_dev,t_dev)==1)
                like_wei = (tau1_kl(seL, reL)+dim3.lam1)/(tau_kl(seL, reL)+dim3.lam1+dim3.lam2);
            else
                like_wei = (tau0_kl(seL, reL)+dim3.lam2)/(tau_kl(seL, reL)+dim3.lam1+dim3.lam2);
            end
            
            if t_dev > 1
                ps = (alphas*betas(seL)+dim3.kappa(2)*pre_Nikt(i_dev, seL)+Nikt(i_dev, seL))/(alphas+(dim3.kappa(2)+1)*2*dim3.dataNum-1);
                pr = (alphas*betas(reL)+dim3.kappa(2)*pre_Nikt(j_dev, reL)+Nikt(j_dev, reL))/(alphas+(dim3.kappa(2)+1)*2*dim3.dataNum-1);
            else
                ps = (alphas*betas(seL)+Nikt(i_dev, seL))/(alphas+2*dim3.dataNum-1);
                pr = (alphas*betas(reL)+Nikt(j_dev, reL))/(alphas+2*dim3.dataNum-1);
            end
            cu_jointp = cu_jointp+log(like_wei)+log(ps)+log(pr);
%             ratio_l = min([log(like_wei)/log(ps), log(like_wei)/log(pr)]);
%             if ratio_l < 1
%                 ratio_time = ratio_time + 1;
%             end
%             ratio_u = max([ratio_u, log(like_wei)/log(ps), log(like_wei)/log(pr)]);
            li_jps = li_jps + log(like_wei);
%             if i_dev == j_dev
%                 Nikt(i_dev,:) = Nikt(i_dev,:)-1;
%             end
            
            Nikt(i_dev, seL) = Nikt(i_dev, seL)+1;
            Nikt(j_dev, reL) = Nikt(j_dev, reL)+1;
            
            tau_kl(seL, reL)=tau_kl(seL, reL)+1;
            tau1_kl(seL, reL)= tau1_kl(seL, reL)+dim3.datas(i_dev,j_dev,t_dev);
        end
    end
    pre_Nikt = Nikt;
end
deviance = -2*deviance;
end