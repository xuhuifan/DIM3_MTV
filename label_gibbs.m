function dim3 = label_gibbs( dim3 )
% Sampling re_label, se_label
se_Labels = dim3.seLabel;
re_Labels = dim3.reLabel;
numClass = dim3.numClass;
betas = dim3.betas;
indexLabel = dim3.indexLabel;

% the formal sampler of se_Labels, re_Labels
pre_Nikt = zeros(dim3.dataNum, numClass);
for t=1:dim3.tTime
    % calculate likelihood information
    tau_kl = zeros(numClass, numClass);
    tau1_kl = zeros(numClass, numClass);
    
    for k = 1:numClass
        for l=1:numClass
            [x_loc, y_loc]=find((se_Labels(:,:,t)==indexLabel(k))&(re_Labels(:,:,t)==indexLabel(l)));
            tau1_kl(k,l)=sum(diag(dim3.datas(x_loc, y_loc, t)));
            tau_kl(k,l) = length(x_loc);
        end
    end
    Nikt = zeros(dim3.dataNum, numClass);
    la_Nikt = zeros(dim3.dataNum, numClass);
    for i=randperm(dim3.dataNum)
        Nikt(i, :) = N_count(se_Labels(i,:,t), (re_Labels(:,i,t))', indexLabel);
        if t<dim3.tTime
            la_Nikt(i,:) = N_count(se_Labels(i,:,t+1), (re_Labels(:,i,t+1))', indexLabel);
        end
    end
    
    for i=randperm(dim3.dataNum)
        for j=randperm(dim3.dataNum)
            if (size(pre_Nikt, 2) < numClass)
                pre_Nikt = [pre_Nikt zeros(dim3.dataNum, 1)];
                la_Nikt = [la_Nikt zeros(dim3.dataNum, 1)];
            end
            % se_Labels(i,j,t) weight
            se_la = find(indexLabel == se_Labels(i,j,t));
            Nikt(i, se_la) = Nikt(i, se_la)-1;
            re_la = find(indexLabel == re_Labels(i,j,t));
            Nikt(j,re_la) = Nikt(j,re_la)-1;
            
            ns_weight = dim3.alpha*betas+[(Nikt(i,:)+pre_Nikt(i,:)*dim3.kappa(2)) 0];  %% here pre_Nikt can not follow numClass's increasement
            nr_weight = dim3.alpha*betas+[(Nikt(j,:)+pre_Nikt(j,:)*dim3.kappa(2)) 0];
            if t<dim3.tTime
                la_wei1 = exp(gammaln(dim3.alpha*betas+[dim3.kappa(2)*Nikt(i,:)+la_Nikt(i,:) 0]+dim3.kappa(2))-gammaln(dim3.alpha*betas+[dim3.kappa(2)*Nikt(i,:)+la_Nikt(i,:) 0]));
                la_wei2 = exp(gammaln(dim3.alpha*betas+[dim3.kappa(2)*Nikt(i,:) 0]+dim3.kappa(2))-gammaln(dim3.alpha*betas+[dim3.kappa(2)*Nikt(i,:) 0]));
                ns_weight = (ns_weight.*la_wei1)./la_wei2;
                la_wei1 = exp(gammaln(dim3.alpha*betas+[dim3.kappa(2)*Nikt(j,:)+la_Nikt(j,:) 0]+dim3.kappa(2))-gammaln(dim3.alpha*betas+[dim3.kappa(2)*Nikt(j,:)+la_Nikt(j,:) 0]));
                la_wei2 = exp(gammaln(dim3.alpha*betas+[dim3.kappa(2)*Nikt(j,:) 0]+dim3.kappa(2))-gammaln(dim3.alpha*betas+[dim3.kappa(2)*Nikt(j,:) 0]));
                nr_weight = (nr_weight.*la_wei1)./la_wei2;
            end
            % edge(i,j,t)'s likelihood calculation
            tau_kl(se_la, re_la)=tau_kl(se_la, re_la)-1;
            tau1_kl(se_la, re_la)= tau1_kl(se_la, re_la)-dim3.datas(i,j,t);
            tau0_kl = tau_kl-tau1_kl;
            
            % calculating the likehood value
            if (dim3.datas(i,j,t)==1)
                like_wei = ([tau1_kl zeros(numClass,1);zeros(1,(numClass+1))]+dim3.lam1)./([tau_kl zeros(numClass,1);zeros(1,numClass+1)]+dim3.lam1+dim3.lam2);  % change the denominator
            else
                like_wei = ([tau0_kl zeros(numClass,1);zeros(1,(numClass+1))]+dim3.lam2)./([tau_kl zeros(numClass,1);zeros(1,numClass+1)]+dim3.lam1+dim3.lam2);
            end
%             like_wei = ones(numClass+1, numClass+1);
            p_weight = diag(ns_weight)*like_wei*diag(nr_weight);

            p_weight = reshape(p_weight, 1, []);
            
            % sampling
            ath_value = 1+sum(rand*sum(p_weight) > cumsum(p_weight));
            ath_col = ceil(ath_value/(numClass+1));
            ath_row = ath_value - (ath_col-1)*(numClass+1);
            
            %%%
            % consider the case (ath_col > numClass) or (ath_row > numClass)
            % Nikt's changing
            if ((ath_row > numClass)||(ath_col > numClass))
                bb = dirrnd([1 dim3.gamma], 1);
                betas(end:(end+1)) = betas(end)*bb;
                
                Nikt = [Nikt zeros(dim3.dataNum, 1)];
                tau_kl = [tau_kl zeros(numClass, 1); zeros(1, (numClass+1))];
                tau1_kl = [tau1_kl zeros(numClass, 1); zeros(1, (numClass+1))];
                indexLabel = [indexLabel indexLabel(end)+1];
                numClass = numClass+1;

            end
            
            se_Labels(i,j,t) = (ath_row);
            re_Labels(i,j,t) = (ath_col);
            %%%
            tau_kl(ath_row, ath_col)=tau_kl(ath_row, ath_col)+1;
            tau1_kl(ath_row, ath_col)=tau1_kl(ath_row, ath_col)+dim3.datas(i,j,t);
            
            % we need to increase Nikt's specific value
            Nikt(i, ath_row) = Nikt(i, ath_row)+1;
            Nikt(j, ath_col) = Nikt(j, ath_col)+1;
            
        end
    end
    pre_Nikt = Nikt;
end

%%%
dim3.seLabel = se_Labels;
dim3.reLabel = re_Labels;
dim3.numClass = numClass;
dim3.betas = betas;
dim3.indexLabel = 1:numClass;

end

