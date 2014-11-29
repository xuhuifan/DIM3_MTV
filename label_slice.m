function dim3 = label_slice(dim3)
% using block-slice sampling scheme to do the sampling
se_Labels = dim3.seLabel;  % data's sender's label
re_Labels = dim3.reLabel;  % data's receiver's label
dataNum = dim3.dataNum;    % data number
numClass = dim3.numClass;  % cluster number
indexLabel = dim3.indexLabel;

alphas = dim3.alpha;
betas = dim3.betas;  % betas's value need to be re-defined


for t_sli = 1:dim3.tTime
    tau_kl = zeros(numClass, numClass);  % counts for inter-cluster occurancy
    tau1_kl = zeros(numClass, numClass); % counts for inter-cluster data occurancy
    
    for k = 1:numClass
        for l=1:numClass
            [x_loc, y_loc]=find((se_Labels(:,:,t_sli)==indexLabel(k))&(re_Labels(:,:,t_sli)==indexLabel(l)));
            tau1_kl(k,l)=sum(diag(dim3.datas(x_loc, y_loc, t_sli)));
            tau_kl(k,l) =length(x_loc);
        end
    end
    % count each class's data size in each time
    Nikt = zeros(dataNum, numClass);
    la_Nikt = zeros(dim3.dataNum, numClass);
    qpi_value = zeros(dataNum, numClass+1); %% \omega's value in Maria's paper
    
    % initialize the \pi's value
    for i_sli=randperm(dataNum)
        Nikt(i_sli, :) = N_count(se_Labels(i_sli,:,t_sli), (re_Labels(:,i_sli,t_sli))', indexLabel);
        if t_sli<dim3.tTime
            la_Nikt(i_sli,:) = N_count(se_Labels(i_sli,:,t_sli+1), (re_Labels(:,i_sli,t_sli+1))', indexLabel);
        end
    end
    
    f2 = zeros(dataNum, 2*dataNum);
    if t_sli > 1
        a_ik = repmat(alphas*betas(1:numClass), dataNum, 1)+Nikt+dim3.kappa(2)*pre_Nikt; %%% beta distribution's parameters
        b_ik = repmat(alphas*(1-cumsum(betas(1:numClass))), dataNum, 1)+2*dataNum-cumsum(Nikt, 2)+dim3.kappa(2)*(2*dataNum-cumsum(pre_Nikt, 2));
    else
        a_ik = repmat(alphas*betas(1:numClass), dataNum, 1)+Nikt; %%% beta distribution's parameters
        b_ik = repmat(alphas*(1-cumsum(betas(1:numClass))), dataNum, 1)+2*dataNum-cumsum(Nikt, 2);
    end
    pi_value=betarnd(a_ik, b_ik);
    qpi_value(:,1:numClass) = pi_value.*cumprod((1-[zeros(dataNum, 1) pi_value(:,(1:(end-1)))]), 2);
    
    qpi_value(:,end)=1-sum(qpi_value, 2);
    
    us_vals = zeros(dataNum, 2*dataNum);
    for i_sli = 1:dataNum
        i_vals = qpi_value(i_sli,1:numClass);
        %         fprintf('length of i_vals is %d\n', numClass);
        %         fprintf('maximum value of labels is %d\n', max([se_Labels(i_sli ,: , t_sli) re_Labels(:, i_sli, t_sli)']));
        [f1, f2(i_sli, :)] = ismember([se_Labels(i_sli ,: , t_sli) re_Labels(:, i_sli, t_sli)'], indexLabel);
        us_vals(i_sli, :) = i_vals(f2(i_sli, :));
    end
    
    uij = rand(dataNum, 2*dataNum).*us_vals;
    clear us_vals f1 i_vals;
    for i_sli = 1:dataNum
        while qpi_value(i_sli, end) > min(uij(i_sli,:))
            betas(end) = betas(end)*betarnd(1, dim3.gamma);
            ai = alphas*betas(end);
            betas(end+1)=1-sum(betas);
            bi = alphas*betas(end);
            qpi_value(:,end) = qpi_value(:,end).*betarnd(repmat(ai, dataNum, 1), repmat(bi, dataNum, 1));
            qpi_value(:, end+1) = 1- sum(qpi_value, 2);
            Nikt = [Nikt zeros(dataNum, 1)];
            la_Nikt = [la_Nikt zeros(dataNum, 1)];
            numClass = numClass + 1;
            indexLabel = [indexLabel indexLabel(end)+1];
            tau_kl = [tau_kl zeros(numClass-1, 1);zeros(1, numClass)];
            tau1_kl = [tau1_kl zeros(numClass-1, 1);zeros(1, numClass)];
        end
    end
    
    for i_sli = 1:dataNum
        for j_sli = 1:dataNum
            
            n_tau = tau_kl;
            n_tau1 = tau1_kl;
            % the related label
            seLa = f2(i_sli, j_sli);
            reLa = f2(j_sli, (dataNum+i_sli));
            
            Nikt(i_sli, seLa) = Nikt(i_sli, seLa) -1;
            Nikt(j_sli, reLa) = Nikt(j_sli, reLa) -1;
            
            ui = uij(i_sli, j_sli);
            uj = uij(j_sli, (dataNum+i_sli));
            
            
            i_label = find(qpi_value(i_sli, :)>ui);
            j_label = find(qpi_value(j_sli, :)>uj);
            n_tau(seLa, reLa)=n_tau(seLa, reLa)-1;
            n_tau1(seLa, reLa)= n_tau1(seLa, reLa)-dim3.datas(i_sli,j_sli,t_sli);
            n_tau0 = n_tau - n_tau1;
            
            if t_sli<dim3.tTime
                cu_Ni = Nikt(i_sli,:);
                la_Ni = la_Nikt(i_sli, :);
                la_wei1 = exp(gammaln(dim3.alpha*betas(i_label)+[dim3.kappa(2)*cu_Ni(i_label)+la_Ni(i_label)]+dim3.kappa(2))-gammaln(dim3.alpha*betas(i_label)+[dim3.kappa(2)*cu_Ni(i_label)+la_Ni(i_label)]));
                la_wei2 = exp(gammaln(dim3.alpha*betas(i_label)+[dim3.kappa(2)*cu_Ni(i_label)]+dim3.kappa(2))-gammaln(dim3.alpha*betas(i_label)+[dim3.kappa(2)*cu_Ni(i_label)]));
                is_wei = la_wei1./la_wei2;
                
                cu_Ni = Nikt(j_sli,:);
                la_Ni = la_Nikt(j_sli, :);
                la_wei1 = exp(gammaln(dim3.alpha*betas(j_label)+[dim3.kappa(2)*cu_Ni(j_label)+la_Ni(j_label)]+dim3.kappa(2))-gammaln(dim3.alpha*betas(j_label)+[dim3.kappa(2)*cu_Ni(j_label)+la_Ni(j_label)]));
                la_wei2 = exp(gammaln(dim3.alpha*betas(j_label)+[dim3.kappa(2)*cu_Ni(j_label)]+dim3.kappa(2))-gammaln(dim3.alpha*betas(j_label)+[dim3.kappa(2)*cu_Ni(j_label)]));
                ir_wei = la_wei1./la_wei2;
            end
            
            % edge(i,j,t)'s likelihood calculation
            if (dim3.datas(i_sli,j_sli,t_sli)==1)
                like_wei = (n_tau1(i_label, j_label)+dim3.lam1)./(n_tau(i_label, j_label)+dim3.lam1+dim3.lam2);  % change the denominator
            else
                like_wei = (n_tau0(i_label, j_label)+dim3.lam2)./(n_tau(i_label, j_label)+dim3.lam1+dim3.lam2);
            end
            
            % the whole weights
%             fprintf('length of is_wei is %d\n', length(is_wei));
%             fprintf('length of ir_wei is %d\n', length(ir_wei));
%             fprintf('size 1 of like_wei is %d\n', size(like_wei, 1));
%             fprintf('size 2 of like_wei is %d\n', size(like_wei, 2));
            if t_sli < dim3.tTime
                like_wei = diag(is_wei)*like_wei*diag(ir_wei);
            end
            like_wei = reshape(like_wei, 1, []);
            like_wei = like_wei/sum(like_wei);
            
            ath_value = 1+sum(rand>cumsum(like_wei));
            ath_col = ceil(ath_value/length(i_label));
            ath_row = ath_value - (ath_col-1)*(length(i_label));
            
            se_Labels(i_sli, j_sli, t_sli) = i_label(ath_row);
            re_Labels(i_sli, j_sli, t_sli) = j_label(ath_col);
            
            Nikt(i_sli, i_label(ath_row)) = Nikt(i_sli, i_label(ath_row))+1;
            Nikt(j_sli, j_label(ath_col)) = Nikt(j_sli, j_label(ath_col))+1;
        end
    end
    
    pre_Nikt = Nikt;
end
dim3.seLabel = se_Labels;
dim3.reLabel = re_Labels;
dim3.numClass = numClass;
dim3.betas = betas;
dim3.indexLabel = 1:numClass;

end