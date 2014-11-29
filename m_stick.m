function dim3 = m_stick( dim3 )
% sampling m's value
%  dim3 is the structure used
%pre_table = zeros(dim3.dataNum, dim3.numClass);
indexLabel = dim3.indexLabel;
betas = dim3.betas;
alphas = dim3.alpha;

%%% need to remove empty clusters to adjust the right class number
empty_clu = [];
for i_m = 1:dim3.numClass
    if ((isempty(find(dim3.seLabel == indexLabel(i_m))))&&(isempty(find(dim3.reLabel == indexLabel(i_m)))))
        empty_clu = [empty_clu i_m];
    end
end

if ~isempty(empty_clu)
    indexLabel(empty_clu) = [];
    dim3.numClass = dim3.numClass-length(empty_clu);
    betas(empty_clu) = [];
end

% get the m_val value
m_val = zeros(1, dim3.numClass);
nohat_m = zeros(1, dim3.numClass);
pre_Nikt = zeros(dim3.dataNum, dim3.numClass);
for t_m = 1:dim3.tTime
    for i_sli=randperm(dim3.dataNum)
        Nikt(i_sli, :) = N_count(dim3.seLabel(i_sli,:,t_m), (dim3.reLabel(:,i_sli,t_m))', indexLabel);
    end
    for i_sli = 1:dim3.dataNum
	    t_Table = zeros(1, dim3.numClass);
        nohat_t = zeros(1, dim3.numClass);
        for k_sli = 1:dim3.numClass
            i_max = Nikt(i_sli,k_sli);
            if i_max > 0
                i_stir = stirling(i_max).*((alphas*betas(k_sli)+dim3.kappa(2)*pre_Nikt(i_sli, k_sli)).^(1:i_max));
                i_stir = i_stir/sum(i_stir);
                i_nt = 1+sum(rand > cumsum(i_stir));
                i_p = (dim3.kappa(2)*pre_Nikt(i_sli, k_sli)/(alphas*betas(k_sli)+dim3.kappa(2)*pre_Nikt(i_sli, k_sli)));
                i_val = binornd(i_nt, i_p);
                t_Table(k_sli) = i_nt-i_val;
                nohat_t(k_sli) = i_nt;
            end
        end
        nohat_m = nohat_m + nohat_t;
		m_val = m_val + t_Table;
    end
    pre_Nikt = Nikt;
end

% disp(m_val);

dim3.nohat_m = nohat_m;
dim3.indexLabel = indexLabel;
dim3.m_val = m_val;
dim3.betas = betas;
end

