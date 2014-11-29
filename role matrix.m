%% role-compatibility matrix    
numClass = 3;
    tau_kl = zeros(numClass, numClass);
    tau1_kl = zeros(numClass, numClass);
    indexLabel = 1:numClass;
    true_tau = zeros(numClass, numClass, dim3.tTime);
    for t=1:dim3.tTime;
    for k = 1:numClass
        for l=1:numClass
            [x_loc, y_loc]=find((like_seL(:,:,t)==indexLabel(k))&(like_reL(:,:,t)==indexLabel(l)));
            tau_kl(k,l) = length(x_loc);
            tau1_kl(k,l)=sum(diag(dim3.datas(x_loc, y_loc, t)));
        end
    end
    
    true_tau(:,:,t) = (tau1_kl)./(tau_kl);
    
    end
    
    mean(true_tau, 3)
    