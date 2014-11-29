function var_est = auto_cov( whole_in )
%% estimate the auto-covaraince of the input g(x)

var_est = zeros(1, 6);
seq_length = length(whole_in);
%% batch estimate

batch_len = [10 20 30];
for i_bat = 1:length(batch_len)
    a = reshape(whole_in, batch_len(i_bat), []);
    a = mean(a);
    var_est(i_bat) = var(a);
end

%% sequence estimate
mu_n = mean(whole_in);
pos_est = 0;
mono_est = 0;
conv_est = 0;
pos_flag = 1;
mono_flag = 1;
conv_flag = 1;

gamma_nt = zeros(1, seq_length);
Gamma_nm = zeros(1, seq_length/2);
for i_bat = 0:(seq_length-1)
    gamma_nt(i_bat+1) = sum((whole_in(1:(seq_length-i_bat))-mu_n).*(whole_in((i_bat+1):seq_length)-mu_n))/seq_length;
    
    if (mod(i_bat, 2)==0) && (i_bat>0)
        Gamma_nm(i_bat/2) = gamma_nt(i_bat - 1)+gamma_nt(i_bat);
        
        if (Gamma_nm(i_bat/2) < 0) %% positive
            pos_flag =0;            
        end
        if pos_flag 
            pos_est = pos_est+Gamma_nm(i_bat/2);
        end
        
        if (i_bat/2>1) && ((Gamma_nm(i_bat/2) <0) || (Gamma_nm(i_bat/2)-Gamma_nm(i_bat/2-1)>0)) %% monotone
            mono_flag = 0;
        end
        if mono_flag
            mono_est = mono_est +Gamma_nm(i_bat/2);
        end
        
        if (i_bat/2>2) && ((Gamma_nm(i_bat/2)<0)||((Gamma_nm(i_bat/2)+Gamma_nm(i_bat/2-2)-2*Gamma_nm(i_bat/2-1))<0)) %% convex
            conv_flag = 0;
        end
        if conv_flag 
            conv_est = conv_est+Gamma_nm(i_bat/2);
        end
        
        if (~pos_flag) && ((~mono_flag)&&(~conv_flag))
            break;
        end
        
    end
    
end

var_est(4:6)=[pos_est mono_est conv_est];

end

