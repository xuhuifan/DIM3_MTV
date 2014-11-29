function Nikt = N_count( se_la, re_la, indexLabel)
% counting node i's occupying status at time t
current_la = [se_la re_la];
%Nikt = zeros(1, numClass);
if length(indexLabel)>1
    [Nikt xout] = hist(current_la, indexLabel);
else
    [Nik xout] = hist(current_la, [indexLabel (indexLabel+1)]);
    Nikt = Nik(1);
end
%Nikt(unique(current_la))= x;  %% some problems
end

