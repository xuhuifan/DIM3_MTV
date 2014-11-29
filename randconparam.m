function alpha = randconparam(alpha,numdata,numclass,aa,bb,numiter);

% Modification of Escobar and West.  Works for multiple groups of data.
% numdata, numclass are row vectors, one element per group.

totalclass = sum(numclass);
num = length(numdata);

for ii = 1:numiter
    
  xx = betarnd((alpha+1)*ones(1,num),numdata);
  zz = rand(1,num).*(alpha+numdata)<numdata;

  gammaa = aa + totalclass - sum(zz);
  gammab = bb - sum(log(xx));

  alpha = randg(gammaa)./gammab;

end

