%%% save data to txt file
% new_gibbs1 = zeros(10000, 5);
% new_gibbs2 = zeros(10000, 5);
% new_gibbs3 = zeros(10000, 5);
% for i = 1:5
%     new_gibbs1(:,i) = ite_numc((10000*i+1):(10000*(i+1)));
%     new_gibbs2(:,i) = deviance_numc((10000*i+1):(10000*(i+1)));
%     new_gibbs3(:,i) = cu_jps((10000*i+1):(10000*(i+1)));
% end
a = 30001:20:130000;
new_gibbs1 = ite_numc(a);
new_gibbs2 = deviance_numc(a);
new_gibbs3 = cu_jps(a);
dlmwrite('case1_gibbsnum.txt',new_gibbs1,'delimiter',' ','newline','pc');
dlmwrite('case1_gibbsdev.txt',new_gibbs2,'delimiter',' ','newline','pc');
dlmwrite('case1_gibbsjps.txt',new_gibbs3,'delimiter',' ','newline','pc');


