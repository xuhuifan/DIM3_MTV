function dim3 = dim3Ini(datas, numClass, dataNum, tTime)

dim3.lam1 = sum(sum(sum(datas)))/(dataNum^2*tTime);
dim3.lam2 = 1-dim3.lam1;
dim3.numClass = numClass;
dim3.dataNum = dataNum;
dim3.tTime = tTime;

re_Label = zeros(dataNum, dataNum);
re_Label = zeros(dataNum, dataNum);
for t = 1:tTime
    for i=1:dataNum
        re_value = mnrnd(1, 1/numClass*ones(1, numClass), dataNum);
        se_value = mnrnd(1, 1/numClass*ones(1, numClass), dataNum);        
        for j=1:dataNum
            reLabel(i,j, t) = find(re_value(j,:)==1);
            seLabel(i,j, t) = find(se_value(j,:)==1);
        end
    end
end


dim3.reLabel = reLabel;
dim3.seLabel = seLabel;

dim3.m_val = ones(1, numClass);