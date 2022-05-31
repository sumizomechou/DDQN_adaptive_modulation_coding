function [BER]=BER_Cacula(S_code,R_code)
if (length(S_code)~=length(R_code))
    error('The length of S_code and R_code must be same!!!')
else
    BER=length(find(S_code-R_code)~=0)/length(S_code);
end
%    index=find(S_code~=R_code)