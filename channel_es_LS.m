function [sigma_ls,  chn_ls_time]=channel_es_LS(FadedSig,pn,chan_order,training_length)
% tr_random_sequence = randn(N+P-1,1);      % 随机数
% tr_random_sequence = round(rand(N+P-1,1))*2-1;      % 随机数
tr_random_sequence = pn';

first_colume_random = tr_random_sequence(chan_order:end);      %%%此处设计测量矩阵C_matrix_random
first_row_random = tr_random_sequence(chan_order:-1:1);
C_matrix_random = toeplitz(first_colume_random,first_row_random);

measure_matrix = C_matrix_random;

Observation_vector =FadedSig(length(pn)-training_length+1:length(pn));       % 此处为观测向量
chn_ls_time = pinv(measure_matrix)*Observation_vector;
noise_variance =  Observation_vector - C_matrix_random*chn_ls_time;
sigma_ls= norm(noise_variance,2)/length(noise_variance);
sigma_ls=sqrt(sigma_ls);
