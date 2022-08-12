function [t, w, s] = get_mel(data_dir, filename_arg)

filename = strcat(data_dir, filename_arg);
[signal_1D,fs] = audioread(filename);

signal_1D = wdenoise(signal_1D,9,'Wavelet','sym4');

[s, w, t] = melSpectrogram(signal_1D,fs, ...
                               'WindowLength',2048,...
                               'OverlapLength',1024, ...
                               'FFTLength',8192, ...
                               'NumBands',150, ...
                               'FrequencyRange', [0, 2e3]);
s = 10*log10(s+eps);
w = w(ones(1,size(t, 1)), :);
t = t(:, ones(1,size(w, 2)));

t = t(:);
w = w(:);
s = abs(s(:));

t(s < quantile(s, 0.05)) = [];
w(s < quantile(s, 0.05)) = [];
s(s < quantile(s, 0.05)) = [];

