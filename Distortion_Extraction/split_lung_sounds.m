function [mins_indices] = split_lung_sounds(signal_1D, Fs, k)

%% compute spectrogram
dimension_t = 150;
len_window = fix(size(signal_1D, 1) / dimension_t);
w_pi = linspace(0,3.1416,dimension_t);

[s,w,~] = spectrogram(signal_1D, len_window, 0, w_pi);

%% apply band-pass filter
f_l = (80*pi) / Fs;
f_h = (1000*pi) / Fs;

band_spec = s(w > f_l,:);
band_pass = w(w > f_l);
band_spec = band_spec(band_pass < f_h,:);

%% Find minimum of smooth p_n
p_n = sum(abs(band_spec));
% X_sn = rsmooth(p_n);
X_sn = smoothn(p_n, 'robust');

order = 2;
framelen = 21;
X_sn_sgola = sgolayfilt(X_sn,order,framelen);

figure;
hold on;
plot(p_n)
plot(X_sn)
plot(X_sn_sgola)
legend('p_n', 'X_sn', 'X_sn_sgola')
title(strcat('k=', int2str(k)))

if std(X_sn_sgola) < 0.1
    mins_indices = [];
else
    mins = islocalmin(X_sn_sgola);
    mins_indices = find(mins);

    if size(mins_indices) > 0
        mins_indices = (mins_indices*size(signal_1D, 1)) / dimension_t;
        mins_indices(1) = 1;
    end
end
end

    


