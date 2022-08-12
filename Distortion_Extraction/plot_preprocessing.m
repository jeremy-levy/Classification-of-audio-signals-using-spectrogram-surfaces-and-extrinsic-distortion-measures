% 

%% Clean workspace and add paths
clear all 
close all

%% Run compute_spec for different file_name
data_dir = "C:\Users\jeremy\Documents\lung respiratory\respiratory-sound-database\Respiratory_Sound_Database\Waves\";
myFiles = dir(fullfile(data_dir,'*.wav'));

order = 3;
framelen = 11;
length_sig = 1000;
begin_sig = 1000;

for k = 1:length(myFiles)
    baseFileName = myFiles(k).name;
    filename = data_dir + baseFileName;
    
    [signal_2D,~] = audioread(filename);

    
    signal_2D_preprocessed = sgolayfilt(signal_2D,order,framelen);
    
    length_sig = 1000;
    begin_sig = 1000;

    f = figure('visible','off');
    
    hold on
    plot(signal_2D(begin_sig:begin_sig+length_sig),'r','LineWidth',1);
    plot(signal_2D_preprocessed(begin_sig:begin_sig+length_sig),'b','LineWidth',1);

    xlabel("Time (sec)");
    ylabel("Amplitude");
    legend({'Raw data','Preprocessed signal'})

    xlim([0 length_sig])

    ax = gca;
    ax.FontSize = 13;
    
    saveas(f,strcat('C:\Users\jeremy\Documents\lung respiratory\figures\prep\', int2str(k)),'jpg')
    
end

