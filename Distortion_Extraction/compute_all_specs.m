% 

%% Clean workspace and add paths

clear all 
close all
Set_ABCD_Paths

%% Run compute_spec for different file_name

% IMPORTANT: TO BE DEFINED
data_dir = "";
base_path_save = "";

to_plot = false;
myFiles = dir(fullfile(data_dir,{'*.wav'}));
threshold = 0.002;


for k = 1 :length(myFiles)
    baseFileName = myFiles(k).name;
    compute_spec(baseFileName, base_path_save, data_dir);
end
