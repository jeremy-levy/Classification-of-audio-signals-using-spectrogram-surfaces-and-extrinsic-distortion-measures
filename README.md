# Classification of audio signals using spectrogram surfaces and extrinsic distortion measures

This is the repository for the paper Classification of audio signals using spectrogram surfaces and extrinsic distortion measures.

## Dependencies

* Matlab (for ABCD optimization: [ABCD] (https://github.com/AlexNaitsat/ABCD_Algorithm))
* Python3 (for feature extraction)

## Instructions

The Matlab script ```compute_all_specs.m``` in the Distrotion_Extraction folder allows to perform the mapping of the spectrograms.
For that, the variable data_dir and base_path_save need to be defined in the script, respectively to be the directory in which there is the data, and the directory in which the mappings will be saved.
For faster computation, the Matlab code should be compiled into an exe file, and then the Python script parallel_mapping.py allows to run in parallel several mappings.

In the folder src, the Python script ```Feature_Extraction.py``` extracts all the disotrtion measures described in the paper from the mapping. 
It automatically saves them into a single csv file.


The original ABCD algorithm is available at the following link: https://github.com/AlexNaitsat/ABCD_Algorithm
