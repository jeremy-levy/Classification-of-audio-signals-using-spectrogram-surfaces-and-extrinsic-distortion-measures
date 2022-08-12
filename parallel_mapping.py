import subprocess
from icecream import ic
import librosa
import os
from glob import glob
from joblib import Parallel, delayed


def run_ABCD(data_path):
    filenames = os.listdir(data_path)[0]
    Parallel(n_jobs=-1)(delayed(run_one_file)(arg) for arg in filenames)


def run_one_file(file):
    args = [os.path.join('Distortion_Extraction', 'compute_all_specs.exe'), file]
    child_proccess = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    child_output = child_proccess.communicate()[0]
    return child_output
