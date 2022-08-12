import pandas as pd
import os
import numpy as np

from ML_analysis.Feature_Engineering.Distortion_Measures import apply_mean_distorsion, quasi_conformal, mips, \
    dirichlet, quasi_isometric, rigidity_energy, symmetric_rigid_energy, linear_energy, area_distortion


def extract_variables_from_file(f, data_path):
    sing_pd = pd.read_csv(os.path.join(data_path, 'sing_val', f), header=None)
    area_pd = pd.read_csv(os.path.join(data_path, 'area', f), header=None)
    freq_pd = pd.read_csv(os.path.join(data_path, 'freq', f), header=None)
    dist_pd = pd.read_csv(os.path.join(data_path, 'dist', f), header=None)
    energy = pd.read_csv(os.path.join(data_path, 'energy', f))

    dist_list = dist_pd.iloc[:, 0]
    dist_list = np.array(dist_list)

    area_list = area_pd.iloc[:, 0]
    area_list = np.array(area_list)

    freq_list = freq_pd.iloc[:, 0]
    freq_list = np.array(freq_list)

    first_member = sing_pd.iloc[:, 0]
    first_member = np.array(first_member)

    second_member = sing_pd.iloc[:, 1]
    second_member = np.array(second_member)

    energy = energy.iloc[:, 0].name
    energy = float(energy)

    return area_list, freq_list, first_member, second_member, dist_list, energy


def apply_all_energies(first_member, second_member, area_list=None):
    qc_value = apply_mean_distorsion(quasi_conformal, first_member, second_member, area_list)
    mips_value = apply_mean_distorsion(mips, first_member, second_member, area_list)
    dirichlet_value = apply_mean_distorsion(dirichlet, first_member, second_member, area_list)
    qi_value = apply_mean_distorsion(quasi_isometric, first_member, second_member, area_list)
    rigidity_value = apply_mean_distorsion(rigidity_energy, first_member, second_member, area_list)
    symetric_value = apply_mean_distorsion(symmetric_rigid_energy, first_member, second_member, area_list)
    linear_value = apply_mean_distorsion(linear_energy, first_member, second_member, area_list)
    area_distortion_value = apply_mean_distorsion(area_distortion, first_member, second_member, area_list)

    return qc_value, mips_value, dirichlet_value, qi_value, rigidity_value, symetric_value, linear_value, \
           area_distortion_value
