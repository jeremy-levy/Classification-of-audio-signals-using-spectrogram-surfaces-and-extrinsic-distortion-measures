import pandas as pd
import os
import numpy as np
from tqdm import tqdm

from src.util_funcs import extract_variables_from_file, apply_all_energies


def apply_distortion_measure_freq(data_path):
    for i, f in enumerate(tqdm(os.listdir(os.path.join(data_path, 'dist')))):
        area_list, freq_list, first_member, second_member, dist_list, _ = extract_variables_from_file(f, data_path)

        p1 = np.percentile(freq_list, 50)

        qc1, mips1, dirichlet1, qi1, rigidity1, symetric1, linear1, ad_list1 = apply_all_energies(
            first_member[freq_list <= p1], second_member[freq_list <= p1],
            area_list=area_list[freq_list <= p1]
        )

        freq_2 = freq_list > p1
        qc2, mips2, dirichlet2, qi2, rigidity2, symetric2, linear2, ad_list2 = apply_all_energies(
            first_member[freq_2], second_member[freq_2],
            area_list=area_list[freq_2]
        )

        qc_noav, mips_noav, dirichlet_noav, qi_noav, rigidity_noav, symmetric_noav, linear_noav, ad_noav = apply_all_energies(
            first_member, second_member,
        )

        dist1 = np.average(dist_list[freq_list <= p1], weights=area_list[freq_list <= p1])
        dist2 = np.average(dist_list[freq_2], weights=area_list[freq_2])

        ID_patient = f

        dist_df = pd.DataFrame({
            "qc_first": [qc1],
            "qc_second": [qc2],
            "mips_first": [mips1],
            "mips_second": [mips2],
            "dirichlet_first": [dirichlet1],
            "dirichlet_second": [dirichlet2],
            "qi_first": [qi1],
            "qi_second": [qi2],
            "rigidity_first": [rigidity1],
            "rigidity_second": [rigidity2],
            "symmetric_first": [symetric1],
            "symmetric_second": [symetric2],
            "linear_first": [linear1],
            "linear_second": [linear2],
            "ad_list_first": [ad_list1],
            "ad_list_second": [ad_list2],

            "qc_noav": [qc_noav],
            "mips_noav": [mips_noav],
            "dirichlet_noav": [dirichlet_noav],
            "qi_noav": [qi_noav],
            "rigidity_noav": [rigidity_noav],
            "symmetric_noav": [symmetric_noav],
            "linear_noav": [linear_noav],
            "ad_list_noav": [ad_noav],

            "dist_first": [dist1],
            "dist_second": [dist2],

            "ID": [ID_patient],
        })

        print('all_features', dist_df.shape)
        if i == 0:
            dist_df.to_csv(os.path.join('data_csv', "all_features.csv"), index=False)
        else:
            dist_df.to_csv(os.path.join('data_csv', "all_features.csv"), mode='a', header=False, index=False)


if __name__ == '__main__':
    apply_distortion_measure_freq()
