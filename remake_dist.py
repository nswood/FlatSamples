# Imports basics

import numpy as np
import pandas as pd
import h5py
import json
import uproot
import os,sys

# Defines important variables

particle_num = 50
file_num_sig = 225
file_num_bkg = 1000
fill_factor = 1
pt_range = [200., 800.]
mass_range = [40., 240.]
signal_list = ['flat_qq']
background_list = ['QCD_HT300to500', 'QCD_HT500to700', 'QCD_HT700to1000', 'QCD_HT_1000to1500', 'QCD_HT_2000toInf', 'QCD_HT_1500to2000']
output_name = "data/FullQCD_225Sig_Zqq_fillfactor1.h5"

# Opens json files for signal and background

with open("pf.json") as jsonfile:
    payload = json.load(jsonfile)
    weight = payload['weight']
    features_track = payload['features_track']
    conversion_track = payload['conversion_track']
    features_tower = payload['features_tower']
    conversion_tower = payload['conversion_tower']
    ss = payload['ss_vars']

# Creates the column names of the final data frame

part_features = []
for iVar in features_track:
    for i0 in range(particle_num):
        part_features.append(iVar + str(i0))

columns = ss + weight + part_features + ['label']

# Unnests a pandas dataframe


def unnest(df, explode):
    """unnest(DataFrame,List)"""
    idx = df.index.repeat(df[explode[0]].str.len())
    df1 = pd.concat([
        pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx

    return df1.join(df.drop(explode, 1), how='left')


# Makes a data set where the distribution of the background across mass and pT is similar to that of the signal

def remake(iFiles_sig, iFiles_bkg, iFile_out):
    """remake(list[array(nxm),...], list[array(nxs),...], str)"""

    # Creates the signal data frame
    
    df_sig_to_concat = []
    for sig in iFiles_sig:
        file_list = os.listdir(payload['samples'][sig])
        print(len(file_list))
        for i in range(file_num_sig):
            print(i)
            print(file_num_sig)
            data_set = payload['samples'][sig]+file_list[i]
            arr_sig_to_concat_temp = []
            file1 = uproot.open(data_set)
            tree = file1['tree']
            branches = tree.arrays()
            event_num = len(branches['jet_pt'])
            print('event num')
            print(event_num)
            df_sig_tower = pd.DataFrame({column: list(branches[conversion_tower[column]]) for column in features_tower})
            df_sig_tower = unnest(df_sig_tower, features_tower)
            df_sig_track = pd.DataFrame({column: list(branches[conversion_track[column]]) for column in features_track})
            df_sig_track = unnest(df_sig_track, features_track)
            for event in range(event_num):
                df_sig_temp = pd.concat([df_sig_track.loc[event], df_sig_tower.loc[event]], sort=False).fillna(0)
                df_sig_temp = df_sig_temp.sort_values("pt", ascending=False).head(particle_num)
                arr_sig_temp = df_sig_temp.values.flatten('F')
                arr_sig_to_concat_temp.append(arr_sig_temp)
            arr_sig_temp = np.vstack(arr_sig_to_concat_temp)
            df_sig_temp = pd.DataFrame(arr_sig_temp, columns=part_features)
            for column in ss + weight:
                df_sig_temp[column] = np.array(branches[column]).reshape(-1, 1)
            df_sig_temp['label'] = 1
            df_sig_temp = df_sig_temp[columns]
            pt_col = df_sig_temp[weight[0]].values.reshape(-1, 1)
            mass_col = df_sig_temp[weight[1]].values.reshape(-1, 1)
            df_sig_temp = df_sig_temp[np.logical_and(np.logical_and(np.greater(pt_col, pt_range[0]), np.less(pt_col, pt_range[1])), np.logical_and(np.greater(mass_col, mass_range[0]), np.less(mass_col, mass_range[1])))]
            df_sig_to_concat.append(df_sig_temp)
    df_sig = pd.concat(df_sig_to_concat)

    # Calculates the distribution of the signal

    sig_hist, _x, _y = np.histogram2d(df_sig[weight[0]], df_sig[weight[1]], bins=20,
                                      range=np.array([pt_range, mass_range]))
    print(sig_hist)
    print(np.sum(sig_hist))
    
    # Creates the background data frame

    df_remade_bkg = pd.DataFrame(columns=columns)
    df_bkg_to_concat = []
    for bkg in iFiles_bkg:
        file_list = os.listdir(payload['samples'][bkg])
        print(len(file_list))
        for i in range(file_num_bkg):
            print(i)
            print(file_num_sig)
            data_set = payload['samples'][bkg]+file_list[i]
            arr_bkg_to_concat_temp = []
            file1 = uproot.open(data_set)
            tree = file1['tree']
            branches = tree.arrays()
            event_num = len(branches['jet_pt'])
            print('event num')
            print(event_num)
            df_bkg_tower = pd.DataFrame({column: list(branches[conversion_tower[column]]) for column in features_tower})
            df_bkg_tower = unnest(df_bkg_tower, features_tower)
            df_bkg_track = pd.DataFrame({column: list(branches[conversion_track[column]]) for column in features_track})
            df_bkg_track = unnest(df_bkg_track, features_track)
            for event in range(event_num):
                df_bkg_temp = pd.concat([df_bkg_track.loc[event], df_bkg_tower.loc[event]], sort=False).fillna(0)
                df_bkg_temp = df_bkg_temp.sort_values("pt", ascending=False).head(particle_num)
                arr_bkg_temp = df_bkg_temp.values.flatten('F')
                arr_bkg_to_concat_temp.append(arr_bkg_temp)
            arr_bkg_temp = np.vstack(arr_bkg_to_concat_temp)
            df_bkg_temp = pd.DataFrame(arr_bkg_temp, columns=part_features)
            for column in ss + weight:
                df_bkg_temp[column] = np.array(branches[column]).reshape(-1, 1)
            df_bkg_temp['label'] = 0
            df_bkg_temp = df_bkg_temp[columns]
            df_bkg_to_concat.append(df_bkg_temp)
    df_bkg = pd.concat(df_bkg_to_concat)

    # Adds background based on signal distribution until fill factor is reached

    for ix in range(len(_x) - 1):
        print(len(_x))
        for iy in range(len(_y) - 1):
            new_df_bkg = df_bkg[((df_bkg[weight[0]] >= _x[ix]) & (df_bkg[weight[0]] < _x[ix + 1]) & (
                                df_bkg[weight[1]] >= _y[iy]) & (df_bkg[weight[1]] < _y[iy + 1]))]
            df_remade_bkg = pd.concat([df_remade_bkg, new_df_bkg.sample(n=min(int(int(sig_hist[ix, iy]) * fill_factor), len(new_df_bkg)))], ignore_index=True)

    # Shows fill factor per bin

    bkg_hist, _, _ = np.histogram2d(df_remade_bkg[weight[0]], df_remade_bkg[weight[1]], bins=20,
                                    range=np.array([[200., 800.], [40., 240.]]))
    print(np.nan_to_num(np.divide(bkg_hist, sig_hist)))

    # Merges data frames

    merged_df = pd.concat([df_sig, df_remade_bkg]).astype('float32')

    # Creates output file

    merged_df = merged_df[columns]
    final_df = merged_df[~(np.sum(np.isinf(merged_df.values), axis=1) > 0)]
    print(list(final_df.columns))
    arr = final_df.values
    print(arr.shape)

    # Open HDF5 file and write dataset

    h5_file = h5py.File(iFile_out, 'w')
    h5_file.create_dataset('deepDoubleQ', data=arr, compression='lzf')
    h5_file.close()
    del h5_file


# Remakes data sets
remake(signal_list, background_list, output_name)
