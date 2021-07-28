import pandas as pd
import numpy as np

det_file = '../DR5_data/DR5_final_sample_lcs_gband.parquet'
#feat_file = '../features/features_milliquas_gband_filtered_sample_pointsources.csv'
#oid_file = '../DR5_data/oids_anomalies.csv'

det_numpy_file = '../DR5_data/DR5_final_sample_lcs_gband_lcs_list.npy'
oid_numpy_file = '../DR5_data/DR5_final_sample_lcs_gband_oid_list.npy'

df_det = pd.read_parquet(det_file)
#oids_list = pd.read_csv(oid_file)

#df_det = df_det.loc[df_det.oid_alerce.isin(oids_list.oid_alerce.values)]

df_det = df_det.sort_values('mjd')
df_det = df_det.drop_duplicates(['oid_alerce', 'mjd'])

group = df_det.groupby(['oid_alerce'])

lcs = [np.r_[group.get_group(i)[['mjd','mag','magerr']].values.tolist()] for i in list(group.groups)]
oids = [np.r_[group.get_group(i)[['oid_alerce']].values.tolist()[0]] for i in list(group.groups)]

np.save(det_numpy_file, lcs)

np.save(oid_numpy_file, oids)
