import pandas as pd
import numpy as np

# defining file paths

feat_file = '../DR5_features/features_DR5_SDSSDR14_QSO_noblazar_gband_all_final_sample_specprops.csv'

det_file = '../DR5_data/DR5_final_sample_lcs_gband.parquet'

out_feat_file = '../DR5_features/balanced_inndet_inprop_features_DR5_SDSSDR14_QSO_noblazar_gband_all_final_sample_specprops.csv'

out_det_file = '../DR5_data/balanced_inndet_inprop_DR5_SDSSDR14_QSO_noblazar_gband.parquet'

lc_out_numpy_file = '../DR5_data/balanced_inndet_inprop_DR5_SDSSDR14_QSO_noblazar_gband_lcs_list.npy'

oid_out_numpy_file = '../DR5_data/balanced_inndet_inprop_DR5_SDSSDR14_QSO_noblazar_gband_oid_list.npy'

#defining grid (Lbol vs BHmass)

binsize = 0.10

#min_Lbol = 45.05
#max_Lbol = 47.251
#Lbol = np.arange(min_Lbol, max_Lbol, binsize)


min_ndet = 50
max_ndet = 976
ndet = np.arange(min_ndet, max_ndet, 20)

# features file

df_feats = pd.read_csv(feat_file)

#df_feats_large = df_feats[(df_feats.n_good_det>275)]

#df_feats = df_feats[(df_feats.n_good_det<=275)]

df_balanced = df_feats[(df_feats.LOG_LBOL<44.9) | (df_feats.LOG_LBOL>=47.4)]

df_feats = df_feats[(df_feats.LOG_LBOL>=44.9) & (df_feats.LOG_LBOL<47.4)]


for k in range(len(ndet)-1):

    print("range: ", ndet[k], ndet[k+1])

    ndet_df = df_feats[(df_feats.n_good_det>=ndet[k]) & (df_feats.n_good_det<ndet[k+1])]

    min_Lbol = ndet_df.LOG_LBOL.min()
    max_Lbol = ndet_df.LOG_LBOL.max()


    try:
        Lbol = np.arange(min_Lbol, max_Lbol, binsize)


        for i in range(len(Lbol)-1):
            print("range: ", Lbol[i], Lbol[i+1])
            sel_df = ndet_df[(ndet_df.LOG_LBOL>=Lbol[i]) & (ndet_df.LOG_LBOL<Lbol[i+1])]

            min_BHmass = sel_df.LOG_MBH.min()
            max_BHmass = sel_df.LOG_MBH.max()


            try:

                BHmass = np.arange(min_BHmass, max_BHmass, binsize)

                print(BHmass)

                for j in range(len(BHmass)-1):
                    print("range: ", BHmass[j], BHmass[j+1])

                    sel_df2 = sel_df[(sel_df.LOG_MBH>=BHmass[j]) & (sel_df.LOG_MBH<BHmass[j+1])]
                    print(len(sel_df2.LOG_MBH.values))
                    try: random_df = sel_df2.sample(n=2, replace=False)
                    except: random_df = sel_df2
                    print(len(random_df.LOG_MBH.values))
                    df_balanced = df_balanced.append(random_df, ignore_index=True)

            except: print("could not create BHmass bins")

    except: print("could not create Lbol bins")

#df_balanced = df_balanced.append(df_feats_large, ignore_index=True)

print("final balanced sample size: ", len(df_balanced.LOG_MBH.values))

df_balanced.to_csv(out_feat_file)

#getting balanced detection sample

df_det = pd.read_parquet(det_file)

df_det_out = df_det.loc[df_det.oid_alerce.isin(df_balanced.oid_alerce.values)]

df_det_out.to_parquet(out_det_file)
print("saved ", out_det_file)

# saving the data as a list of arrays (to save time when using Colab)
group = df_det_out.groupby(['oid_alerce'])

lcs = [np.r_[group.get_group(i)[['mjd','mag','magerr']].values.tolist()] for i in list(group.groups)]

oids = [np.r_[group.get_group(i)[['oid_alerce']].values.tolist()[0]] for i in list(group.groups)]


np.save(lc_out_numpy_file, lcs)

np.save(oid_out_numpy_file, oids)
