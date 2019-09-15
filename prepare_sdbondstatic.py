# -*- coding: utf-8 -*-
import os
import numpy
import pickle
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


os.chdir(os.path.dirname(__file__))

model_training_input    = '../model_input/training_input/'

client = MongoClient('10.198.37.101', 28181, unicode_decode_error_handler='ignore')
client.admin.authenticate('svc.mongoraap.001', 'SCBpassword1', mechanism = 'PLAIN', source='$external')
db_name                 = client['ROBADEV']
collection_detailbonds  = db_name['pre_sdb']
result_sdbondstatic     = collection_detailbonds.find()
df_sdbondstatic         = pd.DataFrame(list(result_sdbondstatic))

df_sdbondstatic.to_csv(model_training_input+'sdbondstatic_original.csv',index=False)

df_sdbondstatic         = df_sdbondstatic[df_sdbondstatic['nickname'] != '']
df_sdbondstatic         = df_sdbondstatic[df_sdbondstatic['issuer_ticker'] != '']
df_sdbondstatic         = df_sdbondstatic[df_sdbondstatic['bbg_industry_group'] != '']
df_sdbondstatic         = df_sdbondstatic[df_sdbondstatic['bbg_industry_sector'] != '']

df_sdbondstatic         = df_sdbondstatic.apply(lambda x: x.str.upper() if (x.dtype == 'object') else x)
df_sdbondstatic         = df_sdbondstatic.apply(lambda x: x.str.strip() if (x.dtype == 'object') else x)



df_sdbondstatic.loc[df_sdbondstatic['basel_classification']   == 'DATA NOT AVAILABLE', 'basel_classification']    = 'N/A'
df_sdbondstatic.loc[df_sdbondstatic['distribution_type']      == 'DATA NOT AVAILABLE', 'distribution_type']       = 'N/A'
df_sdbondstatic.loc[df_sdbondstatic['distribution_type']      == 'NOT DEFINED', 'distribution_type']              = 'N/A'
df_sdbondstatic.loc[df_sdbondstatic['distribution_type']      == '144A', 'distribution_type']                     = 'SEC_REGISTERED'
df_sdbondstatic.loc[df_sdbondstatic['maturity_type']          == 'HYBRID', 'maturity_type']                       = 'PERP'
df_sdbondstatic.loc[df_sdbondstatic['ranking']                == '', 'ranking']                                   = 'SENIOR'
df_sdbondstatic.loc[df_sdbondstatic['rating_type']            == 'UNRATED', 'rating_type']                        = 'N/A'
df_sdbondstatic.loc[df_sdbondstatic['type_le']                == 'SUPRA', 'type_le']                              = 'SOV'
df_sdbondstatic.loc[df_sdbondstatic['issuer_rating']          == 'NR', 'issuer_rating']                           = 'N/A'
df_sdbondstatic.loc[df_sdbondstatic['issuer_rating']          == 'DATA NOT AVAILABLE', 'issuer_rating']           = 'N/A'


num_cols = df_sdbondstatic.select_dtypes(include=['float64','int64']).columns
str_cols = df_sdbondstatic.select_dtypes(include=['object']).columns
df_sdbondstatic.loc[:, num_cols] = df_sdbondstatic.loc[:, num_cols].replace(r'^\s*$', 0.0, regex=True)
df_sdbondstatic.loc[:, str_cols] = df_sdbondstatic.loc[:, str_cols].replace(r'^\s*$', 'N/A', regex=True)

df_sdbondstatic                  = df_sdbondstatic.loc[df_sdbondstatic['res_maturity'] > 0]
#df_sdbondstatic['res_maturity']  = numpy.log2(df_sdbondstatic['res_maturity'])


def map_rating (row):
    if row['issuer_rating'] == 'AAA':
        return 0
    elif row['issuer_rating'] == 'AA':
        return 0
    elif row['issuer_rating'] == 'A':
        return 1
    elif row['issuer_rating'] == 'BBB':
        return 2
    elif row['issuer_rating'] == 'BB':
        return 3
    elif row['issuer_rating'] == 'B':
        return 4
    elif row['issuer_rating'] == 'CCC':
        return 5
    else:
        return 6

df_sdbondstatic['rating_map'] = df_sdbondstatic.apply(lambda row: map_rating(row), axis=1)



def map_product (row):
    if row['rating_type'] == 'IG':
        return 0
    elif row['rating_type'] == 'CROSSOVER':
        return 1
    elif row['rating_type'] == 'HY':
        return 2
    else:
        return 3
  
df_sdbondstatic['product_map'] = df_sdbondstatic.apply(lambda row: map_product(row), axis=1)



def map_fi_corp_sov (row):
    if row['type_le'] == 'SOV':
        return 0
    elif row['type_le'] == 'SUPRA':
        return 0
    elif row['type_le'] == 'QUASI':
        return 1
    elif row['type_le'] == 'FI':
        return 2
    else:
        return 3
   
df_sdbondstatic['fi_corp_sov_map'] = df_sdbondstatic.apply(lambda row: map_fi_corp_sov(row), axis=1)

df_sdbondstatic = df_sdbondstatic.reset_index(drop=True)

cols_encoding = ['_id',
                'nickname',
                'issuer_ticker',
                      
                'res_maturity',
                'product_map',	
                'rating_map',	
                'fi_corp_sov_map',
                               
                'currency',
                
                'basel_classification',
                'bbg_industry_group',
                'bbg_industry_sector',
                'coupon_type',
                'country',
                'region',
                'ranking',
                'special_classification',
                'maturity_type']


df_similarity = df_sdbondstatic[cols_encoding]
df_similarity = df_similarity.rename(columns={"res_maturity": "bond_age"})


# One-hot encoding 
ohe_encoder     = OneHotEncoder(handle_unknown ='ignore')
df_cat          = df_similarity.iloc[ : , 7:]
ohe_encoder.fit(df_cat)

with open(model_training_input+'sdbondstatic_ohe.pickle', 'wb') as handle:
    pickle.dump(ohe_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL) 

ohe_encoded         = ohe_encoder.transform(df_cat).toarray()
df_ohe              = pd.DataFrame(ohe_encoded)
df_ohe.columns      = list(ohe_encoder.get_feature_names())

df_one_hot_encoded  = pd.concat([df_similarity.iloc[ : , :7],  df_ohe], axis=1)
df_one_hot_encoded  = df_one_hot_encoded.drop_duplicates()
df_one_hot_encoded  = df_one_hot_encoded.reset_index(drop=True)



# Min-max scaling
cols_scale  = df_one_hot_encoded.columns[4:]
scaler      = MinMaxScaler()

df_scaled   = scaler.fit_transform(df_one_hot_encoded[cols_scale])

with open(model_training_input+'sdbondstatic_mms.pickle', 'wb') as handle:
    pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL) 

df_scaled               = pd.DataFrame(df_scaled, columns=cols_scale)
df_final                = pd.concat([df_one_hot_encoded.iloc[: ,:4], df_scaled], axis=1)



numeric_scale_1 = 0.8
df_final['bond_age']        = df_final['bond_age'] * numeric_scale_1

numeric_scale_2 = 5.0
df_final['rating_map']      = df_final['rating_map'] * numeric_scale_2

numeric_scale_3 = 3.0
df_final['product_map']     = df_final['product_map'] * numeric_scale_3

numeric_scale_4 = 1.5
df_final['fi_corp_sov_map'] = df_final['fi_corp_sov_map'] * numeric_scale_4


df_final.to_csv(model_training_input+'sdbondstatic_input.csv',index=False)



















