# -*- coding: utf-8 -*-
import os
import numpy
import pickle
import datetime
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


os.chdir(os.path.dirname(__file__))

model_training_input = '../model_input/training_input/'

client = MongoClient('10.198.37.101', 28181, unicode_decode_error_handler='ignore')
client.admin.authenticate('svc.mongoraap.001', 'SCBpassword1', mechanism = 'PLAIN', source='$external')
db_name = client['ROBADEV']
collection_detailbonds = db_name['detailbonds_pa']
result_detailbonds = collection_detailbonds.find()
df_detailbonds = pd.DataFrame(list(result_detailbonds))


def create_day_diff(row):
    if len(row['issue_date']) == 10:
        year    = int(row['issue_date'].split('-')[0])
        month   = int(row['issue_date'].split('-')[1])
        day     = int(row['issue_date'].split('-')[2])

        issue_day = datetime.date(year, month, day)
        today     = datetime.date.today()
        delta     = today - issue_day
        return delta.days
    else:
        return 0

df_detailbonds['day_diff'] = df_detailbonds.apply(lambda row: create_day_diff(row), axis=1)

df_detailbonds  = df_detailbonds[df_detailbonds['investor_name'] != '']
df_detailbonds  = df_detailbonds.apply(lambda x: x.str.upper() if (x.dtype == 'object') else x)
df_detailbonds  = df_detailbonds.apply(lambda x: x.str.strip() if (x.dtype == 'object') else x)
df_exclude      = pd.read_csv(model_training_input + 'InvestorExclusionList.csv', encoding = "ISO-8859-1", keep_default_na=False)
df_exclude      = df_exclude.apply(lambda x: x.astype(str).str.upper())
df_exclude      = df_exclude.apply(lambda x: x.astype(str).str.strip())
df_detailbonds  = df_detailbonds.loc[~df_detailbonds['investor_name'].isin(df_exclude['investor_name'])]

df_detailbonds.to_csv(model_training_input + 'detailbonds_orginal.csv',index=False)



df_detailbonds.loc[df_detailbonds['basel_classification']   == 'DATA NOT AVAILABLE', 'basel_classification']    = 'N/A'
df_detailbonds.loc[df_detailbonds['distribution_type']      == 'DATA NOT AVAILABLE', 'distribution_type']       = 'N/A'
df_detailbonds.loc[df_detailbonds['distribution_type']      == 'NOT DEFINED', 'distribution_type']              = 'N/A'
df_detailbonds.loc[df_detailbonds['distribution_type']      == '144A', 'distribution_type']                     = 'SEC_REGISTERED'
df_detailbonds.loc[df_detailbonds['fi_corp_sov']            == 'SUPRA', 'fi_corp_sov']                          = 'SOV'
df_detailbonds.loc[df_detailbonds['rating']                 == 'NR', 'rating']                                  = 'N/A'
df_detailbonds.loc[df_detailbonds['rating']                 == 'DATA NOT AVAILABLE', 'rating']                  = 'N/A'
df_detailbonds.loc[df_detailbonds['product']                == 'UNRATED', 'product']                            = 'N/A'
df_detailbonds.loc[df_detailbonds['maturity_type']          == 'HYBRID', 'maturity_type']                       = 'PERP'
df_detailbonds.loc[df_detailbonds['ranking_name']           == '', 'ranking_name']                              = 'SENIOR'
df_detailbonds.loc[df_detailbonds['investor_country']       == 'DATA NOT AVAILABLE', 'investor_country']        = 'N/A'
df_detailbonds.loc[df_detailbonds['investor_country']       == 'UNDEFINED', 'investor_country']                 = 'N/A'


num_cols = df_detailbonds.select_dtypes(include=['float64','int64']).columns
str_cols = df_detailbonds.select_dtypes(include=['object']).columns
df_detailbonds.loc[:, num_cols] = df_detailbonds.loc[:, num_cols].replace(r'^\s*$', 0.0, regex=True)
df_detailbonds.loc[:, str_cols] = df_detailbonds.loc[:, str_cols].replace(r'^\s*$', 'N/A', regex=True)


def map_rating (row):
    if row['rating'] == 'AAA':
        return 0
    elif row['rating'] == 'AA':
        return 0
    elif row['rating'] == 'A':
        return 1
    elif row['rating'] == 'BBB':
        return 2
    elif row['rating'] == 'BB':
        return 3
    elif row['rating'] == 'B':
        return 4
    elif row['rating'] == 'CCC':
        return 5
    else:
        return 6

df_detailbonds['rating_map'] = df_detailbonds.apply(lambda row: map_rating(row), axis=1)


def map_product (row):
    if row['product'] == 'IG':
        return 0
    elif row['product'] == 'CROSSOVER':
        return 1
    elif row['product'] == 'HY':
        return 2
    else:
        return 3
  
df_detailbonds['product_map'] = df_detailbonds.apply(lambda row: map_product(row), axis=1)


def map_fi_corp_sov (row):
    if row['fi_corp_sov'] == 'SOV':
        return 0
    elif row['fi_corp_sov'] == 'SUPRA':
        return 0
    elif row['fi_corp_sov'] == 'QUASI':
        return 1
    elif row['fi_corp_sov'] == 'FI':
        return 2
    else:
        return 3
   
df_detailbonds['fi_corp_sov_map']   = df_detailbonds.apply(lambda row: map_fi_corp_sov(row), axis=1)

df_detailbonds['security']          = df_detailbonds['security'] + '-' + df_detailbonds['currency']

cols_investor_retrieve = [
                         'investor_name',
                         'deal_size_mil',
                         'firm',
                         'allocated',
                         'issue_date',
                         'security',
                         'investor_country',
                         'investor_type'
                         ]

df_all_trans = df_detailbonds[cols_investor_retrieve]

df_all_trans.to_csv(model_training_input+'detailbonds_all_transactions.csv', index=False, na_rep='N/A')


df_detailbonds = df_detailbonds.reset_index(drop=True)

df_detailbonds = df_detailbonds[df_detailbonds['bond_age'] > 0]
df_detailbonds['bond_age'] = numpy.log2(df_detailbonds['bond_age'])


cols_encoding = ['security',
                      
                'bond_age',
                 #'deal_size_mil',
                 #'sbn_coupon',
                'product_map',	
                'rating_map',	
                'fi_corp_sov_map',
               
                'issuer_ticker',
                'currency',
                
                'basel_classification',
                'bbg_industry_group',
                'bbg_industry_sector',
                'coupon',
                'distribution_type',
                'country',
                'region',
                'ranking_name',
                'special_caracters',
                'maturity_type']


df_similarity = df_detailbonds[cols_encoding]


# One-hot encoding 
ohe_encoder     = OneHotEncoder(handle_unknown ='ignore')
df_cat          = df_similarity.iloc[ : , 5:]
ohe_encoder.fit(df_cat)

with open(model_training_input+'detailbonds_ohe.pickle', 'wb') as handle:
    pickle.dump(ohe_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL) 

ohe_encoded         = ohe_encoder.transform(df_cat).toarray()
df_ohe              = pd.DataFrame(ohe_encoded)
df_ohe.columns      = list(ohe_encoder.get_feature_names())

df_one_hot_encoded  = pd.concat([df_similarity.iloc[ : , :5],  df_ohe], axis=1)
df_one_hot_encoded  = df_one_hot_encoded.drop_duplicates()
df_one_hot_encoded  = df_one_hot_encoded.reset_index(drop=True)



# Min-max scaling
cols_scale  = df_one_hot_encoded.columns[1:]
scaler      = MinMaxScaler()

df_scaled   = scaler.fit_transform(df_one_hot_encoded[cols_scale])

with open(model_training_input+'detailbonds_mms.pickle', 'wb') as handle:
    pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL) 

df_scaled   = pd.DataFrame(df_scaled, columns=cols_scale)
df_final    = pd.concat([df_one_hot_encoded.iloc[: ,:1], df_scaled], axis=1)



numeric_scale_1 = 5.0
df_final['bond_age']        = df_final['bond_age'] * numeric_scale_1

numeric_scale_2 = 3.0
df_final['fi_corp_sov_map'] = df_final['fi_corp_sov_map'] * numeric_scale_2

numeric_scale_3 = 3.0
df_final['rating_map']      = df_final['rating_map'] * numeric_scale_3

numeric_scale_4 = 3.0
df_final['product_map']     = df_final['product_map'] * numeric_scale_4




df_final.to_csv(model_training_input+'detailbonds_unique_bond.csv',index=False)
























