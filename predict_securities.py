# -*- coding: utf-8 -*-
import os
import sys
import numpy
import pickle
import pandas as pd
import datetime 
from sklearn.preprocessing import MinMaxScaler

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

#user_input = sys.argv[1]

os.chdir(os.path.dirname(__file__))

model_training_input    = '../model_input/training_input/'
user_input              = '../model_input/user_input/STANLN.csv'
model_output            = '../model_output/predicted_securities/'

df_user_input           = pd.read_csv(user_input, delimiter = '|',encoding = "ISO-8859-1", keep_default_na=False)
df_user_input           = df_user_input.apply(lambda x: x.astype(str).str.upper())
df_user_input           = df_user_input.apply(lambda x: x.astype(str).str.strip())


df_user_input.loc[df_user_input['Value'] == 'DATA NOT AVAILABLE', 'Value']  = 'N/A'
df_user_input.loc[df_user_input['Value'] == 'UNDEFINED', 'Value']           = 'N/A'
df_user_input.loc[df_user_input['Value'] == 'UNRATED', 'Value']             = 'N/A'
df_user_input.loc[df_user_input['Value'] == 'NOT DEFINED', 'Value']         = 'N/A'
df_user_input.loc[df_user_input['Value'] == '', 'Value']                    = 'N/A'
df_user_input.loc[df_user_input['Value'] == 'HYBRID', 'Value']              = 'PERP'
df_user_input.loc[df_user_input['Value'] == '144A', 'Value']                = 'SEC_REGISTERED'
df_user_input.loc[df_user_input['Field'].eq('ISSUER RATINGS') & df_user_input['Value'].eq('NR'), 'Value'] = 'N/A'

#print (df_user_input)

dict_user_input         = df_user_input.set_index('Field')['Value'].to_dict()

BASEL_CLASSIFICATION    =  dict_user_input['BASEL CLASSIFICATION'].strip()
#COUPON_RATE             =  float(dict_user_input['COUPON RATE'].strip())
COUPON_TYPE             =  dict_user_input['COUPON TYPE'].strip()
CURRENCY                =  dict_user_input['CURRENCY'].strip()
#DEAL_SIZE_MIL           =  float(dict_user_input['DEAL SIZE MIL'].strip())
#DISTRIBUTION_TYPE       =  dict_user_input['DISTRIBUTION TYPE'].strip()
INDUSTRY                =  dict_user_input['INDUSTRY'].strip()
INDUSTRY_SECTOR         =  dict_user_input['INDUSTRY SECTOR'].strip()
ISSUER_COUNTRY          =  dict_user_input['ISSUER COUNTRY'].strip()
ISSUER_RATINGS          =  dict_user_input['ISSUER RATINGS'].strip()
ISSUER_REGION           =  dict_user_input['ISSUER REGION'].strip()
ISSUER_TICKER           =  dict_user_input['ISSUER TICKER'].strip()
ISSUER_TYPE             =  dict_user_input['ISSUER TYPE'].strip()
MATURITY_TYPE           =  dict_user_input['MATURITY TYPE'].strip()
RANKING                 =  dict_user_input['RANKING'].strip()
RATING_TYPE             =  dict_user_input['RATING TYPE'].strip()
SPECIAL_CLASSIFICATION  =  dict_user_input['SPECIAL CLASSIFICATION'].strip()
TENOR                   =  float(dict_user_input['TENOR'].strip())
#FLOAT_INDEX             =  dict_user_input['FLOAT INDEX'].strip()

user_input_cat = []
user_input_num = []


if TENOR <= 0:
    TENOR = 0.1
#user_input_num.append(numpy.log2(TENOR))
#user_input_num.append(TENOR)
#user_input_num.append(DEAL_SIZE_MIL)
#user_input_num.append(COUPON_RATE)


def map_product(product):
    if product.upper()      == 'IG':
        return 0
    elif product.upper()    == 'CROSSOVER':
        return 1
    elif product.upper()    == 'HY':
        return 2
    else:
        return 3

new_RATING_TYPE = map_product(RATING_TYPE.upper())
user_input_num.append(new_RATING_TYPE)


def map_rating(rating):
    if rating.upper()       == 'AAA':
        return 0
    elif rating.upper()     == 'AA':
        return 0
    elif rating.upper()     == 'A':
        return 1
    elif rating.upper()     == 'BBB':
        return 2
    elif rating.upper()     == 'BB':
        return 3
    elif rating.upper()     == 'B':
        return 4
    elif rating.upper()     == 'CCC':
        return 5
    else:
        return 6

new_ISSUER_RATINGS = map_rating(ISSUER_RATINGS.upper())
user_input_num.append(new_ISSUER_RATINGS)


def map_fi_corp_sov(fi_corp_sov):
    if fi_corp_sov.upper()      == 'SOV':
        return 0
    elif fi_corp_sov.upper()    == 'SUPRA':
        return 0
    elif fi_corp_sov.upper()    == 'QUASI':
        return 1
    elif fi_corp_sov.upper()    == 'FI':
        return 2
    else:
        return 3

new_ISSUER_TYPE = map_fi_corp_sov(ISSUER_TYPE.upper())
user_input_num.append(new_ISSUER_TYPE)


user_input_cat.append(CURRENCY.upper())
user_input_cat.append(BASEL_CLASSIFICATION.upper())
user_input_cat.append(INDUSTRY.upper())
user_input_cat.append(INDUSTRY_SECTOR.upper())
user_input_cat.append(COUPON_TYPE.upper())
#user_input_cat.append(DISTRIBUTION_TYPE.upper())
user_input_cat.append(ISSUER_COUNTRY.upper())
user_input_cat.append(ISSUER_REGION.upper())
user_input_cat.append(RANKING.upper())
user_input_cat.append(SPECIAL_CLASSIFICATION.upper())
user_input_cat.append(MATURITY_TYPE.upper())


user_input_cat_2D =[]
user_input_cat_2D.append(user_input_cat)


with open(model_training_input+'sdbondstatic_ohe.pickle', 'rb') as handle:
    ohe = pickle.load(handle) 

ohe_encoded = ohe.transform(user_input_cat_2D).toarray()
ohe_user_input = user_input_num + ohe_encoded.tolist()[0]


with open(model_training_input+'sdbondstatic_mms.pickle', 'rb') as handle:
    mms = pickle.load(handle) 

user_input_scaled                   = mms.transform([ohe_user_input])
user_input_scaled_tenor             = numpy.insert(user_input_scaled[0],0,TENOR)
user_input_scaled_tenor_list        = user_input_scaled_tenor.tolist()
user_input_scaled_tenor_list_2D     = [user_input_scaled_tenor_list] 



df_sdbondstatic             = pd.read_csv(model_training_input+'sdbondstatic_input.csv', keep_default_na=False)
dict_nickname               = df_sdbondstatic.set_index('_id')['nickname'].to_dict()
dict_ticker                 = df_sdbondstatic.set_index('_id')['issuer_ticker'].to_dict()
df_final_user_input         = pd.DataFrame(user_input_scaled_tenor_list_2D)  
df_final_user_input.columns = df_sdbondstatic.columns[3:]



numeric_scale_1 = 0.8
df_final_user_input['bond_age']        = df_final_user_input['bond_age'] * numeric_scale_1

numeric_scale_2 = 5.0
df_final_user_input['rating_map']      = df_final_user_input['rating_map'] * numeric_scale_2

numeric_scale_3 = 3.0
df_final_user_input['product_map']     = df_final_user_input['product_map'] * numeric_scale_3

numeric_scale_4 = 1.5
df_final_user_input['fi_corp_sov_map'] = df_final_user_input['fi_corp_sov_map'] * numeric_scale_4



#==============================================================================
# header_important  = []
# header_medium     = []
# header_low        = []
# 
# header_important  = list(set(df_user_input.loc[df_user_input['Importance'] == 'HIGH']['Field']))
# header_medium     = list(set(df_user_input.loc[df_user_input['Importance'] == 'MEDIUM']['Field']))
# header_low        = list(set(df_user_input.loc[df_user_input['Importance'] == 'LOW']['Field']))
#==============================================================================


header_important = ['TENOR', 'CURRENCY', 'ISSUER COUNTRY', 'ISSUER RATINGS']
header_medium    = ['ISSUER REGION', 'BASEL CLASSIFICATION', 'ISSUER TYPE', 'SPECIAL CLASSIFICATION', 'RATING TYPE', 'INDUSTRY', 'COUPON TYPE', 'INDUSTRY SECTOR']
header_low       = ['RANKING', 'MATURITY TYPE']


dict_col_header = {
                    'bond':'TENOR',
                    'deal':'DEAL SIZE MIL',
                    'sbn':'COUPON RATE',
                    'product':'RATING TYPE',
                    'rating':'ISSUER RATINGS',
                    'fi':'ISSUER TYPE',
                    'x0':'CURRENCY',
                    'x1':'BASEL CLASSIFICATION',
                    'x2':'INDUSTRY',
                    'x3':'INDUSTRY SECTOR',
                    'x4':'COUPON TYPE',
                     #'x5':'DISTRIBUTION TYPE',
                    'x5':'ISSUER COUNTRY',
                    'x6':'ISSUER REGION',
                    'x7':'RANKING',
                    'x8':'SPECIAL CLASSIFICATION',
                    'x9':'MATURITY TYPE'
                   }


cols_important  = []
cols_medium     = []
cols_low        = []

for each_col in df_sdbondstatic.columns[3:]:
    header_name = dict_col_header[each_col.lower().split('_')[0].strip()]
    header_name = header_name.strip()
    
    if header_name == 'BASEL CLASSIFICATION' and ISSUER_TYPE.upper() != 'FI':
        continue
     
    if header_name in header_important:
        cols_important.append(each_col)
    elif header_name in header_medium:
        cols_medium.append(each_col)
    else:
        cols_low.append(each_col)
    



first   = False
second  = False
third   = False

def ticker_weight(row):
    _id     = row['_id']
    ticker  = dict_ticker[_id.strip()]
    if ticker.strip().upper() != ISSUER_TICKER.upper():
        return 0.5
    else:
        return 0 


def first_similarity (row):
   one_row = numpy.array(row[1: ])
   dist = numpy.linalg.norm(one_row - df_final_user_input_first)
   return round(dist,8)       

if len(cols_important) != 0:
    first = True
    df_final_user_input_first                = df_final_user_input[cols_important]
    df_final_user_input_first                = numpy.array(df_final_user_input_first.iloc[0 , : ])
    df_sdbondstatic_first                    = df_sdbondstatic[['_id'] + cols_important]
    df_sdbondstatic_first['similarity_1st']  = df_sdbondstatic_first.apply(lambda row: first_similarity(row), axis=1)
    df_similarity_first                      = df_sdbondstatic_first[['_id','similarity_1st']]
    
    if 'ISSUER TICKER' in header_important:
        df_similarity_first['ticker_dist_square']    = df_similarity_first.apply(lambda row: ticker_weight(row), axis=1)
        df_similarity_first['similarity_1st']        = (df_similarity_first['similarity_1st']*df_similarity_first['similarity_1st'] + df_similarity_first['ticker_dist_square']) ** 0.5
        df_similarity_first                          = df_similarity_first[['_id','similarity_1st']]
    
elif 'ISSUER TICKER' in header_important:
    first = True
    df_sdbondstatic_first                            = df_sdbondstatic[['_id']]
    df_sdbondstatic_first['similarity_1st']          = df_sdbondstatic_first.apply(lambda row: ticker_weight(row), axis=1)
    df_sdbondstatic_first['similarity_1st']          = df_sdbondstatic_first['similarity_1st'] ** 0.5
    df_similarity_first                              = df_sdbondstatic_first[['_id','similarity_1st']]

#df_similarity_first   = df_similarity_first.sort_values(by='similarity_1st') 


   
def second_similarity (row):
   one_row = numpy.array(row[1: ])
   dist = numpy.linalg.norm(one_row - df_final_user_input_second)
   return round(dist,8) 

if (len(cols_medium) != 0):
    second = True
    df_final_user_input_second                = df_final_user_input[cols_medium]
    df_final_user_input_second                = numpy.array(df_final_user_input_second.iloc[0 , : ])
    df_sdbondstatic_second                    = df_sdbondstatic[['_id'] + cols_medium]
    df_sdbondstatic_second['similarity_2nd']  = df_sdbondstatic_second.apply (lambda row: second_similarity(row), axis=1)
    df_similarity_second                      = df_sdbondstatic_second[['_id','similarity_2nd']]
    
    if 'ISSUER TICKER' in header_medium:
        df_similarity_second['ticker_dist_square']    = df_similarity_second.apply(lambda row: ticker_weight(row), axis=1)
        df_similarity_second['similarity_2nd']        = (df_similarity_second['similarity_2nd']*df_similarity_second['similarity_2nd'] + df_similarity_second['ticker_dist_square']) ** 0.5
        df_similarity_second                          = df_similarity_second[['_id','similarity_2nd']]
    
elif 'ISSUER TICKER' in header_medium:
    second = True
    df_sdbondstatic_second                            = df_sdbondstatic[['_id']]
    df_sdbondstatic_second['similarity_2nd']          = df_sdbondstatic_second.apply(lambda row: ticker_weight(row), axis=1)
    df_sdbondstatic_second['similarity_2nd']          = df_sdbondstatic_second['similarity_2nd'] ** 0.5
    df_similarity_second                              = df_sdbondstatic_second[['_id','similarity_2nd']]
    
#df_similarity_second   = df_similarity_second.sort_values(by='similarity_2nd')



def third_similarity (row):
   one_row = numpy.array(row[1: ])
   dist = numpy.linalg.norm(one_row - df_final_user_input_third)
   return round(dist,8)
       
if (len(cols_low) != 0):
    third = True
    df_final_user_input_third                = df_final_user_input[cols_low]
    df_final_user_input_third                = numpy.array(df_final_user_input_third.iloc[0 , : ])
    df_sdbondstatic_third                    = df_sdbondstatic[['_id'] + cols_low]
    df_sdbondstatic_third['similarity_3rd']  = df_sdbondstatic_third.apply (lambda row: third_similarity(row), axis=1)
    df_similarity_third                      = df_sdbondstatic_third[['_id','similarity_3rd']]

    if 'ISSUER TICKER' in header_low:
        df_similarity_third['ticker_dist_square']   = df_similarity_third.apply(lambda row: ticker_weight(row), axis=1)
        df_similarity_third['similarity_3rd']       = (df_similarity_third['similarity_3rd']*df_similarity_third['similarity_3rd'] + df_similarity_third['ticker_dist_square']) ** 0.5
        df_similarity_third                         = df_similarity_third[['_id','similarity_3rd']]
    
elif 'ISSUER TICKER' in header_low:
    third = True
    df_sdbondstatic_third                           = df_sdbondstatic[['_id']]
    df_sdbondstatic_third['similarity_3rd']         = df_sdbondstatic_third.apply(lambda row: ticker_weight(row), axis=1)
    df_sdbondstatic_third['similarity_3rd']         = df_sdbondstatic_third['similarity_3rd'] ** 0.5
    df_similarity_third                             = df_sdbondstatic_third[['_id','similarity_3rd']]

#df_similarity_third   = df_similarity_third.sort_values(by='similarity_3rd')



first_weight    =   10.0
second_weight   =   2.0
third_weight    =   1.0

if (first == True and second == True and third == True):
    df_join             = pd.merge(pd.merge(df_similarity_first, df_similarity_second,on='_id', how='inner'),df_similarity_third, on='_id', how='inner')
    df_join['distance'] = round((first_weight*df_join['similarity_1st'] + second_weight*df_join['similarity_2nd'] + third_weight*df_join['similarity_3rd']) / (first_weight+second_weight+third_weight) , 5)
    df_join             = df_join.drop(['similarity_1st','similarity_2nd','similarity_3rd'],axis=1)
    df_join_final       = df_join.sort_values(by='distance')


if (first != True and second == True and third == True):
    df_join             = pd.merge(df_similarity_second, df_similarity_third, on='_id', how='inner')
    df_join['distance'] = round(( second_weight*df_join['similarity_2nd'] + third_weight*df_join['similarity_3rd']) / (second_weight+third_weight) , 5)
    df_join             = df_join.drop(['similarity_2nd','similarity_3rd'],axis=1)
    df_join_final       = df_join.sort_values(by='distance')

 
if (first == True and second != True and third == True):
    df_join             = pd.merge(df_similarity_first, df_similarity_third, on='_id', how='inner')
    df_join['distance'] = round(( first_weight*df_join['similarity_1st'] + third_weight*df_join['similarity_3rd']) / (first_weight+third_weight) , 5)
    df_join             = df_join.drop(['similarity_1st','similarity_3rd'],axis=1)
    df_join_final       = df_join.sort_values(by='distance')        
        
        
if (first == True and second == True and third != True):
    df_join             = pd.merge(df_similarity_first, df_similarity_second, on='_id', how='inner')
    df_join['distance'] = round(( first_weight*df_join['similarity_1st'] + second_weight*df_join['similarity_2nd']) / (first_weight+second_weight) , 5)
    df_join             = df_join.drop(['similarity_1st','similarity_2nd'],axis=1)
    df_join_final       = df_join.sort_values(by='distance')        
              
        
if (first == True and second != True and third != True):
    df_join         = df_similarity_first
    df_join_final   = df_join.sort_values(by='similarity_1st')      

        
if (first != True and second == True and third != True):
    df_join         = df_similarity_second
    df_join_final   = df_join.sort_values(by='similarity_2nd')          
     
    
if (first != True and second != True and third == True):
    df_join         = df_similarity_third
    df_join_final   = df_join.sort_values(by='similarity_3rd')           


df_join_final.columns = ['_id', 'distance']


df_join_final['_id']    = df_join_final['_id'].map(dict_nickname)
df_join_final           = df_join_final.groupby(['_id'])['distance'].min().reset_index()
df_join_final           = df_join_final.sort_values(by='distance') 
df_join_final.columns   = ['security', 'distance']


score_scaler                = MinMaxScaler()
scaled_dist                 = score_scaler.fit_transform(numpy.array(df_join_final['distance']).reshape(len(df_join_final['distance']) ,1))
df_join_final['distance']   = scaled_dist
df_join_final               = df_join_final.round(5)


df_sdbondstatic_ori         = pd.read_csv(model_training_input+'sdbondstatic_original.csv', keep_default_na=False)

cols_used    = ['nickname',
                'issuer_ticker',
                'currency',
                'res_maturity',
                'rating_type',
                'issuer_rating',	
                'region',
                'country',
                'bbg_industry_sector',
                'bbg_industry_group',
                'type_le',
                'ranking',
                'coupon_type',
                'distribution_type',
                'deal_size_mil',
                'issue_date',
                'coupon_rate',
                'basel_classification',
                'special_classification',
                'maturity_type']


df_sdbondstatic_ori         = df_sdbondstatic_ori[cols_used]
df_sdbondstatic_ori_dedup   = df_sdbondstatic_ori.groupby('nickname').first().reset_index()
df_sdbondstatic_ori_dedup   = df_sdbondstatic_ori_dedup.apply(lambda x: x.str.upper() if (x.dtype == 'object') else x)
df_sdbondstatic_ori_dedup   = df_sdbondstatic_ori_dedup.apply(lambda x: x.str.strip() if (x.dtype == 'object') else x)
df_sdbondstatic_ori_dedup.loc[df_sdbondstatic_ori_dedup['ranking'] == '', 'ranking'] = 'SENIOR'

if BASEL_CLASSIFICATION == 'TIER 1':
    df_sdbondstatic_ori_dedup = df_sdbondstatic_ori_dedup[df_sdbondstatic_ori_dedup['ranking'] != 'SENIOR']

if RANKING == 'SENIOR':
    df_sdbondstatic_ori_dedup = df_sdbondstatic_ori_dedup[df_sdbondstatic_ori_dedup['ranking'] != 'SUBORDINATED']
    df_sdbondstatic_ori_dedup = df_sdbondstatic_ori_dedup[df_sdbondstatic_ori_dedup['maturity_type'] != 'PERP']
    df_sdbondstatic_ori_dedup = df_sdbondstatic_ori_dedup[df_sdbondstatic_ori_dedup['basel_classification'] != 'TIER 1']


df_sdbondstatic_ori_dedup['res_mat_diff_abs'] = (df_sdbondstatic_ori_dedup['res_maturity'] - TENOR).abs()

def create_day_diff(row):
    if len(row['issue_date']) == 10:
        year    = int(row['issue_date'].split('-')[0])
        month   = int(row['issue_date'].split('-')[1])
        day     = int(row['issue_date'].split('-')[2])

        issue_day = datetime.date(year, month, day)
        today     = datetime.date.today()
        delta = today - issue_day
        return delta.days
    else:
        return 0

df_sdbondstatic_ori_dedup['day_diff'] = df_sdbondstatic_ori_dedup.apply (lambda row: create_day_diff(row), axis=1)


df_sdbondstatic_ori_dedup_top = df_sdbondstatic_ori_dedup.loc[(df_sdbondstatic_ori_dedup['issuer_ticker'] == ISSUER_TICKER) & (df_sdbondstatic_ori_dedup['currency'] == CURRENCY) & (df_sdbondstatic_ori_dedup['ranking'] == RANKING)]
df_sdbondstatic_ori_dedup_top = df_sdbondstatic_ori_dedup_top.sort_values(by=['res_mat_diff_abs', 'deal_size_mil','day_diff'], ascending=[True, False, True])


df_sdbondstatic_ori_dedup_top = df_sdbondstatic_ori_dedup_top.iloc[ :3 , :]


for i in range(len(df_join_final)):
    nickname = str(df_join_final.iloc[i,0]).upper().strip()
    print (nickname)
    if nickname not in df_sdbondstatic_ori_dedup_top['nickname'] and nickname in df_sdbondstatic_ori_dedup['nickname']:
        print (nickname)
    








list_final_output = []

for i in range(len(df_join_final)):
    one_row  = []
    nickname = str(df_join_final.iloc[i,0]).upper().strip()
    distance = df_join_final.iloc[i,1]
    one_row.append(nickname)
    one_row.append(distance)
    one_row  = one_row + df_sdbondstatic_ori_dedup[df_sdbondstatic_ori_dedup['nickname'] == nickname].iloc[:, 1:].values.tolist()[0]
    list_final_output.append(one_row)


df_final_output         = pd.DataFrame(list_final_output)
cols_used.insert(1,'Distance')
df_final_output.columns = cols_used
dict_rename             = {'nickname':'Nickname'}


if 'ISSUER COUNTRY' in header_important:
    dict_rename['country'] = 'Issuer_Country_H'
elif 'ISSUER COUNTRY' in header_medium:
    dict_rename['country'] = 'Issuer_Country_M'
else:
    dict_rename['country'] = 'Issuer_Country_L'

if 'CURRENCY' in header_important:
    dict_rename['currency'] = 'Currency_H'
elif 'CURRENCY' in header_medium:
    dict_rename['currency'] = 'Currency_M'
else:
    dict_rename['currency'] = 'Currency_L'  
    
if 'TENOR' in header_important:
    dict_rename['res_maturity'] = 'Tenor_H'
elif 'TENOR' in header_medium:
    dict_rename['res_maturity'] = 'Tenor_M'
else:
    dict_rename['res_maturity'] = 'Tenor_L'     
    
if 'ISSUER TICKER' in header_important:
    dict_rename['issuer_ticker'] = 'Issuer_Ticker_H'
elif 'ISSUER TICKER' in header_medium:
    dict_rename['issuer_ticker'] = 'Issuer_Ticker_M'
else:
    dict_rename['issuer_ticker'] = 'Issuer_Ticker_L'         
    
if 'ISSUER REGION' in header_important:
    dict_rename['region'] = 'Issuer_Region_H'
elif 'ISSUER REGION' in header_medium:
    dict_rename['region'] = 'Issuer_Region_M'
else:
    dict_rename['region'] = 'Issuer_Region_L'         
    
if 'BASEL CLASSIFICATION' in header_important:
    dict_rename['basel_classification'] = 'Basel_Classification_H'
elif 'BASEL CLASSIFICATION' in header_medium:
    dict_rename['basel_classification'] = 'Basel_Classification_M'
else:
    dict_rename['basel_classification'] = 'Basel_Classification_L'   

if 'ISSUER TYPE' in header_important:
    dict_rename['type_le'] = 'Issuer_Type_H'
elif 'ISSUER TYPE' in header_medium:
    dict_rename['type_le'] = 'Issuer_Type_M'
else:
    dict_rename['type_le'] = 'Issuer_Type_L'   

if 'RATING TYPE' in header_important:
    dict_rename['rating_type'] = 'Rating_Type_H'
elif 'RATING TYPE' in header_medium:
    dict_rename['rating_type'] = 'Rating_Type_M'
else:
    dict_rename['rating_type'] = 'Rating_Type_L'   

if 'SPECIAL CLASSIFICATION' in header_important:
    dict_rename['special_classification'] = 'Special_Classification_H'
elif 'SPECIAL CLASSIFICATION' in header_medium:
    dict_rename['special_classification'] = 'Special_Classification_M'
else:
    dict_rename['special_classification'] = 'Special_Classification_L'  

if 'INDUSTRY SECTOR' in header_important:
    dict_rename['bbg_industry_sector'] = 'Industry_Sector_H'
elif 'INDUSTRY SECTOR' in header_medium:
    dict_rename['bbg_industry_sector'] = 'Industry_Sector_M'
else:
    dict_rename['bbg_industry_sector'] = 'Industry_Sector_L'  

if 'DISTRIBUTION TYPE' in header_important:
    dict_rename['distribution_type'] = 'Distribution_Type_H'
elif 'DISTRIBUTION TYPE' in header_medium:
    dict_rename['distribution_type'] = 'Distribution_Type_M'
else:
    dict_rename['distribution_type'] = 'Distribution_Type_L'  

if 'DEAL SIZE MIL' in header_important:
    dict_rename['deal_size_mil'] = 'Deal_Size_H'
elif 'DEAL SIZE MIL' in header_medium:
    dict_rename['deal_size_mil'] = 'Deal_Size_M'
else:
    dict_rename['deal_size_mil'] = 'Deal_Size_L'  

if 'MATURITY TYPE' in header_important:
    dict_rename['maturity_type'] = 'Maturity_Type_H'
elif 'MATURITY TYPE' in header_medium:
    dict_rename['maturity_type'] = 'Maturity_Type_M'
else:
    dict_rename['maturity_type'] = 'Maturity_Type_L'  

if 'COUPON TYPE' in header_important:
    dict_rename['coupon_type'] = 'Coupon_Type_H'
elif 'COUPON TYPE' in header_medium:
    dict_rename['coupon_type'] = 'Coupon_Type_M'
else:
    dict_rename['coupon_type'] = 'Coupon_Type_L'  

if 'COUPON RATE' in header_important:
    dict_rename['coupon_rate'] = 'Coupon_Rate_H'
elif 'COUPON RATE' in header_medium:
    dict_rename['coupon_rate'] = 'Coupon_Rate_M'
else:
    dict_rename['coupon_rate'] = 'Coupon_Rate_L'  

if 'INDUSTRY' in header_important:
    dict_rename['bbg_industry_group'] = 'Industry_H'
elif 'INDUSTRY' in header_medium:
    dict_rename['bbg_industry_group'] = 'Industry_M'
else:
    dict_rename['bbg_industry_group'] = 'Industry_L'  

if 'ISSUER RATINGS' in header_important:
    dict_rename['issuer_rating'] = 'Issuer_Rating_H'
elif 'ISSUER RATINGS' in header_medium:
    dict_rename['issuer_rating'] = 'Issuer_Rating_M'
else:
    dict_rename['issuer_rating'] = 'Issuer_Rating_L'  

if 'RANKING' in header_important:
    dict_rename['ranking'] = 'Ranking_H'
elif 'RANKING' in header_medium:
    dict_rename['ranking'] = 'Ranking_M'
else:
    dict_rename['ranking'] = 'Ranking_L'  


df_final_output = df_final_output.rename(columns=dict_rename)





df_final_output.to_csv(model_output + 'predicted_securities.csv',index=False)





 













