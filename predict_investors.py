# -*- coding: utf-8 -*-
import os
import sys
import math
import numpy
import pickle
import pandas as pd
from scipy.spatial import distance

#user_input = sys.argv[1]

os.chdir(os.path.dirname(__file__))

model_training_input    = '../model_input/training_input/'
user_input              = '../model_input/user_input/ICBCAS.csv'
model_output            = '../model_output/predicted_investors/'

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
df_user_input.loc[df_user_input['Field'].eq('ISSUER TYPE') & df_user_input['Value'].eq('SUPRA'), 'Value'] = 'SOV'

print (df_user_input[['Field','Value']])

dict_user_input         = df_user_input.set_index('Field')['Value'].to_dict()

BASEL_CLASSIFICATION    =  dict_user_input['BASEL CLASSIFICATION'].strip()
#COUPON_RATE             =  float(dict_user_input['COUPON RATE'].strip())
COUPON_TYPE             =  dict_user_input['COUPON TYPE'].strip()
CURRENCY                =  dict_user_input['CURRENCY'].strip()
DEAL_SIZE_MIL           =  float(dict_user_input['DEAL SIZE MIL'].strip())
DISTRIBUTION_TYPE       =  dict_user_input['DISTRIBUTION TYPE'].strip()
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



df_detailbonds_original = pd.read_csv(model_training_input+'detailbonds_orginal.csv', keep_default_na=False)
df_detailbonds_original = df_detailbonds_original[df_detailbonds_original['investor_name'] != '']

df_detailbonds_original['product']              = df_detailbonds_original['product'].str.upper() 
df_detailbonds_original['fi_corp_sov']          = df_detailbonds_original['fi_corp_sov'].str.upper() 
df_detailbonds_original['basel_classification'] = df_detailbonds_original['basel_classification'].str.upper() 
df_detailbonds_original['special_caracters']    = df_detailbonds_original['special_caracters'].str.upper() 
df_detailbonds_original['currency']             = df_detailbonds_original['currency'].str.upper() 


df_detailbonds_original_day_diff = df_detailbonds_original.groupby(['investor_name'])['day_diff'].min()
df_detailbonds_original_day_diff = df_detailbonds_original_day_diff.reset_index()
df_detailbonds_original_day_diff.columns = ['investor_name','recent']
df_detailbonds_original_day_diff         = df_detailbonds_original_day_diff[df_detailbonds_original_day_diff['recent'] <= 366]
df_detailbonds_original_filtered         = df_detailbonds_original.loc[df_detailbonds_original['investor_name'].isin(set(df_detailbonds_original_day_diff['investor_name']))]

df_detailbonds_original_filtered = df_detailbonds_original_filtered.loc[df_detailbonds_original_filtered['currency'] == CURRENCY]


special     = False
rating_type = False
basel       = False
#issuer_type = False


if SPECIAL_CLASSIFICATION != 'N/A':
    special = True
    df_detailbonds_original_filtered = df_detailbonds_original_filtered.loc[df_detailbonds_original_filtered['special_caracters'] == SPECIAL_CLASSIFICATION]

if RATING_TYPE != 'N/A':
    rating_type = True
    df_detailbonds_original_filtered = df_detailbonds_original_filtered.loc[df_detailbonds_original_filtered['product'] == RATING_TYPE]

if ISSUER_TYPE == 'FI' and BASEL_CLASSIFICATION != 'N/A':
    basel = True
    df_detailbonds_original_filtered = df_detailbonds_original_filtered.loc[(df_detailbonds_original_filtered['basel_classification'] == BASEL_CLASSIFICATION) & (df_detailbonds_original_filtered['fi_corp_sov'] == ISSUER_TYPE)]


#==============================================================================
# 
# if SPECIAL_CLASSIFICATION == 'N/A' and CURRENCY == 'USD' and RATING_TYPE != 'CROSSOVER' and basel == False:
#     df_detailbonds_original_filtered = df_detailbonds_original_filtered.loc[df_detailbonds_original_filtered['fi_corp_sov'] == ISSUER_TYPE]
# 
#     
#==============================================================================
    

df_detailbonds_original_filtered_sum = df_detailbonds_original_filtered.groupby(['investor_name'])['allocated'].agg('sum')
df_detailbonds_original_filtered_sum = df_detailbonds_original_filtered_sum.reset_index()
df_detailbonds_original_filtered_sum.columns = ['investor_name', 'allocated_sum']
df_detailbonds_original_filtered_sum = df_detailbonds_original_filtered_sum.sort_values(['allocated_sum'], ascending = False)

df_investors_pool                    = df_detailbonds_original_filtered_sum.iloc[ :int(len(df_detailbonds_original_filtered_sum)*1) , :]



if special:
    if basel and RATING_TYPE in ('HY', 'CROSSOVER'):
        df_investors_first_filter = df_investors_pool.iloc[ :85, :]
    elif basel or RATING_TYPE in ('HY', 'CROSSOVER'):
        df_investors_first_filter = df_investors_pool.iloc[ :80, :]
    else:
        df_investors_first_filter = df_investors_pool.iloc[ :75, :]
else:
    if basel and RATING_TYPE in ('HY', 'CROSSOVER'):
        df_investors_first_filter = df_investors_pool.iloc[ :30, :]
    else:
        df_investors_first_filter = df_investors_pool.iloc[ :20, :]


print (df_investors_first_filter)
       
user_input_cat = []
user_input_num = []

if TENOR <= 0:
    TENOR = 1
user_input_num.append(numpy.log2(TENOR))
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


user_input_cat.append(ISSUER_TICKER.upper())
user_input_cat.append(CURRENCY.upper())
user_input_cat.append(BASEL_CLASSIFICATION.upper())
user_input_cat.append(INDUSTRY.upper())
user_input_cat.append(INDUSTRY_SECTOR.upper())
user_input_cat.append(COUPON_TYPE.upper())
user_input_cat.append(DISTRIBUTION_TYPE.upper())
user_input_cat.append(ISSUER_COUNTRY.upper())
user_input_cat.append(ISSUER_REGION.upper())
user_input_cat.append(RANKING.upper())
user_input_cat.append(SPECIAL_CLASSIFICATION.upper())
user_input_cat.append(MATURITY_TYPE.upper())
#user_input_cat.append(FLOAT_INDEX.upper())

user_input_cat_2D =[]
user_input_cat_2D.append(user_input_cat)


with open(model_training_input+'detailbonds_ohe.pickle', 'rb') as handle:
    ohe = pickle.load(handle) 

ohe_encoded = ohe.transform(user_input_cat_2D).toarray()
ohe_user_input = user_input_num + ohe_encoded.tolist()[0]


with open(model_training_input+'detailbonds_mms.pickle', 'rb') as handle:
    mms = pickle.load(handle) 

user_input_scaled = mms.transform([ohe_user_input])

df_detailbonds_unique_bond  = pd.read_csv(model_training_input+'detailbonds_unique_bond.csv', keep_default_na=False)
df_final_user_input         = pd.DataFrame(user_input_scaled)  
df_final_user_input.columns = df_detailbonds_unique_bond.columns[1:]


numeric_scale_1 = 5.0
df_final_user_input['bond_age']         = df_final_user_input['bond_age'] * numeric_scale_1

numeric_scale_2 = 3.0
df_final_user_input['fi_corp_sov_map']  = df_final_user_input['fi_corp_sov_map'] * numeric_scale_2

numeric_scale_3 = 3.0
df_final_user_input['rating_map']       = df_final_user_input['rating_map'] * numeric_scale_3

numeric_scale_4 = 3.0
df_final_user_input['product_map']      = df_final_user_input['product_map'] * numeric_scale_4


if CURRENCY == 'USD':
    if ISSUER_COUNTRY == 'SINGAPORE':
        header_important = ['CURRENCY', 'ISSUER TICKER','ISSUER COUNTRY']
        header_medium    = ['TENOR','ISSUER REGION', 'BASEL CLASSIFICATION', 'SPECIAL CLASSIFICATION','RATING TYPE']
        header_low       = ['RANKING', 'MATURITY TYPE','DISTRIBUTION TYPE', 'INDUSTRY', 'INDUSTRY SECTOR', 'ISSUER RATINGS', 'COUPON TYPE','ISSUER TYPE']
    else:
        header_important = ['CURRENCY', 'ISSUER TICKER','RATING TYPE','ISSUER TYPE', 'SPECIAL CLASSIFICATION']
        header_medium    = ['TENOR','ISSUER REGION', 'BASEL CLASSIFICATION', 'INDUSTRY', 'COUPON TYPE', 'INDUSTRY SECTOR', 'ISSUER RATINGS']
        header_low       = ['RANKING', 'MATURITY TYPE','DISTRIBUTION TYPE','ISSUER COUNTRY']
    
else:
    if ISSUER_COUNTRY == 'SINGAPORE':
        header_important = ['CURRENCY', 'ISSUER TICKER','ISSUER COUNTRY']
        header_medium    = ['TENOR','ISSUER REGION', 'BASEL CLASSIFICATION', 'SPECIAL CLASSIFICATION','RATING TYPE']
        header_low       = ['RANKING', 'MATURITY TYPE','DISTRIBUTION TYPE', 'INDUSTRY', 'INDUSTRY SECTOR', 'ISSUER RATINGS', 'COUPON TYPE','ISSUER TYPE']
    
    
    
    else:
        header_important = ['CURRENCY', 'RATING TYPE', 'SPECIAL CLASSIFICATION']
        header_medium    = ['ISSUER TYPE','ISSUER TICKER','TENOR','ISSUER REGION', 'BASEL CLASSIFICATION', 'INDUSTRY', 'COUPON TYPE', 'INDUSTRY SECTOR', 'ISSUER RATINGS']
        header_low       = ['RANKING', 'MATURITY TYPE','DISTRIBUTION TYPE','ISSUER COUNTRY']
    
    



dict_col_header = {
                    'bond':'TENOR',
                     #'deal':'DEAL SIZE MIL',
                     #'sbn':'COUPON RATE',
                    'product':'RATING TYPE',
                    'rating':'ISSUER RATINGS',
                    'fi':'ISSUER TYPE',
                    'x0':'ISSUER TICKER',
                    'x1':'CURRENCY',
                    'x2':'BASEL CLASSIFICATION',
                    'x3':'INDUSTRY',
                    'x4':'INDUSTRY SECTOR',
                    'x5':'COUPON TYPE',
                    'x6':'DISTRIBUTION TYPE',
                    'x7':'ISSUER COUNTRY',
                    'x8':'ISSUER REGION',
                    'x9':'RANKING',
                    'x10':'SPECIAL CLASSIFICATION',
                    'x11':'MATURITY TYPE'
                     #'x12':'FLOAT INDEX'
                  }


cols_important  = []
cols_medium     = []
cols_low        = []

for each_col in df_detailbonds_unique_bond.columns[1:]:
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
    
        

if (len(cols_important) != 0):
    # first group of features
    df_final_user_input_first           =   df_final_user_input[cols_important]
    df_detailbonds_unique_bond_first    =   df_detailbonds_unique_bond[['security'] + cols_important]
    
    list_final_first = []
    
    for i in range(len(df_detailbonds_unique_bond_first.index)):
        one_security    = df_detailbonds_unique_bond_first.iloc[i, 1: ]
        dist            = distance.euclidean(df_final_user_input_first.iloc[0, : ], one_security)
        one_list        = []
        one_list.append(df_detailbonds_unique_bond_first.iloc[i, 0] )
        one_list.append(round(dist, 5))
        list_final_first.append(one_list)
    
    df_similaity_first          = pd.DataFrame(list_final_first)
    df_similaity_first.columns  = ['security','similarity_1st']
    df_similaity_first          = df_similaity_first.sort_values(by='similarity_1st')



if (len(cols_medium) != 0):
    # second group of features
    df_final_user_input_second           =   df_final_user_input[cols_medium]
    df_detailbonds_unique_bond_second    =   df_detailbonds_unique_bond[['security'] + cols_medium]
    
    list_final_second = []
    
    for i in range(len(df_detailbonds_unique_bond_second.index)):
        one_security    = df_detailbonds_unique_bond_second.iloc[i, 1: ]
        dist            = distance.euclidean(df_final_user_input_second.iloc[0, : ], one_security)
        one_list        = []
        one_list.append(df_detailbonds_unique_bond_second.iloc[i, 0])
        one_list.append(round(dist, 5))
        list_final_second.append(one_list)
    
    df_similaity_second         = pd.DataFrame(list_final_second)
    df_similaity_second.columns = ['security','similarity_2nd']
    df_similaity_second         = df_similaity_second.sort_values(by='similarity_2nd')



if (len(cols_low) != 0):
    # third group of features
    df_final_user_input_third           =   df_final_user_input[cols_low]
    df_detailbonds_unique_bond_third    =   df_detailbonds_unique_bond[['security'] + cols_low]
    
    list_final_third = []
    
    for i in range(len(df_detailbonds_unique_bond_third.index)):
        one_security    = df_detailbonds_unique_bond_third.iloc[i, 1: ]
        dist            = distance.euclidean(df_final_user_input_third.iloc[0, : ], one_security)
        one_list        = []
        one_list.append(df_detailbonds_unique_bond_third.iloc[i, 0])
        one_list.append(round(dist, 5))
        list_final_third.append(one_list)
    
    df_similaity_third         = pd.DataFrame(list_final_third)
    df_similaity_third.columns = ['security','similarity_3rd']
    df_similaity_third         = df_similaity_third.sort_values(by='similarity_3rd')



first_weight    =   3.0
second_weight   =   2.0
third_weight    =   1.0

if (len(cols_important) != 0 and len(cols_medium) != 0 and len(cols_low) != 0):
    df_join             = pd.merge(pd.merge(df_similaity_first, df_similaity_second,on='security', how='inner'),df_similaity_third, on='security', how='inner')
    df_join['distance'] = round((first_weight*df_join['similarity_1st'] + second_weight*df_join['similarity_2nd'] + third_weight*df_join['similarity_3rd']) / (first_weight+second_weight+third_weight) , 5)
    df_join             = df_join.drop(['similarity_1st','similarity_2nd','similarity_3rd'],axis=1)
    df_join_final       = df_join.sort_values(by='distance')


if (len(cols_important) == 0 and len(cols_medium) != 0 and len(cols_low) != 0):
    df_join             = pd.merge(df_similaity_second, df_similaity_third, on='security', how='inner')
    df_join['distance'] = round(( second_weight*df_join['similarity_2nd'] + third_weight*df_join['similarity_3rd']) / (second_weight+third_weight) , 5)
    df_join             = df_join.drop(['similarity_2nd','similarity_3rd'],axis=1)
    df_join_final       = df_join.sort_values(by='distance')

 
if (len(cols_important) != 0 and len(cols_medium) == 0 and len(cols_low) != 0):
    df_join             = pd.merge(df_similaity_first, df_similaity_third, on='security', how='inner')
    df_join['distance'] = round(( first_weight*df_join['similarity_1st'] + third_weight*df_join['similarity_3rd']) / (first_weight+third_weight) , 5)
    df_join             = df_join.drop(['similarity_1st','similarity_3rd'],axis=1)
    df_join_final       = df_join.sort_values(by='distance')        
        
        
if (len(cols_important) != 0 and len(cols_medium) != 0 and len(cols_low) == 0):
    df_join             = pd.merge(df_similaity_first, df_similaity_second, on='security', how='inner')
    df_join['distance'] = round(( first_weight*df_join['similarity_1st'] + second_weight*df_join['similarity_2nd']) / (first_weight+second_weight) , 5)
    df_join             = df_join.drop(['similarity_1st','similarity_2nd'],axis=1)
    df_join_final       = df_join.sort_values(by='distance')        
              
        
if (len(cols_important) != 0 and len(cols_medium) == 0 and len(cols_low) == 0):
    df_join         = df_similaity_first
    df_join_final   = df_join.sort_values(by='similarity_1st')      
    df_join_final.columns = ['security','distance']
        
    
if (len(cols_important) == 0 and len(cols_medium) != 0 and len(cols_low) == 0):
    df_join         = df_similaity_second
    df_join_final   = df_join.sort_values(by='similarity_2nd')          
    df_join_final.columns = ['security','distance']
    
    
if (len(cols_important) == 0 and len(cols_medium) == 0 and len(cols_low) != 0):
    df_join         = df_similaity_third
    df_join_final   = df_join.sort_values(by='similarity_3rd')           
    df_join_final.columns = ['security','distance']


df_join_final = df_join_final.groupby(['security'])['distance'].min().reset_index()


# select top similar security
df_join_final           = df_join_final.sort_values(by='distance') 
df_top_security         = df_join_final.loc[df_join_final['distance'] <= 1.0]

if len(df_top_security) >= 5:
    df_top_security = df_top_security.iloc[ :5 , :]
else:
    df_top_security = df_join_final.iloc[ :4, :]

print (df_top_security)




# create local and global frequency dataframe
df_detailbonds_all_transactions         = pd.read_csv(model_training_input+'detailbonds_all_transactions.csv', keep_default_na=False)
df_detailbonds_all_transactions_dedup   = df_detailbonds_all_transactions[['investor_name','security']].drop_duplicates()
df_investor_name_count_global_freq      = df_detailbonds_all_transactions_dedup['investor_name'].value_counts().rename_axis('investor_name').reset_index(name='count')


df_investor_pool_withdup                = df_detailbonds_all_transactions.loc[df_detailbonds_all_transactions['security'].isin(df_top_security['security'])][['security','investor_name']]
df_investor_pool_dedup                  = df_investor_pool_withdup.drop_duplicates()
df_investor_name_count_local_freq       = df_investor_pool_dedup['investor_name'].value_counts().rename_axis('investor_name').reset_index(name='count')


dict_investor_country                   = df_detailbonds_all_transactions.set_index('investor_name')['investor_country'].to_dict()


df_detailbonds_all_transactions_amount          = df_detailbonds_all_transactions.loc[df_detailbonds_all_transactions['deal_size_mil'] != 0]
df_detailbonds_all_transactions_amount          = df_detailbonds_all_transactions_amount.loc[df_detailbonds_all_transactions_amount['allocated'] > 0]
df_detailbonds_all_transactions_amount          = df_detailbonds_all_transactions_amount.groupby(['security', 'investor_name']).agg({'deal_size_mil':'max','allocated':'sum'}).reset_index()
df_detailbonds_all_transactions_amount['ratio'] = df_detailbonds_all_transactions_amount['allocated'] / df_detailbonds_all_transactions_amount['deal_size_mil']
df_allocation_ratio                             = df_detailbonds_all_transactions_amount.loc[df_detailbonds_all_transactions_amount['ratio'] < 1.0]


df_allocation_percentage                        = df_detailbonds_all_transactions.groupby(['security', 'investor_name'])['firm','allocated'].agg('sum').reset_index()
df_allocation_percentage['percentage']          = df_allocation_percentage['allocated'] / df_allocation_percentage['firm']


df_highest_similarity                           = df_join_final.loc[df_join_final['distance'] <= 0.3]
df_investor_pool_withdup_highest_similarity     = df_detailbonds_all_transactions.loc[df_detailbonds_all_transactions['security'].isin(df_highest_similarity['security'])][['security','investor_name']]
investors_highest_similarity_security           = set(df_investor_pool_withdup_highest_similarity['investor_name'])




local_freq_threshold                            = len(df_top_security) - 1
investors_selected_high_local_freq              = set(df_investor_name_count_local_freq.loc[df_investor_name_count_local_freq['count'] >= local_freq_threshold]['investor_name'])

investors_selected_high_local_freq = investors_selected_high_local_freq - (investors_selected_high_local_freq - set(df_detailbonds_original_filtered_sum['investor_name']))

if len(investors_selected_high_local_freq | set(df_investors_first_filter['investor_name'])) > 90:
    local_freq_threshod                         = len(df_top_security) 
    investors_selected_high_local_freq          = set(df_investor_name_count_local_freq.loc[df_investor_name_count_local_freq['count'] >= local_freq_threshold]['investor_name'])

    
selected_investors = investors_selected_high_local_freq | set(df_investors_first_filter['investor_name'])


rest_potential_investors = set(df_investor_name_count_local_freq['investor_name']) - selected_investors


list_investor_score_2nd_group = []

for each_investor in rest_potential_investors:
    
    local_freq      = int(df_investor_name_count_local_freq.loc[df_investor_name_count_local_freq['investor_name'] == each_investor]['count'])
    global_freq     = int(df_investor_name_count_global_freq.loc[df_investor_name_count_global_freq['investor_name'] == each_investor]['count'])

    final_score = local_freq * math.log10(global_freq)
        
    if each_investor in investors_highest_similarity_security:
        final_score = final_score * 1.5
    
    
    if each_investor in dict_investor_country:
        if dict_investor_country[each_investor] != 'N/A':
            if ISSUER_COUNTRY == dict_investor_country[each_investor].upper():
                final_score = final_score * 1.2
      
  
    
    if each_investor in dict_investor_country:
        if CURRENCY == 'GBP' and dict_investor_country[each_investor].upper() == 'UNITED KINGDOM':
            final_score = final_score * 1.2
            
        if CURRENCY == 'SGD' and dict_investor_country[each_investor].upper() in ('HONG KONG', 'SINGAPORE'):
            final_score = final_score * 1.2
         
        if CURRENCY == 'AUD' and dict_investor_country[each_investor].upper() == 'AUSTRALIA':
            final_score = final_score * 1.2 
        
        if CURRENCY == 'HKD' and dict_investor_country[each_investor].upper() in ('HONG KONG', 'SINGAPORE'):
            final_score = final_score * 1.2
        
        if CURRENCY == 'EUR' and dict_investor_country[each_investor].upper() in ('UNITED KINGDOM','SWITZERLAND','GERMANY','FRANCE'):
            final_score = final_score * 1.2
    
               
    
    
    df_investor_allocation_percentage     = df_allocation_percentage.loc[df_allocation_percentage['investor_name'] == each_investor]
    df_top_security_allocation_percentage = df_investor_allocation_percentage.loc[df_investor_allocation_percentage['security'].isin(df_top_security['security'])]
       
    mean_allocation_percentage = 0
    
    if (len(df_top_security_allocation_percentage) > 0):
        mean_allocation_percentage = df_top_security_allocation_percentage['percentage'].mean()
    
    if (mean_allocation_percentage >= 0.5):
        final_score = final_score * 1.5
    elif (mean_allocation_percentage >= 0.3 and mean_allocation_percentage < 0.5):
        final_score = final_score * 1.2  
        
           
        
              
    df_investor_allocation_ratio     = df_allocation_ratio.loc[df_allocation_ratio['investor_name'] == each_investor]
    df_top_security_allocation_ratio = df_investor_allocation_ratio.loc[df_investor_allocation_ratio['security'].isin(df_top_security['security'])]
    
    mean_allocation_ratio = 0
   
    if (len(df_top_security_allocation_ratio) > 0):
        mean_allocation_ratio = df_top_security_allocation_ratio['ratio'].mean()
    if (mean_allocation_ratio >= 0.015):
        final_score = final_score * 1.5
        
       
        
        
    if float(local_freq)/global_freq >= 0.05:
        final_score = final_score * 1.2
    
    
    one_list  = []
    one_list.append(each_investor)
    one_list.append(round(final_score, 5))
    list_investor_score_2nd_group.append(one_list)




df_investor_score_2nd_group          = pd.DataFrame(list_investor_score_2nd_group)
df_investor_score_2nd_group.columns  = ['investor name','score']
df_investor_score_2nd_group          = df_investor_score_2nd_group.sort_values(by='score', ascending  = False)


df_investor_score_2nd_group = df_investor_score_2nd_group.loc[df_investor_score_2nd_group['investor name'].isin(set(df_detailbonds_original_filtered_sum['investor_name']))]


rest_selected_investors     = df_investor_score_2nd_group.iloc[ :100-len(selected_investors), :]




 
final_selected_investors    = selected_investors | set(rest_selected_investors['investor name'])
 



















