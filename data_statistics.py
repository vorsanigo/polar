import pandas as pd
import os

TRAIN_PATH = "tot_data_semeval/train/"
DEV_PATH = "tot_data_semeval/dev/"
TEST_PATH = "tot_data_semeval/test/"

TRAIN_PATH_SPLIT = "dev_phase/merged subtasks split train/"
DEV_PATH_SPLIT = "dev_phase/merged subtasks split dev/"

LABELS_LIST = ['polarization', 'political', 'racial/ethnic', 'religious', 'gender/sexual', 'other', 'stereotype', 'vilification', 'dehumanization', 'extreme_language', 'lack_of_empathy' , 'invalidation']
LABELS_LIST_NO_S3 = ['polarization', 'political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
LABELS_LIST_S3 = ['polarization', 'stereotype', 'vilification', 'dehumanization', 'extreme_language', 'lack_of_empathy' , 'invalidation']
LABELS_LIST_S2 = ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
LIST_LANG_NO_S3 = ['mya.csv', 'ita.csv', 'pol.csv', 'rus.csv', 'merged_train_ita_split_shuffled.tsv', 'merged_train_pol_split_shuffled.tsv', 'merged_train_mya_split_shuffled.tsv', 'merged_train_rus_split_shuffled.tsv', 'merged_dev_ita_split_shuffled.tsv', 'merged_dev_pol_split_shuffled.tsv', 'merged_dev_mya_split_shuffled.tsv', 'merged_dev_rus_split_shuffled.tsv']

# flag to choose between original train, dev split or our train, dev split
ORIGINAL = True


# ORIGINAL DATA : size train, dev, test

# train df

print('TRAIN')

if ORIGINAL:
    train_files = os.listdir(TRAIN_PATH)
else:
    train_files = os.listdir(TRAIN_PATH_SPLIT)

train_files = sorted(train_files)

# read and concatenate train df, compute percentage of each label per language and in total
tot_train_df = pd.DataFrame()
tot_train_df_s3 = pd.DataFrame()
labels_train_df_1_2 = pd.DataFrame()
labels_train_df_3 = pd.DataFrame()
labels_train_df_1_2_frac = pd.DataFrame()
labels_train_df_3_frac = pd.DataFrame()

for file in train_files:

    #print(file)
    # read specific lang train df
    if ORIGINAL:
        train_df = pd.read_csv(TRAIN_PATH + file)
    else:
        train_df = pd.read_csv(TRAIN_PATH_SPLIT + file, sep='\t')
    #print(train_df)

    # check if lang is ita, pol, mya, rus, create one df for S1 and S2, another for S3
    if file in LIST_LANG_NO_S3:
        # case 1: only languages that do NOT have subtask 3

        # absolute number : sum over each category (no subtask 3 included here)
        dict_lang_sum_train_1_2 = train_df[LABELS_LIST_NO_S3].sum(axis=0).to_dict()
        # detect language from file name
        if ORIGINAL:
            dict_lang_sum_train_1_2['language'] = file[:3]
        else:
            dict_lang_sum_train_1_2['language'] = file[13:16]
        
        # fraction : total fraction over each category (no subtask 3 included here)
        dict_lang_sum_train_1_2_frac = (train_df[LABELS_LIST_NO_S3].sum(axis=0) / len(train_df) * 100).to_dict()
        # detect language from file name
        if ORIGINAL:
            dict_lang_sum_train_1_2_frac['language'] = file[:3]
        else:
            dict_lang_sum_train_1_2_frac['language'] = file[13:16]

        # add total number of instances
        dict_lang_sum_train_1_2['num instances'] = len(train_df)
        dict_lang_sum_train_1_2_frac['num instances'] = len(train_df)

    else:
        # case 1: only languages that have subtask 3

        # absolute number : sum over each category (subtask 3 included here)
        # create different df for s1, s2 and s3
        dict_lang_sum_train_1_2 = train_df[LABELS_LIST_NO_S3].sum(axis=0).to_dict()
        dict_lang_sum_train_3 = train_df[LABELS_LIST_S3].sum(axis=0).to_dict()
        # detect language from file name
        if ORIGINAL:
            dict_lang_sum_train_1_2['language'] = file[:3]
            dict_lang_sum_train_3['language'] = file[:3]
        else:
            dict_lang_sum_train_1_2['language'] = file[13:16]
            dict_lang_sum_train_3['language'] = file[13:16]

        # fraction : total fraction over each category (subtask 3 included here)
        dict_lang_sum_train_1_2_frac = (train_df[LABELS_LIST_NO_S3].sum(axis=0) / len(train_df) * 100).to_dict()
        dict_lang_sum_train_3_frac = (train_df[LABELS_LIST_S3].sum(axis=0) / len(train_df) * 100).to_dict()
        # detect language from file name
        if ORIGINAL:
            dict_lang_sum_train_1_2_frac['language'] = file[:3]
            dict_lang_sum_train_3_frac['language'] = file[:3]
        else:
            dict_lang_sum_train_1_2_frac['language'] = file[13:16]
            dict_lang_sum_train_3_frac['language'] = file[13:16]

        # add total number of instances
        dict_lang_sum_train_1_2['num instances'] = len(train_df)
        dict_lang_sum_train_1_2_frac['num instances'] = len(train_df)
        dict_lang_sum_train_3['num instances'] = len(train_df)
        dict_lang_sum_train_3_frac['num instances'] = len(train_df)
        
        # concatenate per-language statistics for subtask 3 (if language has subtask 3)
        labels_train_df_3 = pd.concat([labels_train_df_3, pd.DataFrame(dict_lang_sum_train_3, index=[0])])
        labels_train_df_3_frac = pd.concat([labels_train_df_3_frac, pd.DataFrame(dict_lang_sum_train_3_frac, index=[0])])

    # concatenate per-language statistics for subtasks 1 2
    labels_train_df_1_2 = pd.concat([labels_train_df_1_2, pd.DataFrame(dict_lang_sum_train_1_2, index=[0])])
    labels_train_df_1_2_frac = pd.concat([labels_train_df_1_2_frac, pd.DataFrame(dict_lang_sum_train_1_2_frac, index=[0])])
    
    if file not in LIST_LANG_NO_S3:
        tot_train_df_s3 = pd.concat([tot_train_df_s3, train_df])
    tot_train_df = pd.concat([tot_train_df, train_df])
    

# absolute number s1, s2
dict_tot_sum_train_1_2 = tot_train_df[LABELS_LIST_NO_S3].sum(axis=0).to_dict()
dict_tot_sum_train_1_2['language'] = 'tot'
dict_tot_sum_train_1_2['num instances'] = len(tot_train_df)
labels_train_df_1_2 = pd.concat([labels_train_df_1_2, pd.DataFrame(dict_tot_sum_train_1_2, index=[0])])

# fraction s1, s2
dict_tot_sum_train_1_2_frac = (tot_train_df[LABELS_LIST_NO_S3].sum(axis=0) / len(tot_train_df) * 100).to_dict()
dict_tot_sum_train_1_2_frac['language'] = 'tot'
dict_tot_sum_train_1_2_frac['num instances'] = len(tot_train_df)
labels_train_df_1_2_frac = pd.concat([labels_train_df_1_2_frac, pd.DataFrame(dict_tot_sum_train_1_2_frac, index=[0])])

# absolute number s3
dict_tot_sum_train_3 = tot_train_df_s3[LABELS_LIST_S3].sum(axis=0).to_dict()
dict_tot_sum_train_3['language'] = 'tot'
dict_tot_sum_train_3['num instances'] = len(tot_train_df)
labels_train_df_3 = pd.concat([labels_train_df_3, pd.DataFrame(dict_tot_sum_train_3, index=[0])])
labels_train_df_3.drop(columns='polarization')
# fraction s3
dict_tot_sum_train_3_frac = (tot_train_df_s3[LABELS_LIST_S3].sum(axis=0) / len(tot_train_df_s3) * 100).to_dict()
dict_tot_sum_train_3_frac['language'] = 'tot'
dict_tot_sum_train_3_frac['num instances'] = len(tot_train_df)
labels_train_df_3_frac = pd.concat([labels_train_df_3_frac, pd.DataFrame(dict_tot_sum_train_3_frac, index=[0])])
labels_train_df_3_frac.drop(columns='polarization')

# drop polarization column from df s3 (here we have less data)
labels_train_df_3 = labels_train_df_3.drop(columns=['polarization', 'num instances'])
labels_train_df_3_frac = labels_train_df_3_frac.drop(columns=['polarization', 'num instances'])

# total s2 + s3
# absolute number
labels_train_df_2_3 = pd.merge(labels_train_df_1_2, labels_train_df_3, on=['language'], how='left')
# fraction s2 + s3
labels_train_df_2_3_frac = pd.merge(labels_train_df_1_2_frac, labels_train_df_3_frac, on=['language'], how='left')

print('merged tot', labels_train_df_2_3)
print('merged frac', labels_train_df_2_3_frac)



# for each category of subtask 2 , count fraction of instances classified in each category of subtask 3

rel_s2_s3_df = pd.DataFrame()

for label in LABELS_LIST_S2:
    df_label = tot_train_df_s3[tot_train_df_s3[label] == 1]
    dict_sum_frac_s3 = (df_label[LABELS_LIST_S3].sum(axis=0) / len(df_label)).to_dict()
    dict_sum_frac_s3['Category S2'] = label
    rel_s2_s3_df = pd.concat([rel_s2_s3_df, pd.DataFrame(dict_sum_frac_s3, index=[0])])

# move "Category S2" col to the beginning, drop polarization col
col_category = rel_s2_s3_df.pop("Category S2")
col_polarization = rel_s2_s3_df.pop("polarization")
rel_s2_s3_df.insert(0, "Category S2", col_category)

# create latex table 

latex_table_s2_s3 = rel_s2_s3_df.to_latex(
    index=False,
    float_format="%.3f",
    caption='caption',
    label='label'
)
print(latex_table_s2_s3)


# move "lang" column at the beginning and add column with total number of instances (len df)

col_lang = labels_train_df_2_3.pop("language")
col_num_ins = labels_train_df_2_3.pop("num instances")
labels_train_df_2_3.insert(0, "num instances", col_num_ins)
labels_train_df_2_3.insert(0, "language", col_lang)

col_lang = labels_train_df_2_3_frac.pop("language")
col_num_ins = labels_train_df_2_3_frac.pop("num instances")
labels_train_df_2_3_frac.insert(0, "num instances", col_num_ins)
labels_train_df_2_3_frac.insert(0, "language", col_lang)

# fill nan (ita, pol, mya, rus) of subtask 3 with "-"
labels_train_df_2_3 = labels_train_df_2_3.fillna(0)
labels_train_df_2_3_frac = labels_train_df_2_3_frac.fillna(0)


'''col_lang = labels_train_df_3.pop("language")
col_num_ins = labels_train_df_3.pop("num instances")
labels_train_df_3.insert(0, "num instances", col_num_ins)
labels_train_df_3.insert(0, "language", col_lang)

col_lang = labels_train_df_3_frac.pop("language")
col_num_ins = labels_train_df_3_frac.pop("num instances")
labels_train_df_3_frac.insert(0, "num instances", col_num_ins)
labels_train_df_3_frac.insert(0, "language", col_lang)'''

'''print(labels_train_df_1_2)
print('\n')
print(labels_train_df_3)
print('\n')
print(labels_train_df_1_2_frac)
print('\n')
print(labels_train_df_3_frac)
'''

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# dev df

print('\n\nDEV')

if ORIGINAL:
    dev_files = os.listdir(DEV_PATH)
else:
    dev_files = os.listdir(DEV_PATH_SPLIT)

dev_files = sorted(dev_files)

# read and concatenate train df, compute percentage of each label per language and in total
tot_dev_df = pd.DataFrame()
tot_dev_df_s3 = pd.DataFrame()
labels_dev_df_1_2 = pd.DataFrame()
labels_dev_df_3 = pd.DataFrame()
labels_dev_df_1_2_frac = pd.DataFrame()
labels_dev_df_3_frac = pd.DataFrame()

for file in dev_files:

    # read specific lang train df
    if ORIGINAL:
        dev_df = pd.read_csv(DEV_PATH + file)
    else:
        dev_df = pd.read_csv(DEV_PATH_SPLIT + file, sep='\t')

    # check if lang is ita, pol, mya, rus, create one df for S1 and S2, another for S3
    if file in LIST_LANG_NO_S3:
        dict_lang_sum_dev_1_2 = dev_df[LABELS_LIST_NO_S3].sum(axis=0).to_dict()
        if ORIGINAL:
            dict_lang_sum_dev_1_2['language'] = file[:3]
        else:
            dict_lang_sum_dev_1_2['language'] = file[11:14]
        # fraction
        dict_lang_sum_dev_1_2_frac = (dev_df[LABELS_LIST_NO_S3].sum(axis=0) / len(dev_df) * 100).to_dict()
        if ORIGINAL:
            dict_lang_sum_dev_1_2_frac['language'] = file[:3]
        else:
            dict_lang_sum_dev_1_2_frac['language'] = file[11:14]
        # add column with number of instances
        dict_lang_sum_dev_1_2['num instances'] = len(dev_df)
        dict_lang_sum_dev_1_2_frac['num instances'] = len(dev_df)
    else:
        # absolute number
        dict_lang_sum_dev_1_2 = dev_df[LABELS_LIST_NO_S3].sum(axis=0).to_dict()
        dict_lang_sum_dev_3 = dev_df[LABELS_LIST_S3].sum(axis=0).to_dict()
        if ORIGINAL:
            dict_lang_sum_dev_1_2['language'] = file[:3]
            dict_lang_sum_dev_3['language'] = file[:3]
        else:
            dict_lang_sum_dev_1_2['language'] = file[11:14]
            dict_lang_sum_dev_3['language'] = file[11:14]
        # fraction
        dict_lang_sum_dev_1_2_frac = (dev_df[LABELS_LIST_NO_S3].sum(axis=0) / len(dev_df) * 100).to_dict()
        dict_lang_sum_dev_3_frac = (dev_df[LABELS_LIST_S3].sum(axis=0) / len(dev_df) * 100).to_dict()
        if ORIGINAL:
            dict_lang_sum_dev_1_2_frac['language'] = file[:3]
            dict_lang_sum_dev_3_frac['language'] = file[:3]
        else:
            dict_lang_sum_dev_1_2_frac['language'] = file[11:14]
            dict_lang_sum_dev_3_frac['language'] = file[11:14]
        # add column with number of instances
        dict_lang_sum_dev_1_2['num instances'] = len(dev_df)
        dict_lang_sum_dev_1_2_frac['num instances'] = len(dev_df)
        dict_lang_sum_dev_3['num instances'] = len(dev_df)
        dict_lang_sum_dev_3_frac['num instances'] = len(dev_df)
        
        labels_dev_df_3 = pd.concat([labels_dev_df_3, pd.DataFrame(dict_lang_sum_dev_3, index=[0])])
        labels_dev_df_3_frac = pd.concat([labels_dev_df_3_frac, pd.DataFrame(dict_lang_sum_dev_3_frac, index=[0])])

    labels_dev_df_1_2 = pd.concat([labels_dev_df_1_2, pd.DataFrame(dict_lang_sum_dev_1_2, index=[0])])
    labels_dev_df_1_2_frac = pd.concat([labels_dev_df_1_2_frac, pd.DataFrame(dict_lang_sum_dev_1_2_frac, index=[0])])
    
    # if lang has subtask 3 concatenate its results to df with subtask 3 results
    if file not in LIST_LANG_NO_S3:
        tot_dev_df_s3 = pd.concat([tot_dev_df_s3, dev_df])
    # concatenate everything
    tot_dev_df = pd.concat([tot_dev_df, dev_df])
    

# absolute number s1, s2 on total df with all langugages of s1 , s2
dict_tot_sum_dev_1_2 = tot_dev_df[LABELS_LIST_NO_S3].sum(axis=0).to_dict()
dict_tot_sum_dev_1_2['language'] = 'tot'
dict_tot_sum_dev_1_2['num instances'] = len(tot_dev_df)
# concatenate statistics on all the languages aggregated
labels_dev_df_1_2 = pd.concat([labels_dev_df_1_2, pd.DataFrame(dict_tot_sum_dev_1_2, index=[0])])
# fraction s1, s2 on total df with all langugage of s1, s2
dict_tot_sum_dev_1_2_frac = (tot_dev_df[LABELS_LIST_NO_S3].sum(axis=0) / len(tot_dev_df) * 100).to_dict()
dict_tot_sum_dev_1_2_frac['language'] = 'tot'
dict_tot_sum_dev_1_2_frac['num instances'] = len(tot_dev_df)
# concatenate statistics on all the languages aggregated
labels_dev_df_1_2_frac = pd.concat([labels_dev_df_1_2_frac, pd.DataFrame(dict_tot_sum_dev_1_2_frac, index=[0])])

# absolute number s3 on total df with all languages of s3 (no ita, pol, mya, rus)
dict_tot_sum_dev_3 = tot_dev_df_s3[LABELS_LIST_S3].sum(axis=0).to_dict()
dict_tot_sum_dev_3['language'] = 'tot'
dict_tot_sum_dev_3['num instances'] = len(tot_dev_df)
# concatenate statistics on all the data aggregated
labels_dev_df_3 = pd.concat([labels_dev_df_3, pd.DataFrame(dict_tot_sum_dev_3, index=[0])])
# fraction s3 on total df with all languages of s3 (no ita, pol, mya, rus)
dict_tot_sum_dev_3_frac = (tot_dev_df_s3[LABELS_LIST_S3].sum(axis=0) / len(tot_dev_df_s3) * 100).to_dict()
dict_tot_sum_dev_3_frac['language'] = 'tot'
dict_tot_sum_dev_3_frac['num instances'] = len(tot_dev_df)
# concatenate statistics on all the data aggregated
labels_dev_df_3_frac = pd.concat([labels_dev_df_3_frac, pd.DataFrame(dict_tot_sum_dev_3_frac, index=[0])])

print('tot s1 s2', tot_dev_df)
print('tot only with s3', tot_dev_df_s3)
print('stats frac s1 s2', labels_dev_df_1_2_frac)
print('stats frac s1 s2', labels_dev_df_3_frac)
print('stats s1 s2', labels_dev_df_1_2_frac)
print('stats s1 s2', labels_dev_df_3_frac)

'''print('tot', tot_dev_df_s3)
print(labels_dev_df_3_frac)
print(dict_tot_sum_dev_3_frac)
print(pd.DataFrame(dict_tot_sum_dev_3_frac, index=[0]))
print(labels_dev_df_1_2_frac)'''
'''print(labels_dev_df_1_2)
print(labels_dev_df_3)
print(labels_dev_df_1_2_frac)
print(labels_dev_df_3_frac)'''

# drop polarization column from df s3 (here we have less data)
labels_dev_df_3 = labels_dev_df_3.drop(columns=['polarization', 'num instances'])
labels_dev_df_3_frac = labels_dev_df_3_frac.drop(columns=['polarization', 'num instances'])

# total s2 + s3
# absolute number
labels_dev_df_2_3 = pd.merge(labels_dev_df_1_2, labels_dev_df_3, on=['language'], how='left')
# fraction s2 + s3
labels_dev_df_2_3_frac = pd.merge(labels_dev_df_1_2_frac, labels_dev_df_3_frac, on=['language'], how='left')

print('merged tot', labels_train_df_2_3)
print('merged frac', labels_train_df_2_3_frac)

# move "lang" column at the beginning and add column with total number of instances (len df)

col_lang = labels_dev_df_2_3.pop("language")
col_num_ins = labels_dev_df_2_3.pop("num instances")
labels_dev_df_2_3.insert(0, "num instances", col_num_ins)
labels_dev_df_2_3.insert(0, "language", col_lang)

col_lang = labels_dev_df_2_3_frac.pop("language")
col_num_ins = labels_dev_df_2_3_frac.pop("num instances")
labels_dev_df_2_3_frac.insert(0, "num instances", col_num_ins)
labels_dev_df_2_3_frac.insert(0, "language", col_lang)

# fill nan (ita, pol, mya, rus) of subtask 3 with "-"
labels_dev_df_2_3 = labels_dev_df_2_3.fillna(0)
labels_dev_df_2_3_frac = labels_dev_df_2_3_frac.fillna(0)

'''col_lang = labels_dev_df_1_2.pop("language")
col_num_ins = labels_dev_df_1_2.pop("num instances")
labels_dev_df_1_2.insert(0, "num instances", col_num_ins)
labels_dev_df_1_2.insert(0, "language", col_lang)

col_lang = labels_dev_df_1_2_frac.pop("language")
col_num_ins = labels_dev_df_1_2_frac.pop("num instances")
labels_dev_df_1_2_frac.insert(0, "num instances", col_num_ins)
labels_dev_df_1_2_frac.insert(0, "language", col_lang)

col_lang = labels_dev_df_3.pop("language")
col_num_ins = labels_dev_df_3.pop("num instances")
labels_dev_df_3.insert(0, "num instances", col_num_ins)
labels_dev_df_3.insert(0, "language", col_lang)

col_lang = labels_dev_df_3_frac.pop("language")
col_num_ins = labels_dev_df_3_frac.pop("num instances")
labels_dev_df_3_frac.insert(0, "num instances", col_num_ins)
labels_dev_df_3_frac.insert(0, "language", col_lang)'''


'''print(labels_train_df_1_2)
print('\n')
print(labels_train_df_3)
print('\n')
print(labels_train_df_1_2_frac)
print('\n')
print(labels_train_df_3_frac)'''

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# test df

print('\n\nTEST')
test_files = os.listdir(TEST_PATH)
test_files = sorted(test_files)

# read and concatenate train df, compute percentage of each label per language and in total
tot_test_df = pd.DataFrame()
tot_test_df_s3 = pd.DataFrame()
labels_test_df_1_2 = pd.DataFrame()
labels_test_df_3 = pd.DataFrame()
labels_test_df_1_2_frac = pd.DataFrame()
labels_test_df_3_frac = pd.DataFrame()

for file in test_files:

    # read specific lang train df
    test_df = pd.read_csv(TEST_PATH + file)
    
    # check if lang is ita, pol, mya, rus, create one df for S1 and S2, another for S3
    if file in LIST_LANG_NO_S3:
        dict_lang_sum_test_1_2 = test_df[LABELS_LIST_NO_S3].sum(axis=0).to_dict()
        dict_lang_sum_test_1_2['language'] = file[:3]
        # fraction
        dict_lang_sum_test_1_2_frac = (test_df[LABELS_LIST_NO_S3].sum(axis=0) / len(test_df) * 100).to_dict()
        dict_lang_sum_test_1_2_frac['language'] = file[:3]
        # add column with number of instances
        dict_lang_sum_test_1_2['num instances'] = len(test_df)
        dict_lang_sum_test_1_2_frac['num instances'] = len(test_df)
    else:
        # absolute number
        dict_lang_sum_test_1_2 = test_df[LABELS_LIST_NO_S3].sum(axis=0).to_dict()
        dict_lang_sum_test_3 = test_df[LABELS_LIST_S3].sum(axis=0).to_dict()
        dict_lang_sum_test_1_2['language'] = file[:3]
        dict_lang_sum_test_3['language'] = file[:3]
        # fraction
        dict_lang_sum_test_1_2_frac = (test_df[LABELS_LIST_NO_S3].sum(axis=0) / len(test_df) * 100).to_dict()
        dict_lang_sum_test_3_frac = (test_df[LABELS_LIST_S3].sum(axis=0) / len(test_df) * 100).to_dict()
        dict_lang_sum_test_1_2_frac['language'] = file[:3]
        dict_lang_sum_test_3_frac['language'] = file[:3]
        # add column with number of instances
        dict_lang_sum_test_1_2['num instances'] = len(test_df)
        dict_lang_sum_test_1_2_frac['num instances'] = len(test_df)
        dict_lang_sum_test_3['num instances'] = len(test_df)
        dict_lang_sum_test_3_frac['num instances'] = len(test_df)
        
        labels_test_df_3 = pd.concat([labels_test_df_3, pd.DataFrame(dict_lang_sum_test_3, index=[0])])
        labels_test_df_3_frac = pd.concat([labels_test_df_3_frac, pd.DataFrame(dict_lang_sum_test_3_frac, index=[0])])

    labels_test_df_1_2 = pd.concat([labels_test_df_1_2, pd.DataFrame(dict_lang_sum_test_1_2, index=[0])])
    labels_test_df_1_2_frac = pd.concat([labels_test_df_1_2_frac, pd.DataFrame(dict_lang_sum_test_1_2_frac, index=[0])])
    
    if file[:3] not in LIST_LANG_NO_S3:
        tot_test_df_s3 = pd.concat([tot_test_df_s3, test_df])
    tot_test_df = pd.concat([tot_test_df, test_df])
    

# absolute number s1, s2
dict_tot_sum_test_1_2 = tot_test_df[LABELS_LIST_NO_S3].sum(axis=0).to_dict()
dict_tot_sum_test_1_2['language'] = 'tot'
dict_tot_sum_test_1_2['num instances'] = len(tot_test_df)
labels_test_df_1_2 = pd.concat([labels_test_df_1_2, pd.DataFrame(dict_tot_sum_test_1_2, index=[0])])
# fraction s1, s2
dict_tot_sum_test_1_2_frac = (tot_test_df[LABELS_LIST_NO_S3].sum(axis=0) / len(tot_test_df) * 100).to_dict()
dict_tot_sum_test_1_2_frac['language'] = 'tot'
dict_tot_sum_test_1_2_frac['num instances'] = len(tot_test_df)
labels_test_df_1_2_frac = pd.concat([labels_test_df_1_2_frac, pd.DataFrame(dict_tot_sum_test_1_2_frac, index=[0])])

# absolute number s3
dict_tot_sum_test_3 = tot_test_df_s3[LABELS_LIST_S3].sum(axis=0).to_dict()
dict_tot_sum_test_3['language'] = 'tot'
dict_tot_sum_test_3['num instances'] = len(tot_test_df)
labels_test_df_3 = pd.concat([labels_test_df_3, pd.DataFrame(dict_tot_sum_test_3, index=[0])])
# fraction s3
dict_tot_sum_test_3_frac = (tot_test_df_s3[LABELS_LIST_S3].sum(axis=0) / len(tot_test_df_s3) * 100).to_dict()
dict_tot_sum_test_3_frac['language'] = 'tot'
dict_tot_sum_test_3_frac['num instances'] = len(tot_test_df)
labels_test_df_3_frac = pd.concat([labels_test_df_3_frac, pd.DataFrame(dict_tot_sum_test_3_frac, index=[0])])

# drop polarization column from df s3 (here we have less data)
labels_test_df_3 = labels_test_df_3.drop(columns=['polarization', 'num instances'])
labels_test_df_3_frac = labels_test_df_3_frac.drop(columns=['polarization', 'num instances'])

# total s2 + s3
# absolute number
labels_test_df_2_3 = pd.merge(labels_test_df_1_2, labels_test_df_3, on=['language'], how='left')
# fraction s2 + s3
labels_test_df_2_3_frac = pd.merge(labels_test_df_1_2_frac, labels_test_df_3_frac, on=['language'], how='left')

print('merged tot', labels_test_df_2_3)
print('merged frac', labels_test_df_2_3_frac)
# move "lang" column at the beginning and add column with total number of instances (len df)

col_lang = labels_test_df_2_3.pop("language")
col_num_ins = labels_test_df_2_3.pop("num instances")
labels_test_df_2_3.insert(0, "num instances", col_num_ins)
labels_test_df_2_3.insert(0, "language", col_lang)

col_lang = labels_test_df_2_3_frac.pop("language")
col_num_ins = labels_test_df_2_3_frac.pop("num instances")
labels_test_df_2_3_frac.insert(0, "num instances", col_num_ins)
labels_test_df_2_3_frac.insert(0, "language", col_lang)

# fill nan (ita, pol, mya, rus) of subtask 3 with "-"
labels_test_df_2_3 = labels_test_df_2_3.fillna(0)
labels_test_df_2_3_frac = labels_test_df_2_3_frac.fillna(0)

'''col_lang = labels_test_df_1_2.pop("language")
col_num_ins = labels_test_df_1_2.pop("num instances")
labels_test_df_1_2.insert(0, "num instances", col_num_ins)
labels_test_df_1_2.insert(0, "language", col_lang)

col_lang = labels_test_df_1_2_frac.pop("language")
col_num_ins = labels_test_df_1_2_frac.pop("num instances")
labels_test_df_1_2_frac.insert(0, "num instances", col_num_ins)
labels_test_df_1_2_frac.insert(0, "language", col_lang)

col_lang = labels_test_df_3.pop("language")
col_num_ins = labels_test_df_3.pop("num instances")
labels_test_df_3.insert(0, "num instances", col_num_ins)
labels_test_df_3.insert(0, "language", col_lang)

col_lang = labels_test_df_3_frac.pop("language")
col_num_ins = labels_test_df_3_frac.pop("num instances")
labels_test_df_3_frac.insert(0, "num instances", col_num_ins)
labels_test_df_3_frac.insert(0, "language", col_lang)'''


# create latex tables

# train

print('TRAIN')

latex_table = labels_train_df_2_3.to_latex( #labels_train_df_1_2.to_latex(
    index=False,          # remove index column
    float_format="%.1f",  # optional: format floats
    caption="Your caption",
    label="tab:your_label"
)
print(latex_table)

latex_table = labels_train_df_2_3_frac.to_latex( #labels_train_df_1_2_frac.to_latex(
    index=False,          # remove index column
    float_format="%.1f",  # optional: format floats
    caption="Your caption",
    label="tab:your_label"
)
print(latex_table)

'''latex_table = labels_train_df_3.to_latex(
    index=False,          # remove index column
    float_format="%.3f",  # optional: format floats
    caption="Your caption",
    label="tab:your_label"
)
print(latex_table)

latex_table = labels_train_df_3_frac.to_latex(
    index=False,          # remove index column
    float_format="%.3f",  # optional: format floats
    caption="Your caption",
    label="tab:your_label"
)
print(latex_table)'''

print('\n\n')


# dev

print('DEV')

latex_table = labels_dev_df_2_3.to_latex( #labels_dev_df_2_3.to_latex(
    index=False,          # remove index column
    float_format="%.1f",  # optional: format floats
    caption="Your caption",
    label="tab:your_label"
)
print(latex_table)

latex_table = labels_dev_df_2_3_frac.to_latex( #labels_dev_df_1_2_frac.to_latex(
    index=False,          # remove index column
    float_format="%.1f",  # optional: format floats
    caption="Your caption",
    label="tab:your_label"
)
print(latex_table)

'''latex_table = labels_dev_df_3.to_latex(
    index=False,          # remove index column
    float_format="%.3f",  # optional: format floats
    caption="Your caption",
    label="tab:your_label"
)
print(latex_table)

latex_table = labels_dev_df_3_frac.to_latex(
    index=False,          # remove index column
    float_format="%.3f",  # optional: format floats
    caption="Your caption",
    label="tab:your_label"
)
print(latex_table)'''

print('\n\n')


# test

print('TEST')

latex_table = labels_test_df_2_3.to_latex( #labels_test_df_1_2.to_latex(
    index=False,          # remove index column
    float_format="%.1f",  # optional: format floats
    caption="Your caption",
    label="tab:your_label"
)
print(latex_table)

latex_table = labels_test_df_2_3_frac.to_latex( #labels_test_df_1_2_frac.to_latex(
    index=False,          # remove index column
    float_format="%.1f",  # optional: format floats
    caption="Your caption",
    label="tab:your_label"
)
print(latex_table)

'''latex_table = labels_test_df_3.to_latex(
    index=False,          # remove index column
    float_format="%.3f",  # optional: format floats
    caption="Your caption",
    label="tab:your_label"
)
print(latex_table)

latex_table = labels_test_df_3_frac.to_latex(
    index=False,          # remove index column
    float_format="%.3f",  # optional: format floats
    caption="Your caption",
    label="tab:your_label"
)
print(latex_table)'''

print('\n\n')

# dev df

'''dev_files = os.listdir(DEV_PATH)

# read and concatenate dev df, compute percentage of each label per language and in total
tot_dev_df = pd.DataFrame()
labels_dev_df = pd.DataFrame()

for file in dev_files:

    dev_df = pd.read_csv(DEV_PATH + file)

    if file[:3] in LIST_LANG_NO_S3:
        dict_lang_sum_dev = dev_df[LABELS_LIST_NO_S3].sum(axis=0).to_dict()
    else:
        dict_lang_sum_dev = dev_df[LABELS_LIST].sum(axis=0).to_dict()
    dict_lang_sum_dev['lang'] = file[:3]

    labels_dev_df = pd.concat([labels_dev_df, pd.DataFrame(dict_lang_sum_dev, index=[0])])

    tot_dev_df = pd.concat([tot_dev_df, dev_df])

dict_tot_sum_dev = tot_dev_df[LABELS_LIST].sum(axis=0).to_dict()
dict_tot_sum_dev['lang'] = 'tot'
labels_dev_df = labels_dev_df._append(dict_tot_sum_dev)


# test df

test_files = os.listdir(TEST_PATH)

# read and concatenate train df
tot_test_df = pd.DataFrame()
labels_test_df = pd.DataFrame()

for file in test_files:

    test_df = pd.read_csv(TEST_PATH + file)

    if file[:3] in LIST_LANG_NO_S3:
        dict_lang_sum_test = test_df[LABELS_LIST_NO_S3].sum(axis=0).to_dict()
    else:
        dict_lang_sum_test = test_df[LABELS_LIST].sum(axis=0).to_dict()
    dict_lang_sum_test['lang'] = file[:3]

    labels_test_df = pd.concat([labels_test_df, pd.DataFrane(dict_lang_sum_test, index=[0])])

    tot_test_df = pd.concat([tot_test_df, test_df])

dict_tot_sum_test = tot_test_df[LABELS_LIST].sum(axis=0).to_dict()
dict_tot_sum_test['lang'] = 'tot'
labels_test_df = labels_test_df._append(dict_tot_sum_test)



print('Lenght train:', len(tot_train_df))
#print('Length dev:', len(tot_dev_df))
#print('Length test:', len(test_df))
# percentage train - dev
len_train = len(tot_train_df)
#len_dev = len(tot_dev_df)
len_train_dev = len_train + len_dev
perc_train = len_train / len_train_dev
perc_dev = len_dev / len_train_dev
print(f'Percentage train: {perc_train:.3f} - {perc_dev:.3f}')



print('labels df train', labels_train_df)
print('labels df dev', labels_dev_df)
print('labels df test', labels_test_df)



# DATA AFTER OUR SPLIT : size train, dev, test

# train df

train_files = os.listdir(TRAIN_PATH_SPLIT)

# read and concatenate train df
tot_train_df = pd.DataFrame()

for file in train_files:
    train_df = pd.read_csv(TRAIN_PATH_SPLIT + file, sep='\t')
    tot_train_df = pd.concat([tot_train_df, train_df])


# dev df

dev_files = os.listdir(DEV_PATH_SPLIT)

# read and concatenate train df
tot_dev_df = pd.DataFrame()

for file in dev_files:
    dev_df = pd.read_csv(DEV_PATH_SPLIT + file, sep='\t')
    tot_dev_df = pd.concat([tot_dev_df, dev_df])


# test df

test_files = os.listdir(TEST_PATH)

# read and concatenate train df
tot_test_df = pd.DataFrame()

for file in test_files:
    test_df = pd.read_csv(TEST_PATH + file)
    tot_test_df = pd.concat([tot_test_df, test_df])


print('Lenght train:', len(tot_train_df))
print('Length dev:', len(tot_dev_df))
print('Length test:', len(test_df))
# percentage train - dev
len_train = len(tot_train_df)
len_dev = len(tot_dev_df)
len_train_dev = len_train + len_dev
perc_train = len_train / len_train_dev
perc_dev = len_dev / len_train_dev
print(f'Percentage train: {perc_train:.3f} - {perc_dev:.3f}')





# DISTRIBUTION OF LABELS

# train
'''