import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np


# train & dev df preparation

DIR_TRAIN_1 = 'dev_phase/subtask1/train/'
DIR_TRAIN_2 = 'dev_phase/subtask2/train/'
DIR_TRAIN_3 = 'dev_phase/subtask3/train/'

DIR_DEV_1 = 'dev_phase/subtask1/dev/'
DIR_DEV_2 = 'dev_phase/subtask2/dev/'
DIR_DEV_3 = 'dev_phase/subtask3/dev/'

DIR_TEST_1 = 'dev_phase/subtask1/test/'
DIR_TEST_2 = 'dev_phase/subtask2/test/'
DIR_TEST_3 = 'dev_phase/subtask3/test/'



# classes
CLASSES_1 = ['polarization']
CLASSES_2 = ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
CLASSES_3 = ['stereotype', 'vilification', 'dehumanization', 'extreme_language', 'lack_of_empathy', 'invalidation']

# columns
COLUMNS_TOT = ['text'] + CLASSES_1 + CLASSES_2 + CLASSES_3
COLUMNS_TOT_ID_TEXT = ['id', 'text'] + CLASSES_1 + CLASSES_2 + CLASSES_3
COLUMNS_LABELS = CLASSES_1 + CLASSES_2 + CLASSES_3

# subtasks
subtasks = ['subtask1', 'subtask2', 'subtask3']


# languages of df
LANG_LIST = ['amh', 'arb', 'ben', 'deu', 'eng', 'fas', 'hau', 'hin', 'ita', 'khm', 'mya', 'nep', 'ori', 'pan', 'pol', 'rus', 'spa', 'swa', 'tel', 'tur', 'urd', 'zho']#'eng', 'ita'
LANG_ALL_TASKS = ['amh', 'arb', 'ben', 'deu', 'eng', 'fas', 'hau', 'hin', 'khm', 'nep', 'ori', 'pan', 'spa', 'swa', 'tel', 'tur', 'urd', 'zho']



def clean_text_machamp(text):
    '''Clean text to be MaChAmp-friendly'''

    if pd.isna(text):
        return ""
    # Convert to string
    text = str(text)
    # Remove newlines and tabs
    text = re.sub(r'[\n\r\t]+', ' ', text)
    # Remove double quotes
    text = text.replace('"', '')
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing spaces
    return text.strip()


def create_merged_df(dir_df_1, dir_df_2, dir_df_3, lang, lang_all_tasks):
    '''Merge df of the 3 subtasks (if all available), otherwise for the available subtasks'''

    # read df
    df_1 = pd.read_csv(dir_df_1 + lang + '.csv')
    df_2 = pd.read_csv(dir_df_2 + lang + '.csv')
    if lang in lang_all_tasks:
        #print(lang)
        df_3 = pd.read_csv(dir_df_3 + lang + '.csv')

    # merge dataframes
    df_tot = pd.merge(df_1, df_2, on=['id', 'text'])
    if lang in lang_all_tasks:
        #print(lang)
        df_tot = pd.merge(df_tot, df_3, on=['id', 'text'])
    else:
        df_tot[CLASSES_3] = "_"
    
    # clean text to be machamp friendly
    df_tot['text'] = df_tot['text'].apply(clean_text_machamp)

    return df_tot


def create_dev_train_machamp(dev_train_path, subtask, lang, output_dir):
    '''Needed format: all subtasks columns, tsv format'''

    dev_train_df = pd.read_csv(dev_train_path)

    if subtask == 'subtask1':   
        columns_keep = ['id', 'text'] + CLASSES_1
        new_cols = CLASSES_2 + CLASSES_3
        dev_train_keep_df = dev_train_df[columns_keep]
        dev_train_keep_df[new_cols] = "_"
    elif subtask == 'subtask2':   
        columns_keep = ['id', 'text'] + CLASSES_2
        new_cols = CLASSES_2 + CLASSES_3
        dev_train_keep_df = dev_train_df[columns_keep]
        dev_train_keep_df[new_cols] = "_"
    # add classes columns
    df_new_cols = CLASSES_1 + CLASSES_2 + CLASSES_3

    # save as tsv
    dev_train_df.to_csv(output_dir + lang + '.tsv', index=False, sep='\t')



def create_dev_test_machamp(dev_test_path, subtask, lang, output_dir):
    '''Needed format: all subtasks columns, tsv format, no labels but "_" to indicate empty'''

    dev_test_df = pd.read_csv(dev_test_path)

    # keep id, text
    dev_test_df = dev_test_df[['id', 'text']]

    # add classes columns
    df_new_cols = CLASSES_1 + CLASSES_2 + CLASSES_3
    dev_test_df[df_new_cols] = "_"

    # save as tsv
    dev_test_df.to_csv(output_dir + lang + '.tsv', index=False, sep='\t')




if __name__ == "__main__":

    # 1) create df for each lang with the labels of each task

    for lang in LANG_LIST:

        # 1) TRAIN DF

        # create merged train df
        df_train_tot = create_merged_df(DIR_TRAIN_1, DIR_TRAIN_2, DIR_TRAIN_3, lang, LANG_ALL_TASKS)
        print('lang', lang)

        # shuffle data to give classes labels unordered
        df_train_tot_shuffled = df_train_tot.sample(frac=1, random_state=42).reset_index(drop=True)

        # save merged train df
        df_train_tot_shuffled.to_csv('dev_phase/merged subtasks train/merged_train_' + lang + '_data_shuffled.tsv', index=False, sep='\t')


        # 2) DEV DF

        # create dev df
        df_dev_tot = create_merged_df(DIR_DEV_1, DIR_DEV_2, DIR_DEV_3, lang, LANG_ALL_TASKS)
        
        # shuffle data to give classes labels unordered
        df_dev_tot_shuffled = df_dev_tot.sample(frac=1, random_state=42).reset_index(drop=True)

        # save merged dev df
        df_dev_tot_shuffled.to_csv('dev_phase/merged subtasks dev/merged_dev_' + lang + '_data_shuffled.tsv', index=False, sep='\t')


        # 3) TEST DF

        # create test df
        df_test_tot = create_merged_df(DIR_TEST_1, DIR_TEST_2, DIR_TEST_3, lang, LANG_ALL_TASKS)

        # insert "_" because it is test, so no labels are given
        df_test_tot[COLUMNS_LABELS] = "_"

        # save merged test df
        df_test_tot.to_csv('dev_phase/merged subtasks test/merged_test_' + lang + '_data.tsv', index=False, sep='\t')
    

    # 2) concatenate and shuffle df of different languages to obtain a final multilingual df

    df_multilingual_train = pd.DataFrame()
    df_multilingual_dev = pd.DataFrame()
    df_multilingual_test = pd.DataFrame()

    for lang in LANG_LIST:

        # train
        df_train_lang = pd.read_csv('dev_phase/merged subtasks train/merged_train_' + lang + '_data_shuffled.tsv', sep='\t')
        df_multilingual_train = pd.concat([df_multilingual_train, df_train_lang])

        # dev
        df_dev_lang = pd.read_csv('dev_phase/merged subtasks dev/merged_dev_' + lang + '_data.tsv', sep='\t')
        df_multilingual_dev = pd.concat([df_multilingual_dev, df_dev_lang])

        # test
        df_test_lang = pd.read_csv('dev_phase/merged subtasks test/merged_test_' + lang + '_data.tsv', sep='\t')
        df_multilingual_test = pd.concat([df_multilingual_test, df_test_lang])

    df_multilingual_train = df_multilingual_train.fillna('_')
    df_multilingual_dev = df_multilingual_dev.fillna('_')
    df_multilingual_test = df_multilingual_test.fillna('_')

    # shuffle df
    df_multilingual_train = df_multilingual_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_multilingual_dev = df_multilingual_dev.sample(frac=1, random_state=42).reset_index(drop=True)

    # save df
    df_multilingual_train.to_csv('dev_phase/merged subtasks multilingual/merged_train_tot_data_shuffled.tsv', sep='\t', index=False)
    df_multilingual_dev.to_csv('dev_phase/merged subtasks multilingual/merged_dev_tot_data_shuffled.tsv', sep='\t', index=False)
    df_multilingual_test.to_csv('dev_phase/merged subtasks multilingual/merged_test_tot_data.tsv', sep='\t', index=False)


    # 3) for each total train set, create the pre-dev set (before dev sets are released)

    # test/dev percentage of training set
    test_size = 0.07

    dir_df = 'dev_phase/merged subtasks train/'
    dir_df_dev_original = 'dev_phase/merged subtasks dev/'
    #df_list = ['merged_train_eng_data_shuffled.tsv', 'merged_train_spa_data_shuffled.tsv', 'merged_train_tot_data_shuffled.tsv']#
    #df_list = ['merged_train_ita_data_shuffled.tsv']
    #df_list = ['merged_train_eng_it_spa_data_shuffled.tsv']
    output_dir_split_train = 'dev_phase/merged subtasks split train/'
    output_dir_split_dev = 'dev_phase/merged subtasks split dev/'
    output_file_multi_train = 'dev_phase/merged subtasks split multilingual/merged_train_tot_split_shuffled.tsv'
    output_file_multi_dev = 'dev_phase/merged subtasks split multilingual/merged_dev_tot_split_shuffled.tsv'

    labels_list = ['polarization', 'political', 'racial/ethnic', 'religious', 'gender/sexual', 'other', 'stereotype', 'vilification', 'dehumanization', 'extreme_language', 'lack_of_empathy', 'invalidation']
    labels_list_red = ['polarization', 'political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']

    train_multilingual_df = pd.DataFrame()
    dev_multilingual_df = pd.DataFrame()

    for lang in LANG_LIST:
        
        df = pd.read_csv(dir_df + 'merged_train_' + lang + '_data_shuffled.tsv', sep='\t', index_col=False)
        df_dev_original = pd.read_csv(dir_df_dev_original + 'merged_dev_' + lang + '_data_shuffled.tsv', sep='\t', index_col=False)
        # replace "_" with -1 to do the split
        label_cols = df.columns.difference(["id", "text"])
        df[label_cols] = df[label_cols].replace("_", -1)
        X = df[["id", "text"]].values

        try:
            y = df[labels_list].values.astype(float).astype(int)
            X_train, y_train, X_dev, y_dev = iterative_train_test_split(
                X, y, test_size=test_size#0.15
            )
            train_df = pd.DataFrame(X_train, columns=["id", "text"])
            dev_df   = pd.DataFrame(X_dev, columns=["id", "text"])
            train_df[labels_list] = y_train
            dev_df[labels_list] = y_dev
        except:
            y = df[labels_list_red].values.astype(float).astype(int)
            X_train, y_train, X_dev, y_dev = iterative_train_test_split(
                X, y, test_size=test_size#0.15
            )
            train_df = pd.DataFrame(X_train, columns=["id", "text"])
            dev_df   = pd.DataFrame(X_dev, columns=["id", "text"])
            train_df[labels_list_red] = y_train
            dev_df[labels_list_red] = y_dev

        train_df = train_df.replace(-1, '_')
        dev_df = dev_df.replace(-1, '_')

        df_dev_original = df_dev_original.reset_index(drop=True)
        dev_df = dev_df.reset_index(drop=True)
        dev_tot_df = pd.concat([df_dev_original, dev_df], ignore_index=True)
        dev_tot_df = dev_tot_df.reset_index(drop=True)

        train_multilingual_df = pd.concat([train_multilingual_df, train_df])
        dev_multilingual_df = pd.concat([dev_multilingual_df, dev_tot_df])

        train_df.to_csv(output_dir_split_train + 'merged_train_' + lang + '_split_shuffled.tsv', sep='\t', index=False)
        dev_tot_df.to_csv(output_dir_split_dev + 'merged_dev_' + lang + '_split_shuffled.tsv', sep='\t', index=False)
    
    train_multilingual_df.to_csv(output_file_multi_train)
    dev_multilingual_df.to_csv(output_file_multi_dev)


    # 4) eliminate multilingual with only 2 tasks

    INPUT_DIR = 'dev_phase/merged subtasks split multilingual'
    OUTPUT_DIR = 'dev_phase/merged subtasks split multilingual no empty S3'

    for file in os.listdir(INPUT_DIR):
        
        if file.endswith('.tsv'):
            df = pd.read_csv(os.path.join(INPUT_DIR, file), sep='\t')
            print(file)
            print(df)

            # create a mask for rows where subtask 2 is "_"
            mask = df['stereotype'] == "_"
            df = df[~mask]
            print(df)

            # ---- Save clean TSV ----
            df.to_csv(os.path.join(OUTPUT_DIR, file), sep="\t", index=False)

        print("Done! Saved to output_machamp.tsv")

    
    # 5) # put together dev and train to create bigger training set

    INPUT_DIR_DEV = 'dev_phase/merged subtasks split dev'
    INPUT_DIR_TRAIN = 'dev_phase/merged subtasks split train'    
    OUTPUT_DIR = 'dev_phase/merged subtasks train big' 
    LANG_LIST = ['amh', 'arb', 'ben', 'deu', 'eng', 'fas', 'hau', 'hin', 'ita', 'khm', 'mya', 'nep', 'ori', 'pan', 'pol', 'rus', 'spa', 'swa', 'tel', 'tur', 'urd', 'zho']


    for lang in LANG_LIST:

        dev_file = f'merged_dev_{lang}_split_shuffled.tsv'
        train_file = f'merged_train_{lang}_split_shuffled.tsv'

        df_dev = pd.read_csv(os.path.join(INPUT_DIR_DEV, dev_file), sep='\t')
        df_train = pd.read_csv(os.path.join(INPUT_DIR_TRAIN, train_file), sep='\t')

        df_combined = pd.concat([df_train, df_dev], ignore_index=True)

        output_file = f'merged_train_{lang}_data.tsv'

        df_combined.to_csv(os.path.join(OUTPUT_DIR, output_file), sep='\t', index=False)
        print(f'Done! Saved combined file for {lang} to {output_file}')


    multi_train = pd.read_csv('dev_phase/merged subtasks split multilingual/merged_train_tot_split_shuffled.tsv', sep='\t')
    multi_dev = pd.read_csv('dev_phase/merged subtasks split multilingual/merged_dev_tot_split_shuffled.tsv', sep='\t')

    multi_train_tot = pd.concat([multi_train, multi_dev], ignore_index=True)

    multi_train_tot.to_csv('dev_phase/merged subtasks multilingual/merged_train_tot_data_shuffled.tsv', sep='\t', index=False)


    multi_train_no_s3 = pd.read_csv('dev_phase/merged subtasks split multilingual no empty S3/merged_train_tot_split_shuffled.tsv', sep='\t')
    multi_dev_no_s3 = pd.read_csv('dev_phase/merged subtasks split multilingual no empty S3/merged_dev_tot_split_shuffled.tsv', sep='\t')

    multi_train_tot_no_s3 = pd.concat([multi_train_no_s3, multi_dev_no_s3], ignore_index=True)

    multi_train_tot_no_s3.to_csv('dev_phase/merged subtasks multilingual/merged_train_tot_data_shuffled_no_empty_s3.tsv', sep='\t', index=False)



        
    '''df = pd.read_csv('dev_phase/subtask1/dev/eng.csv')
    df['polarization'] = "_"
    df.to_csv('dev_phase/subtask1/dev/eng_for_machamp.tsv', index=False, sep='\t')'''

    '''CLASSES_1 = ['polarization']
    CLASSES_2 = ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
    CLASSES_3 = ['stereotype', 'vilification', 'dehumanization', 'extreme_language', 'lack_of_empathy', 'invalidation']

    df = pd.read_csv('dev_phase/merged subtasks split/merged_train_eng_data_shuffled_dev_split_shuffled.tsv', sep='\t')
    df_1 = pd.read_csv('dev_phase/merged subtasks split/merged_train_eng_data_shuffled_split_shuffled.tsv', sep='\t')
    print(df)
    print(df_1)
    drop = CLASSES_2 + CLASSES_3
    dff = df.drop(columns=drop)
    dff_1 = df_1.drop(columns=drop)
    dff.to_csv('dev_phase/merged subtasks split/merged_train_eng_data_shuffled_dev_split_shuffled_NEW.tsv', index=False, sep='\t')
    dff_1.to_csv('dev_phase/merged subtasks split/merged_train_eng_data_shuffled_split_shuffled_NEW.tsv', index=False, sep='\t')
'''

    '''# create dev df for training machamp friendly
    
    for subtask in subtasks:

        input_dir = 'dev_phase/' + subtask + '/dev/'

        files = os.listdir(input_dir)

        for file in files:

            lang = file[:3]
            output_dir = 'dev_phase/' + subtask + '/dev_machamp_training/'
            dev_test_path = input_dir + lang + '.csv'

            create_dev_train_machamp(dev_test_path, subtask, lang, output_dir)'''
    
    
    '''# create dev df for prediction machamp friendly

    for subtask in subtasks:

        input_dir = 'dev_phase/' + subtask + '/dev/'

        files = os.listdir(input_dir)

        for file in files:

            lang = file[:3]
            output_dir = 'dev_phase/' + subtask + '/dev_machamp_prediction/'
            dev_test_path = input_dir + lang + '.csv'

            create_dev_test_machamp(dev_test_path, subtask, lang, output_dir)'''

    
    








    '''# train

    # read df
    df_train_1 = pd.read_csv(dir_train_1 + lang + '.csv')
    df_train_2 = pd.read_csv(dir_train_2 + lang + '.csv')
    if lang in lang_all_tasks:
        df_train_3 = pd.read_csv(dir_train_3 + lang + '.csv')

    # merge dataframes
    df_train_tot = pd.merge(df_train_1, df_train_2, on=['id', 'text'])
    if lang in lang_all_tasks:
        df_train_tot = pd.merge(df_train_tot, df_train_3, on=['id', 'text'])
    
    # clean text to be machamp friendly
    df_train_tot['text'] = df_train_tot['text'].apply(clean_text_machamp)

    # save merged dataframe
    df_train_tot.to_csv('dev_phase/merged subtasks/merged_train_' + lang + '_data_prova.tsv', index=False, sep='\t')


    # dev

    # read df
    df_dev_1 = pd.read_csv(dir_dev_1 + lang + '.csv')
    df_dev_2 = pd.read_csv(dir_dev_2 + lang + '.csv')
    if lang in lang_all_tasks:
        df_dev_3 = pd.read_csv(dir_dev_3 + lang + '.csv')

    # merge dataframes
    df_dev_tot = pd.merge(df_dev_1, df_dev_2, on=['id', 'text'])
    if lang in lang_all_tasks:
        df_dev_tot = pd.merge(df_dev_tot, df_dev_3, on=['id', 'text'])
    
    # clean text to be machamp friendly
    df_train_tot['text'] = df_train_tot['text'].apply(clean_text_machamp)

    # save merged dataframe
    df_dev_tot.to_csv('dev_phase/merged subtasks/merged_dev_' + lang + '_data_prova.tsv', index=False, sep='\t')'''



# english

# train data
'''
df_train_eng_1 = pd.read_csv(dir_train_1 + 'ita.csv')
df_train_eng_2 = pd.read_csv(dir_train_2 + 'ita.csv')
#df_train_eng_3 = pd.read_csv(dir_train_3 + 'ita.csv')

# merge dataframes
df_train_eng_tot = pd.merge(df_train_eng_1, df_train_eng_2, on=['id', 'text'])
#df_train_eng_tot = pd.merge(df_train_eng_tot, df_train_eng_3, on=['id', 'text'])

# save merged dataframe
df_train_eng_tot.to_csv('dev_phase/merged subtasks/merged_train_ita_data.csv', index=False)


# dev data

df_dev_eng_1 = pd.read_csv(dir_dev_1 + 'ita.csv')
df_dev_eng_2 = pd.read_csv(dir_dev_2 + 'ita.csv')
#df_dev_eng_3 = pd.read_csv(dir_dev_3 + 'eng.csv')

# merge dataframes
df_dev_eng_tot = pd.merge(df_dev_eng_1, df_dev_eng_2, on=['id', 'text'])
#df_dev_eng_tot = pd.merge(df_dev_eng_tot, df_dev_eng_3, on=['id', 'text'])

# save merged dataframe
df_dev_eng_tot.to_csv('dev_phase/merged subtasks/merged_dev_ita_data.csv', index=False)


# save in tsv format for training
df_train = pd.read_csv('dev_phase/merged subtasks/merged_train_ita_data.csv')
df_train.to_csv('dev_phase/merged subtasks/merged_train_ita_data.tsv', index=False, sep='\t')
df_dev = pd.read_csv('dev_phase/merged subtasks/merged_dev_ita_data.csv')
df_dev.to_csv('dev_phase/merged subtasks/merged_dev_ita_data.tsv', index=False, sep='\t')'''

'''df_train = pd.read_csv('dev_phase/merged subtasks/merged_train_ita_data.tsv', sep='\t')
print(df_train.head(10))'''

print('Done')