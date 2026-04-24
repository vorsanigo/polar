import os
import pandas as pd
from sklearn.metrics import f1_score
from pathlib import Path
import numpy as np


# classes
CLASSES_1 = ['polarization']
CLASSES_2 = ['political', 'racial/ethnic', 'religious', 'gender/sexual', 'other']
CLASSES_3 = ['stereotype', 'vilification', 'dehumanization', 'extreme_language', 'lack_of_empathy', 'invalidation']

# columns
COLUMNS_TOT = ['text'] + CLASSES_1 + CLASSES_2 + CLASSES_3
COLUMNS_TOT_ID_TEXT = ['id', 'text'] + CLASSES_1 + CLASSES_2 + CLASSES_3

# submission folders
SUBMISSION_DEV_DIR = "submission_dev/"
SUBMISSION_TEST_DIR = "submission_test/"

# languages list
LANG_LIST = ['amh', 'arb', 'ben', 'deu', 'eng', 'fas', 'hau', 'hin', 'ita', 'khm', 'mya', 'nep', 'ori', 'pan', 'pol', 'rus', 'spa', 'swa', 'tel', 'tur', 'urd', 'zho']
LANG_ALL_TASKS = ['amh', 'arb', 'ben', 'deu', 'eng', 'fas', 'hau', 'hin', 'khm', 'nep', 'ori', 'pan', 'spa', 'swa', 'tel', 'tur', 'urd', 'zho']
LANG_NO_TASK_3 = ['ita', 'pol', 'rus', 'mya']
LANG_MULTILINGUAL = ['amh', 'arb', 'ben', 'deu', 'fas', 'hau', 'hin', 'khm', 'nep', 'ori', 'pan', 'swa', 'tel', 'tur', 'urd', 'zho']

# y_true: list/array of gold labels for this class
# y_pred: list/array of predicted labels for this class

def compute_f1_macro_2(y_true, y_pred, classes):

    f1_per_class = []

    for class_name in classes:
        y_true_class = y_true[class_name]
        y_pred_class = y_pred[class_name]
        f1_class = f1_score(y_true_class, y_pred_class)#, average='macro'
        f1_per_class.append(f1_class)

    # Global macro F1
    global_f1 = sum(f1_per_class) / len(f1_per_class)

    return global_f1


def compute_f1_macro(y_true, y_pred, classes):

    if len(classes) == 0:
        class_name = classes[0]
        print(np.array(list(y_true[class_name])))
        print(np.array(list(y_pred[class_name])))
        global_f1 = f1_score(np.array(list(y_true[class_name])), np.array(list(y_pred[class_name])), average='macro')
    
    else:

        y_true_classes = []
        y_pred_classes = []

        for class_name in classes:
            y_true_classes += [list(y_true[class_name])]
            y_pred_classes += [list(y_pred[class_name])]

        #print(y_true_classes)
        #print(y_pred_classes)
        '''print('\n')
        print(np.array(y_true_classes).T)
        print(np.array(y_true_classes).T.shape)
        print(np.array(y_pred_classes).T)
        print(np.array(y_pred_classes).T.shape)'''
        
        # Global macro F1
        global_f1 = f1_score(np.array(y_true_classes).T, np.array(y_pred_classes).T, average='macro')


    return global_f1


def create_submission_file_dev(logs_df_path, lang_df_path, logs_for_save, prediction_columns, subtask):

    logs_df = pd.read_csv(logs_df_path, sep='\t', header=None)
    lang_df = pd.read_csv(lang_df_path)
    lang_df = lang_df.drop(columns=prediction_columns)

    logs_df.columns = COLUMNS_TOT_ID_TEXT

    sub_lang_df = pd.merge(lang_df, logs_df, on='id', how='left')

    print(sub_lang_df.columns)

    sub_columns = ['id'] + prediction_columns

    sub_submission_df = sub_lang_df[sub_columns]

    output_dir = Path(SUBMISSION_DEV_DIR + subtask + '/' + logs_for_save)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = SUBMISSION_DEV_DIR + subtask + '/' + logs_for_save + '/'

    sub_submission_df.to_csv(output_path + lang + '.csv')


def create_submission_file_test(pred_df_path, lang_df_path, prediction_columns, subtask):
    print('pred_path', pred_df_path)
    pred_df = pd.read_csv(pred_df_path, sep='\t', header=None)
    lang_df = pd.read_csv(lang_df_path)
    lang_df = lang_df.drop(columns=prediction_columns)

    pred_df.columns = COLUMNS_TOT_ID_TEXT
    print(pred_df)
    print(lang_df)

    sub_lang_df = pd.merge(lang_df, pred_df, on='id', how='left')

    print(sub_lang_df.columns)

    sub_columns = ['id'] + prediction_columns

    sub_submission_df = sub_lang_df[sub_columns]
    print(sub_submission_df)
    print('\n')

    output_dir = Path(SUBMISSION_TEST_DIR + subtask + '/' + pred_df_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = SUBMISSION_TEST_DIR + subtask + '/' + pred_df_path + '/'

    sub_submission_df.to_csv(output_path + lang + '.csv', index=False)


def merge_pred_multilingalual(folder_input, list_lang, folder_output, output_file):

    INPUT_DIR = folder_input
    OUTPUT_DIR = folder_output
    COLUMNS = [
        "subtask",
        "F1_macro",
        "configuration"
    ]

    output_df = pd.DataFrame(columns=COLUMNS)
    print(INPUT_DIR)

    for lang in list_lang:
        print(lang)

        df = pd.read_csv(os.path.join(INPUT_DIR, lang + '.csv'))

        df['lang'] = lang

        df = df[['lang'] + COLUMNS]

        print(df)

        df_empty_line = pd.DataFrame([['']*(len(COLUMNS)+1)], columns=['lang'] + COLUMNS)

        output_df = pd.concat([output_df, df, df_empty_line], ignore_index=True)

    output_df.to_csv(os.path.join(OUTPUT_DIR, output_file), index=False)







#TODO MODIFY PRED_DEV FOR TEST WITH GOLDEN LABELS
if __name__ == "__main__":

    task =  'pred_test' #'pred_test''pred_dev''submission_test'  #'submission_dev'  #'submission_test' 'concat_multilingual' #

    multilingual = False  #False  #
    

    if task == 'pred_dev':
        
        if multilingual == False:

            lang = 'pol'  #'ita'  #'eng'  #

            global_f1_1 = 'NA'
            global_f1_2 = 'NA'
            global_f1_3 = 'NA'

            y_pred_path_1 = 'NA'
            y_pred_path_2 = 'NA'
            y_pred_path_3 = 'NA'
            
            # on dev set labels
            y_true_path = "dev_phase/merged subtasks split dev/merged_dev_" + lang + "_split_shuffled.tsv"
            #y_true_path = "dev_phase/merged subtasks split dev no multi/merged_dev_spa_split_shuffled.tsv"

            # subtask 1

            #y_pred_path_1 = "logs/eng_1_2_3_parallel_s1_v1/2026.01.25_17.56.50/POLAR_ENG.out"
            
            # prediction on dev set
            y_pred_path_1 = "dev_prediction/" + lang + "_1_s1_v1_our_train.tsv"

            # on dev
            y_true = pd.read_csv(y_true_path, sep='\t')
            y_pred = pd.read_csv(y_pred_path_1, sep='\t', header=None)

            print('true', y_true)
            print('pred', y_pred)

            # drop unnamed column
            '''print(y_pred)
            print(COLUMNS_TOT)
            print(y_true)'''
            # rename columns
            y_pred.columns = COLUMNS_TOT_ID_TEXT
            print('pred', y_pred)

            y_true = y_true.sort_values(by='id')
            y_pred = y_pred.sort_values(by='id')
            global_f1_1 = compute_f1_macro(y_true, y_pred, CLASSES_1)
            #global_f1_1 = compute_f1_macro_2(y_true, y_pred, CLASSES_1)


            # subtask 2
            
            '''#y_pred_path_2 = "logs/eng_1_2_3_parallel_s2_v1/2026.01.25_18.21.12/POLAR_ENG.out"
            y_pred_path_2 = "dev_prediction/" + lang + "_1_2_3_parallel_s1_s2_s3_v1_our_train_post_sub1_depending.tsv"

            y_true = pd.read_csv(y_true_path, sep='\t')
            y_pred = pd.read_csv(y_pred_path_2, sep='\t', header=None)

            # drop unnamed column
            #y_true = y_true.drop(columns='Unnamed: 0')
            #y_pred = y_pred.drop(columns=y_pred.columns[0])
            # rename columns
            y_pred.columns = COLUMNS_TOT_ID_TEXT

            y_true = y_true.sort_values(by='id')
            #y_true = y_true.drop(columns='Unnamed: 0')
            y_pred = y_pred.sort_values(by='id')
            global_f1_2 = compute_f1_macro(y_true, y_pred, CLASSES_2)
            #global_f1_2_1 = compute_f1_macro_2(y_true, y_pred, CLASSES_2)'''


            # subtask 3

            '''#y_pred_path_3 = "logs/eng_1_2_3_parallel_s3_v1/2026.01.25_18.27.06/POLAR_ENG.out"
            y_pred_path_3 = "dev_prediction/" + lang + "_1_2_3_parallel_s1_s2_s3_v1_our_train_post_sub1_depending.tsv"

            y_true = pd.read_csv(y_true_path, sep='\t')
            #y_true = y_true.drop(columns='Unnamed: 0')
            y_pred = pd.read_csv(y_pred_path_3, sep='\t', header=None)
            
            # drop unnamed column
            #y_true = y_true.drop(columns='Unnamed: 0')
            #y_pred = y_pred.drop(columns=y_pred.columns[0])
            # rename columns
            y_pred.columns = COLUMNS_TOT_ID_TEXT

            y_true = y_true.sort_values(by='id')
            y_pred = y_pred.sort_values(by='id')
            global_f1_3 = compute_f1_macro(y_true, y_pred, CLASSES_3)'''
            #global_f1_3_1 = compute_f1_macro_2(y_true, y_pred, CLASSES_3)

            # save F1 macro scores
            scores = {'subtask': [1, 2, 3], 'F1_macro': [global_f1_1, global_f1_2, global_f1_3], 'configuration': [y_pred_path_1, y_pred_path_2, y_pred_path_3]}
            df_scores = pd.DataFrame(scores)
            df_scores.to_csv('F1_macro_score_dev/' + lang + '_1_s1_v1_our_train.csv', index=False)
            print(global_f1_1)
            print(global_f1_2)
            print(global_f1_3)
            print('\n')
        
        else:
            
            y_pred_path_1 = "dev_prediction/multilingual_1_2_3_parallel_weighted_v2_2_NO_S3_test.tsv"
            #y_pred_path_2 = "dev_prediction/multilingual_1_3_parallel_weighted_v2_2_NO_S3_test.tsv"
            y_pred_path_3 = "dev_prediction/multilingual_1_2_3_parallel_weighted_v2_2_NO_S3_test.tsv"
            
            
            '''y_pred_path_1 = "dev_prediction_original_format/spa_seq_1_alone_2_3_parallel_s2_weighted_multi_s1.tsv"
            y_pred_path_2 = "dev_prediction_original_format/spa_seq_1_alone_2_3_parallel_s2_weighted_multi.tsv"
            y_pred_path_3 = "dev_prediction_original_format/spa_seq_1_alone_2_3_parallel_s2_weighted_multi.tsv"'''

            for lang in LANG_ALL_TASKS:#LANG_LIST
                
                global_f1_1 = 'NA'
                global_f1_2 = 'NA'
                global_f1_3 = 'NA'

                y_pred_path_1 = 'NA'
                y_pred_path_2 = 'NA'
                y_pred_path_3 = 'NA'

                y_true_path = 'dev_phase/merged subtasks split dev/merged_dev_' + lang + '_split_shuffled.tsv'
                
                y_true = pd.read_csv(y_true_path, sep='\t')

                y_true_id = list(y_true['id'])
                print('true', y_true_path)
                print(y_true)

                
                # subtask 1

                #y_pred_path_1 = "logs/eng_1_2_3_parallel_s1_v1/2026.01.25_17.56.50/POLAR_ENG.out"
                y_pred_path_1 = "dev_prediction/multilingual_1_2_3_parallel_weighted_v2_2_NO_S3_test.tsv"
                y_pred_1 = pd.read_csv(y_pred_path_1, sep='\t', header=None)
                print('pred path:', y_pred_path_1)
                print(y_pred_1)

                # drop unnamed column
                '''y_true = y_true.drop(columns='Unnamed: 0')
                y_pred = y_pred.drop(columns=y_pred.columns[0])'''
                # rename columns
                y_pred_1.columns = COLUMNS_TOT_ID_TEXT
                y_pred_1 = y_pred_1[y_pred_1['id'].isin(y_true_id)]
                y_pred_1_id = list(y_pred_1['id'])

                y_true_1 = y_true[y_true['id'].isin(y_pred_1_id)]

                y_true_1 = y_true_1.sort_values(by='id')
                y_pred_1 = y_pred_1.sort_values(by='id')
                global_f1_1 = compute_f1_macro(y_true_1, y_pred_1, CLASSES_1)
                #global_f1_1 = compute_f1_macro_2(y_true, y_pred, CLASSES_1)


                # subtask 2
            
                '''#y_pred_path_2 = "logs/eng_1_2_3_parallel_s2_v1/2026.01.25_18.21.12/POLAR_ENG.out"
                y_pred_path_2 = "dev_prediction/multilingual_1_2_3_parallel_s1_s2_s3_v1_our_train_post_sub1_depending.tsv"
                y_pred_2 = pd.read_csv(y_pred_path_2, sep='\t', header=None)
                

                # drop unnamed column
                #y_true = y_true.drop(columns='Unnamed: 0')
                #y_pred = y_pred.drop(columns=y_pred.columns[0])
                # rename columns
                y_pred_2.columns = COLUMNS_TOT_ID_TEXT
                y_pred_2 = y_pred_2[y_pred_2['id'].isin(y_true_id)]
                y_pred_2_id = list(y_pred_2['id'])

                y_true_2 = y_true[y_true['id'].isin(y_pred_2_id)]

                y_true_2 = y_true_2.sort_values(by='id')
                y_pred_2 = y_pred_2.sort_values(by='id')
                global_f1_2 = compute_f1_macro(y_true_2, y_pred_2, CLASSES_2)'''
                #global_f1_2_1 = compute_f1_macro_2(y_true, y_pred, CLASSES_2)


                # subtask 3

                if lang in LANG_ALL_TASKS:

                    #y_pred_path_3 = "logs/eng_1_2_3_parallel_s3_v1/2026.01.25_18.27.06/POLAR_ENG.out"
                    y_pred_path_3 = "dev_prediction/multilingual_1_2_3_parallel_weighted_v2_2_NO_S3_test.tsv"
                    y_pred_3 = pd.read_csv(y_pred_path_3, sep='\t', header=None)
                    print(y_true)
                    
                    
                    # drop unnamed column
                    #y_true = y_true.drop(columns='Unnamed: 0')
                    #y_pred = y_pred.drop(columns=y_pred.columns[0])
                    # rename columns
                    y_pred_3.columns = COLUMNS_TOT_ID_TEXT
                    y_pred_3 = y_pred_3[y_pred_3['id'].isin(y_true_id)]
                    y_pred_3[CLASSES_3] = y_pred_3[CLASSES_3].astype(int)
                    print(y_pred_3)
                    y_pred_3_id = list(y_pred_3['id'])

                    y_true_3 = y_true[y_true['id'].isin(y_pred_3_id)]
                    
                    y_true_3 = y_true_3.sort_values(by='id')
                    y_pred_3 = y_pred_3.sort_values(by='id')


                    global_f1_3 = compute_f1_macro(y_true_3, y_pred_3, CLASSES_3)
                    #global_f1_3_1 = compute_f1_macro_2(y_true, y_pred, CLASSES_3)

                # save F1 macro scores
                scores = {'subtask': [1, 2, 3], 'F1_macro': [global_f1_1, global_f1_2, global_f1_3], 'configuration': [y_pred_path_1, y_pred_path_2, y_pred_path_3]}
                df_scores = pd.DataFrame(scores)
                df_scores.to_csv('F1_macro_score_dev/multilingual_1_2_3_parallel_weighted_v2_2_NO_S3_test/' + lang + '.csv', index=False)
                print(global_f1_1)
                print(global_f1_2)
                print(global_f1_3)
                print('\n')

    elif task == 'pred_test':
        
        if multilingual == False:

            lang = 'eng'  #'ita'  #'eng' 'pol' 'spa'

            global_f1_1 = 'NA'
            global_f1_2 = 'NA'
            global_f1_3 = 'NA'

            y_pred_path_1 = 'NA'
            y_pred_path_2 = 'NA'
            y_pred_path_3 = 'NA'
            

            # on test set
            y_true_path = "tot_data_semeveal/test/" + lang + ".csv"

            # subtask 1

            # prediction on test set
            y_pred_path_1 = "test_prediction_single/" + lang + "_1_s1_v1_2.tsv"

            # on test
            y_true = pd.read_csv(y_true_path)
            y_pred = pd.read_csv(y_pred_path_1, sep='\t', header=None)

            print('true', y_true)
            print('pred', y_pred)

            # rename columns
            y_pred.columns = COLUMNS_TOT_ID_TEXT
            print('pred', y_pred)
            print(y_true.columns)

            y_true = y_true.sort_values(by='id')
            y_pred = y_pred.sort_values(by='id')
            global_f1_1 = compute_f1_macro(y_true, y_pred, CLASSES_1)

            # subtask 2
            
            #y_pred_path_2 = "logs/eng_1_2_3_parallel_s2_v1/2026.01.25_18.21.12/POLAR_ENG.out"
            y_pred_path_2 = "test_prediction_single/" + lang + "_2_s2_v1_2.tsv"

            y_true = pd.read_csv(y_true_path)
            y_pred = pd.read_csv(y_pred_path_2, sep='\t', header=None)

            # drop unnamed column
            #y_true = y_true.drop(columns='Unnamed: 0')
            #y_pred = y_pred.drop(columns=y_pred.columns[0])
            # rename columns
            y_pred.columns = COLUMNS_TOT_ID_TEXT

            y_true = y_true.sort_values(by='id')
            #y_true = y_true.drop(columns='Unnamed: 0')
            y_pred = y_pred.sort_values(by='id')
            global_f1_2 = compute_f1_macro(y_true, y_pred, CLASSES_2)
            #global_f1_2_1 = compute_f1_macro_2(y_true, y_pred, CLASSES_2)


            # subtask 3

            y_pred_path_3 = "logs/eng_1_2_3_parallel_s3_v1/2026.01.25_18.27.06/POLAR_ENG.out"
            y_pred_path_3 = "test_prediction_single/" + lang + "_3_s3_v1_2.tsv"

            y_true = pd.read_csv(y_true_path)
            #y_true = y_true.drop(columns='Unnamed: 0')
            y_pred = pd.read_csv(y_pred_path_3, sep='\t', header=None)
            
            # drop unnamed column
            #y_true = y_true.drop(columns='Unnamed: 0')
            #y_pred = y_pred.drop(columns=y_pred.columns[0])
            # rename columns
            y_pred.columns = COLUMNS_TOT_ID_TEXT

            y_true = y_true.sort_values(by='id')
            y_pred = y_pred.sort_values(by='id')
            global_f1_3 = compute_f1_macro(y_true, y_pred, CLASSES_3)
            #global_f1_3_1 = compute_f1_macro_2(y_true, y_pred, CLASSES_3)

            # save F1 macro scores
            scores = {'subtask': [1, 2, 3], 'F1_macro': [global_f1_1, global_f1_2, global_f1_3], 'configuration': [y_pred_path_1, y_pred_path_2, y_pred_path_3]}
            df_scores = pd.DataFrame(scores)
            df_scores.to_csv('F1_macro_score_test/' + lang + '_single_task_0.csv', index=False)
            print(global_f1_1)
            print(global_f1_2)
            print(global_f1_3)
            print('\n')
        
        else:
            
            y_pred_path_1 = "test_prediction_single/multilingual_1_s1_v1_2.tsv"
            y_pred_path_2 = "dev_prediction_single/multilingual_2_s2_v1_2.tsv"
            y_pred_path_3 = "dev_prediction_single/multilingual_3_s3_v1_2.tsv"
            
            
            '''y_pred_path_1 = "dev_prediction_original_format/spa_seq_1_alone_2_3_parallel_s2_weighted_multi_s1.tsv"
            y_pred_path_2 = "dev_prediction_original_format/spa_seq_1_alone_2_3_parallel_s2_weighted_multi.tsv"
            y_pred_path_3 = "dev_prediction_original_format/spa_seq_1_alone_2_3_parallel_s2_weighted_multi.tsv"'''

            for lang in LANG_ALL_TASKS:#LANG_LIST
                
                global_f1_1 = 'NA'
                global_f1_2 = 'NA'
                global_f1_3 = 'NA'

                y_pred_path_1 = 'NA'
                y_pred_path_2 = 'NA'
                y_pred_path_3 = 'NA'

                y_true_path = 'tot_data_semeveal/test/' + lang + '.csv'
                
                y_true = pd.read_csv(y_true_path)

                y_true_id = list(y_true['id'])
                print('true', y_true_path)
                print(y_true)

                
                # subtask 1

                #y_pred_path_1 = "logs/eng_1_2_3_parallel_s1_v1/2026.01.25_17.56.50/POLAR_ENG.out"
                y_pred_path_1 = "test_prediction_single/multilingual_1_s1_v1_2.tsv"
                y_pred_1 = pd.read_csv(y_pred_path_1, sep='\t', header=None)
                print('pred path:', y_pred_path_1)
                print(y_pred_1)

                # drop unnamed column
                '''y_true = y_true.drop(columns='Unnamed: 0')
                y_pred = y_pred.drop(columns=y_pred.columns[0])'''
                # rename columns
                y_pred_1.columns = COLUMNS_TOT_ID_TEXT
                y_pred_1 = y_pred_1[y_pred_1['id'].isin(y_true_id)]
                y_pred_1_id = list(y_pred_1['id'])

                y_true_1 = y_true[y_true['id'].isin(y_pred_1_id)]

                y_true_1 = y_true_1.sort_values(by='id')
                y_pred_1 = y_pred_1.sort_values(by='id')
                global_f1_1 = compute_f1_macro(y_true_1, y_pred_1, CLASSES_1)
                #global_f1_1 = compute_f1_macro_2(y_true, y_pred, CLASSES_1)


                # subtask 2
            
                #y_pred_path_2 = "logs/eng_1_2_3_parallel_s2_v1/2026.01.25_18.21.12/POLAR_ENG.out"
                y_pred_path_2 = "test_prediction_single/multilingual_2_s2_v1_2.tsv"
                y_pred_2 = pd.read_csv(y_pred_path_2, sep='\t', header=None)
                

                # drop unnamed column
                #y_true = y_true.drop(columns='Unnamed: 0')
                #y_pred = y_pred.drop(columns=y_pred.columns[0])
                # rename columns
                y_pred_2.columns = COLUMNS_TOT_ID_TEXT
                y_pred_2 = y_pred_2[y_pred_2['id'].isin(y_true_id)]
                y_pred_2_id = list(y_pred_2['id'])

                y_true_2 = y_true[y_true['id'].isin(y_pred_2_id)]

                y_true_2 = y_true_2.sort_values(by='id')
                y_pred_2 = y_pred_2.sort_values(by='id')
                global_f1_2 = compute_f1_macro(y_true_2, y_pred_2, CLASSES_2)
                #global_f1_2_1 = compute_f1_macro_2(y_true, y_pred, CLASSES_2)


                # subtask 3

                if lang in LANG_ALL_TASKS:

                    #y_pred_path_3 = "logs/eng_1_2_3_parallel_s3_v1/2026.01.25_18.27.06/POLAR_ENG.out"
                    y_pred_path_3 = "test_prediction_single/multilingual_3_s3_v1_2.tsv"
                    y_pred_3 = pd.read_csv(y_pred_path_3, sep='\t', header=None)
                    print(y_true)
                    
                    
                    # drop unnamed column
                    #y_true = y_true.drop(columns='Unnamed: 0')
                    #y_pred = y_pred.drop(columns=y_pred.columns[0])
                    # rename columns
                    y_pred_3.columns = COLUMNS_TOT_ID_TEXT
                    y_pred_3 = y_pred_3[y_pred_3['id'].isin(y_true_id)]
                    y_pred_3[CLASSES_3] = y_pred_3[CLASSES_3].astype(int)
                    print(y_pred_3)
                    y_pred_3_id = list(y_pred_3['id'])

                    y_true_3 = y_true[y_true['id'].isin(y_pred_3_id)]
                    
                    y_true_3 = y_true_3.sort_values(by='id')
                    y_pred_3 = y_pred_3.sort_values(by='id')


                    global_f1_3 = compute_f1_macro(y_true_3, y_pred_3, CLASSES_3)
                    #global_f1_3_1 = compute_f1_macro_2(y_true, y_pred, CLASSES_3)

                # save F1 macro scores
                scores = {'subtask': [1, 2, 3], 'F1_macro': [global_f1_1, global_f1_2, global_f1_3], 'configuration': [y_pred_path_1, y_pred_path_2, y_pred_path_3]}
                df_scores = pd.DataFrame(scores)
                df_scores.to_csv('F1_macro_score_test/multilingual_single_task/' + lang + '.csv', index=False)
                print(global_f1_1)
                print(global_f1_2)
                print(global_f1_3)
                print('\n')
    

    elif task == 'submission_dev':

        subtasks = ['subtask1', 'subtask2', 'subtask3']


        logs_df_list = ["eng_1_2_3_parallel_s2_v1_tot_td"]
        dates_list = ["2026.01.25_11.23.20/POLAR_ENG.out"]

        lang_list = ['eng']

        for lang in lang_list:

            for i in range(len(subtasks)):

                subtask = subtasks[i]

                logs_df_path = "logs/" + logs_df_list[i] + '/' + dates_list[i]

                if subtask == 'subtask1':
                    prediction_columns = CLASSES_1
                elif subtask == 'subtask2':
                    prediction_columns = CLASSES_2
                elif subtask == 'subtask3':
                    prediction_columns = CLASSES_3

                lang_df_path = 'dev_phase/' + subtask + '/dev/' + lang + '.csv'

                create_submission_file_dev(logs_df_path, lang_df_path, logs_df_list[i], prediction_columns, subtask)
    

    elif task == 'submission_test':

        subtasks = ['subtask1', 'subtask2']# ,, 'subtask3' 'subtask2'


        pred_df_list = ["multilingual_1_s1_v1_2.tsv", "multilingual_2_s2_v1_2.tsv"]#, "multilingual_1_3_parallel_weighted_v2_2_test.tsv", "multilingual_1_2_parallel_weighted_v2_test.tsv", , ,, "multilingual_1_2_parallel_weighted_v2_test.tsv" "multilingual_1_2_3_parallel_s1_s2_s3_v2_test.tsv", "multilingual_1_2_3_parallel_s1_s2_s3_v2_test.tsv"
        #dates_list = ["2026.01.25_11.23.20/POLAR_ENG.out"]
        # multilingual_1_3_parallel_weighted_v2_2_test.tsv multilingual_1_3_parallel_weighted_v2_2_test.tsv

        lang_list = LANG_LIST # #LANG_NO_TASK_3 LANG_ALL_TASKS LANG_MULTILINGUAL #  ['pol']  # # LANG_ALL_TASKS 

        for lang in lang_list:
            print(lang)

            for i in range(len(subtasks)):

                subtask = subtasks[i]

                pred_path = pred_df_list[i]
                print(pred_path)

                #logs_df_path = "logs/" + logs_df_list[i] + '/' + dates_list[i]

                if subtask == 'subtask1':
                    prediction_columns = CLASSES_1
                elif subtask == 'subtask2':
                    prediction_columns = CLASSES_2
                elif subtask == 'subtask3':
                    prediction_columns = CLASSES_3

                pred_df_path = 'test_prediction_single/' + pred_path

                lang_df_path = 'dev_phase/' + subtask + '/test/' + lang + '.csv'

                create_submission_file_test(pred_df_path, lang_df_path, prediction_columns, subtask)
    
    elif task == 'concat_multilingual':
        
        #folder_input = 'F1_macro_score_dev/multilingual_1_2_3_parallel_s1_s2_s3_v1_our_train_post_sub1_depending/'
        folder_input = 'F1_macro_score_test/multilingual_single_task/'
        folder_output = 'F1_macro_score_test/'
        #output_file = 'multilingual_1_2_3_parallel_s1_s2_s3_v1_our_train_post_sub1_depending.tsv'
        output_file = 'multilingual_single_task.tsv'
        list_lang = LANG_ALL_TASKS#LANG_LIST

        merge_pred_multilingalual(folder_input, list_lang, folder_output, output_file)