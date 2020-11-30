"""
Extract and Preprocess PubMed abstracts or MIMIC-III reports
"""
import os
import re
import traceback
import pandas as pd

from nltk import sent_tokenize, word_tokenize


def pattern_repl(matchobj):
    """
    Return a replacement string: to be instead of pattern
    Args:
      matchobj: Regular expression match object
    Returns:
        str: replacement string(i.e. ' '(space))
    """
    return ' '.rjust(len(matchobj.group(0)))


def find_end(text: str):
    """
    Find the end of the report based on regular expressions.
    Args:
      text: Report texts
    Returns:
        int: index of last character
    """
    ends = [len(text)]
    patterns = [
        re.compile(r'BY ELECTRONICALLY SIGNING THIS REPORT', re.I),
        re.compile(r'\n {3,}DR.', re.I),
        re.compile(r'[ ]{1,}RADLINE ', re.I),
        re.compile(r'.*electronically signed on', re.I),
        re.compile(r'M\[0KM\[0KM')
    ]
    for pattern in patterns:
        matchobj = pattern.search(text)
        if matchobj:
            ends.append(matchobj.start())
    return min(ends)


def split_heading(text: str):
    """
    Split the report into sections
    Args:
      text: Report text
    Yields:
        str: texts found on section
    """

    start = 0
    for matcher in SECTION_TITLES.finditer(text):

        # add last
        end = matcher.start()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        # add title
        start = end
        end = matcher.end()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        start = end

    # add last piece
    end = len(text)
    if start < end:
        section = text[start:end].strip()
        if section:
            yield section


def clean_text(text: str):
    """
    Clean text: remove [**Patterns**] and signatures
    Args:
      text: Report text
    Returns:
        new_text: cleaned texts
    """

    # Replace [**Patterns**] with spaces.
    text = re.sub(r'\[\*\*.*?\*\*\]', pattern_repl, text)
    # Replace `_` with spaces.
    text = re.sub(r'_', ' ', text)

    #################Find the end of the report########################
    start = 0
    end = find_end(text)
    new_text = ''
    if start > 0:
        new_text += ' ' * start
    new_text = text[start:end]
    ###################################################################

    # make sure the new text has the same length of old text.
    if len(text) - end > 0:
        new_text += ' ' * (len(text) - end)
    return new_text


def preprocess_mimic(text: str):
    """
    Preprocess reports in MIMIC-III.
    1. remove [**Patterns**] and signature
    2. split the report into sections
    3. tokenize sentences and words
    4. lowercase
    Args:
      text: Report text
    Yields:
        str: tokenized sentence
    """
    for sec in split_heading(clean_text(text)):
        for sent in sent_tokenize(sec):
            text = ' '.join(word_tokenize(sent))
            yield text.lower()

def get_sentences(text: str):
    """
    Process reports and returns all sentences as list
    Args:
      text: Report texts
    Return:
        list: Processed sentences
    """
    return list(preprocess_mimic(text))

def extract_notes(patient_folders: list, dataset_path: str, df_notes: pd.core.frame.DataFrame, test: bool = 0,\
                      patient_file_name: str = 'notes', all_notes_file_name: str = 'all_notes'):
    """
    Extract notes for each patient and saves in csv files.
    1. All notes in all_notes.csv,
    2. saves notes.csv for each patient in patient folders.
    Args:
      patient_folders:
        A list of folder name of(patients/subject ids)
      dataset_path:
        Path which contains the extracted patient folders from MIMIC
      df_notes:
        Dataframe containing clinical notes
      test:
        Boolean value: If true saves output for two patients and halts the process
      patient_file_name:
        csv name to save in patient folder
      all_notes_file_name:
        csv name to save all notes
    Returns: None
    """

    suceed = 0
    failed = 0
    failed_exception = 0
    notes_all = None

    if patient_file_name == 'notes':
        print('\nExtracting notes for patients...')
        notes_all = pd.DataFrame(columns=['SUBJECT_ID', 'ICUSTAY_ID',\
                     'HADM_ID', 'CATEGORY', 'CHARTTIME', 'TEXT'])

    else:
        print('\nExtracting notes for patients with no charttime...')
        notes_all = pd.DataFrame(columns=['SUBJECT_ID', 'ICUSTAY_ID',\
                     'HADM_ID', 'CATEGORY', 'CHARTDATE', 'TEXT'])

    for folder in patient_folders:
        try:

            ################### Check if note exist ##########
            patient_id = int(folder)
            sliced = df_notes[df_notes.SUBJECT_ID == patient_id]
            if sliced.shape[0] == 0:
                print("No notes for PATIENT_ID : {}".format(patient_id))
                failed += 1
                continue

            ###############################################################################

            ################### get the HADM_IDs, ICUSTAY_IDs from the stays.csv##########
            stays_path = os.path.join(dataset_path, folder, 'stays.csv')
            stays_df = pd.read_csv(stays_path)
            hadm_ids = list(stays_df.HADM_ID.values)
            icu_stay_ids = list(stays_df.ICUSTAY_ID.values)
            if patient_file_name == 'notes':
                df_patient = pd.DataFrame(columns=['ICUSTAY_ID', 'HADM_ID', 'CATEGORY', 'CHARTTIME', 'TEXT'])
                sliced.sort_values(by='CHARTTIME')
            else:
                df_patient = pd.DataFrame(columns=['ICUSTAY_ID', 'HADM_ID', 'CATEGORY', 'CHARTDATE', 'TEXT'])
                sliced.sort_values(by='CHARTDATE')
            ###############################################################################

            ##########################Get Patient notes####################################
            for ind, hid in enumerate(hadm_ids):

                icu_stay_id = icu_stay_ids[ind]
                sliced = sliced[sliced.HADM_ID == hid]

                for index, row in sliced.iterrows():

                    sentences = get_sentences(row['TEXT'])

                    if patient_file_name == 'notes':
                        df_patient = df_patient.append({'ICUSTAY_ID':icu_stay_id, 'HADM_ID': hid, 'CATEGORY':row.CATEGORY, 'CHARTTIME': row['CHARTTIME'],\
                                     'TEXT': sentences}, ignore_index=True)

                        notes_all = notes_all.append({'SUBJECT_ID': patient_id, 'ICUSTAY_ID':icu_stay_id, 'HADM_ID': hid,\
                                     'CATEGORY':row.CATEGORY, 'CHARTTIME': row['CHARTTIME'], 'TEXT': sentences}, ignore_index=True)
                    else:
                        df_patient = df_patient.append({'ICUSTAY_ID':icu_stay_id, 'HADM_ID': hid, 'CATEGORY':row.CATEGORY,\
                                     'CHARTDATE': row['CHARTDATE'], 'TEXT': sentences}, ignore_index=True)

                        notes_all = notes_all.append({'SUBJECT_ID': patient_id, 'ICUSTAY_ID':icu_stay_id, 'HADM_ID': hid,\
                                     'CATEGORY':row.CATEGORY, 'CHARTDATE': row['CHARTDATE'], 'TEXT': sentences}, ignore_index=True)
            ##########################################################################

            #######################Save in patient folder#############################
            path = os.path.join(output_folder, folder)
            if not os.path.exists(path):
                os.makedirs(path)
            df_patient.to_csv(path+'/'+patient_file_name+'.csv')
            ###########################################################################

            #######################Test Output for 2 patients if needed#############################
            suceed += 1
            if test and suceed == 2:
                print('Succeed for {} patients:'.format(suceed))
                notes_all.to_csv(os.path.join(output_folder)+all_notes_file_name+'.csv')
                break
            ###########################################################################
        except:
            traceback.print_exc()
            print("Failed with Exception FOR Patient ID: %s", folder)
            failed_exception += 1

    notes_all.to_csv(os.path.join(output_folder)+all_notes_file_name+'.csv')
    print("Sucessfully Completed: %d/%d" % (suceed, len(patient_folders)))
    print("No Notes for Patients: %d/%d" % (failed, len(patient_folders)))
    print("Failed with Exception: %d/%d" % (failed_exception, len(patient_folders)))



# Different sections in a report to look for
SECTION_TITLES = re.compile(
    r'('
    r'ABDOMEN AND PELVIS|CLINICAL HISTORY|CLINICAL INDICATION|COMPARISON|COMPARISON STUDY DATE'
    r'|EXAM|EXAMINATION|FINDINGS|HISTORY|IMPRESSION|INDICATION'
    r'|MEDICAL CONDITION|PROCEDURE|REASON FOR EXAM|REASON FOR STUDY|REASON FOR THIS EXAMINATION'
    r'|TECHNIQUE'
    r'):|FINAL REPORT',
    re.I | re.M)

if __name__ == "__main__":

    #path to mimic NOTEEVENTS.csv
    mimic_note_path = '../../../../../data/mimic/csv/NOTEEVENTS.csv'
    df = pd.read_csv(mimic_note_path)
    df.CHARTDATE = pd.to_datetime(df.CHARTDATE)
    df.CHARTTIME = pd.to_datetime(df.CHARTTIME)
    df.STORETIME = pd.to_datetime(df.STORETIME)

    ###############################Notes with valid charttime##############################
    df_valid_charttime = df[df.SUBJECT_ID.notnull()]
    df_valid_charttime = df_valid_charttime[df_valid_charttime.HADM_ID.notnull()]
    df_valid_charttime = df_valid_charttime[df_valid_charttime.CHARTTIME.notnull()]
    df_valid_charttime = df_valid_charttime[df_valid_charttime.TEXT.notnull()]
    df_valid_charttime = df_valid_charttime[['SUBJECT_ID', 'HADM_ID', 'CATEGORY', 'CHARTTIME', 'TEXT']]
    #######################################################################################

    ###############################Notes where charttime none##############################
    df_invalid_charttime = df[df.SUBJECT_ID.notnull()]
    df_invalid_charttime = df_invalid_charttime[df_invalid_charttime.HADM_ID.notnull()]
    df_invalid_charttime = df_invalid_charttime[df_invalid_charttime.CHARTTIME.isna()]
    df_invalid_charttime = df_invalid_charttime[df_invalid_charttime.TEXT.notnull()]
    df_invalid_charttime = df_invalid_charttime[['SUBJECT_ID', 'HADM_ID', 'CATEGORY', 'CHARTDATE', 'TEXT']]
    #######################################################################################

    del df

    #path to extracted patient folders from MIMIC
    extracted_patient_folders_path = '../../mimic3-readmission/data/root/'

    #path to output folder
    output_folder = 'text_extract/'
    all_files = os.listdir(extracted_patient_folders_path)
    all_folders = list(filter(lambda x: x.isdigit(), all_files))

    #Proceed to extract notes
    extract_notes(all_folders, extracted_patient_folders_path, df_valid_charttime, test=1)
    extract_notes(all_folders, extracted_patient_folders_path, df_invalid_charttime, test=1,\
                      patient_file_name='notes_charttime_none', all_notes_file_name='all_notes_charttime_none') 
