import os
import pandas as pd

data_folder = 'data/enterface/enterface database/'
betados = ['Ses04M_impro04.txt', 'Ses04M_impro05.txt']
limit = 5 # minimum duration of utterance in seconds

if(os.path.isfile('metadata.csv')):
    df_m = pd.read_csv('metadata.csv')
else:
    df_m = pd.DataFrame(columns = ['subject', 'emotion', 'sentence', 'path'])
    # read each subject
    for subject in [x for x in os.listdir(data_folder) if x.startswith('subject')]:
        subject_folder = data_folder + subject + '/'
        # read each emotion
        for emotion in os.listdir(subject_folder):
            emotion_folder = subject_folder + emotion + '/'
            
            # read each sentence
            for sentence in [x for x in os.listdir(emotion_folder) if x.startswith('sentence')]:
                sentence_folder = emotion_folder + sentence + '/'
                
                # read video
                for video in [x for x in os.listdir(sentence_folder) if x.endswith('.avi')]:
                    path = sentence_folder + video
                    df_m = df_m.append({'subject':subject, 'emotion':emotion, 'sentence':sentence, 'path':path}, ignore_index=True)

    df_m.to_csv('metadata_enterface.csv', index=False)