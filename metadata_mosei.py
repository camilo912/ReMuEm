import pandas as pd
import numpy as np
import os

def main():
    if(not os.path.isfile('metadata_mosei.csv')):
        df = pd.read_csv('data/mosei/Raw/Labels/5000_batch_raw.csv', usecols=['Input.VIDEO_ID', 'Input.CLIP', 'Answer.anger', 'Answer.disgust', 'Answer.fear', 'Answer.happiness', 'Answer.sadness', 'Answer.surprise'])
        df.columns = ['vid', 'clip', 'ang', 'dis', 'fea', 'hap', 'sad', 'sur']

        id_to_emotion = np.array(['ang', 'dis', 'fea', 'hap', 'sad', 'sur'])
        mapa = np.array([(-0.51, 0.59), (-0.6, 0.35), (-0.64, 0.6), (0.81, 0.51), (-0.63, -0.27), (0.4, 0.67)])

        df_n = pd.DataFrame(columns = ['path', 'vid', 'clip', 'emotion', 'arousal', 'valence'])

        # every utterance is annotated thrice
        cont = 0
        for i in range(0, len(df), 3):
            df_i = df.loc[i:i+2, :]

            path = 'data/mosei/Raw/[modal]/Segmented/Combined/' + str(df_i.loc[i, 'vid']) + '_' + str(df_i.loc[i, 'clip']) + '[ext]'

            vid_g, clip_g = df_i.loc[i, ['vid', 'clip']]

            for vid, clip, ang, dis, fea, hap, sad, sur in df_i.values:
                if(vid != vid_g or clip != clip_g):
                    raise Exception('different videos:' + str(vid) + '_' + str(clip) + ' / ' + str(vid_g) + str(clip))
            
            # arousal valence
            data = df_i.loc[:, ['ang', 'dis', 'fea', 'hap', 'sad', 'sur']]
            norm = data.sum().pow(2) / max(data.sum().pow(2).sum(), 1e-15)
            aro, val = norm@mapa
            
            if(np.max(norm) > 0.5):
                maxis = id_to_emotion[np.argwhere(list(norm) == np.amax(norm))].squeeze()
                emo = maxis if len(maxis.shape) == 0 else 'mul'
            else:
                emo = 'neu'

            df_n.loc[cont] = [path, vid_g, clip_g, emo, aro, val]
            cont += 1
        
        
        df_n.to_csv('metadata_mosei.csv', index=False)


if __name__ == '__main__':
    main()