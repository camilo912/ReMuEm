import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def main():
    df = pd.read_csv('data/mosei/Raw/Labels/saved_5000_batch_raw.csv', usecols=['Input.VIDEO_ID', 'Input.CLIP', 'Answer.anger', 'Answer.disgust', 'Answer.fear', 'Answer.happiness', 'Answer.sadness', 'Answer.surprise'])
    # df = pd.read_csv('data/mosei/Raw/Labels/5000_batch_raw.csv', usecols=['Input.VIDEO_ID', 'Input.CLIP', 'Answer.anger', 'Answer.disgust', 'Answer.fear', 'Answer.happiness', 'Answer.sadness', 'Answer.surprise'])
    # df = pd.read_csv('data/mosei/Raw/Labels/Batch_2980374_batch_results.csv', usecols=['Input.VIDEO_ID', 'Input.CLIP', 'Answer.anger', 'Answer.disgust', 'Answer.fear', 'Answer.happiness', 'Answer.sadness', 'Answer.surprise'])
    df.columns = ['vid', 'clip', 'ang', 'dis', 'fea', 'hap', 'sad', 'sur']

    id_to_emotion = np.array(['ang', 'dis', 'fea', 'hap', 'sad', 'sur'])
    mapa = np.array([(-0.51, 0.59), (-0.6, 0.35), (-0.64, 0.6), (0.81, 0.51), (-0.63, -0.27), (0.4, 0.67)])

    df_n = pd.DataFrame(columns = ['path', 'vid', 'clip', 'emotion', 'arousal', 'valence'])

    print(len(df))

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
        # val, aro = norm@mapa
        
        intensity = data.sum()/9
        val, aro = (intensity*norm)@mapa

        
        if(np.max(intensity) > 0.5):
            maxis = id_to_emotion[np.argwhere(list(intensity) == np.amax(intensity))].squeeze()
            emo = maxis if len(maxis.shape) == 0 else 'mul'
        # if(np.max(norm) > 0.5):
        #     maxis = id_to_emotion[np.argwhere(list(norm) == np.amax(norm))].squeeze()
        #     emo = maxis if len(maxis.shape) == 0 else 'mul'
        else:
            emo = 'neu'

        df_n.loc[cont] = [path, vid_g, clip_g, emo, aro, val]
        cont += 1
    
    # df_n.to_csv('tmp.csv', index=False)
    print(cont)
    print(df_n[['arousal', 'valence']].describe())

    # print(len(df_n[df_n['arousal']>=0.33]))
    # print(len(df_n[df_n['arousal']<=-0.33]))
    # print(len(df_n[df_n['valence']>=0.33]))
    # print(len(df_n[df_n['valence']<=-0.33]))

    print(df_n.loc[(df_n['arousal']<=-0.33) & (df_n['valence']<=-0.33), ['arousal', 'valence']].describe())
    print(df_n.loc[(df_n['arousal']<=-0.33) & (df_n['valence'].between(-0.33, 0.33,inclusive=False)), ['arousal', 'valence']].describe())
    print(df_n.loc[(df_n['arousal']<=-0.33) & (df_n['valence']>=0.33), ['arousal', 'valence']].describe())
    print(df_n.loc[(df_n['arousal'].between(-0.33, 0.33,inclusive=False)) & (df_n['valence']<=-0.33), ['arousal', 'valence']].describe())
    print(df_n.loc[(df_n['arousal'].between(-0.33, 0.33,inclusive=False)) & (df_n['valence'].between(-0.33, 0.33,inclusive=False)), ['arousal', 'valence']].describe())
    print(df_n.loc[(df_n['arousal'].between(-0.33, 0.33,inclusive=False)) & (df_n['valence']>=0.33), ['arousal', 'valence']].describe())
    print(df_n.loc[(df_n['arousal']>=0.33) & (df_n['valence']<=-0.33), ['arousal', 'valence']].describe())
    print(df_n.loc[(df_n['arousal']>=0.33) & (df_n['valence'].between(-0.33, 0.33,inclusive=False)), ['arousal', 'valence']].describe())
    print(df_n.loc[(df_n['arousal']>=0.33) & (df_n['valence']>=0.33), ['arousal', 'valence']].describe())

    fig, ax = plt.subplots()
    plt.plot(-1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, -1, 'wo', marker=".", markersize=0)
    plt.plot(1, 0, 'wo', marker=".", markersize=0)
    plt.plot(0, 1, 'wo', marker=".", markersize=0)
    plt.plot(0, 0, 'ko', marker="+")
    plt.plot(df_n['valence'].values, df_n['arousal'].values, 'bo', label='labels', markersize=2)
    ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()