import os
import pandas as pd
from matplotlib import pyplot as plt

data_folder = 'data/IEMOCAP_full_release/'
betados = [] # ['Ses04M_impro04.txt', 'Ses04M_impro05.txt']
limit = 0 # minimum duration of utterance in seconds

df_m = pd.DataFrame(columns = ['session', 'filename', 'uterance_name', 't_inf', 't_upp', 'arousal', 'valence'])
# read each session
for sess in [x for x in os.listdir(data_folder) if x.startswith('Session')]:
    sess_folder = data_folder + sess + '/dialog/EmoEvaluation/'
    # read each scene (improvised or script)
    for scene in [x for x in os.listdir(sess_folder) if x.endswith('.txt') and not x in betados]:
        scene_filename = sess_folder + scene
        try:
            lines = open(scene_filename, 'r', encoding='latin1').readlines()[2:]
        except Exception as e:
            print(scene_filename)
            raise e
        # read metadata for each scene
        for i in range(len(lines)):
            if(']' in lines[i]):
                l = lines[i]
                pre = l[:l.index(']')+1]
                l = l[len(pre):]

                post = l[l.index('['):]
                l = l[:-len(post)]

                # extract utterance time
                t_inf, t_upp = pre.replace('[', '').replace(']', '').split(' - ')
                # extract valence and arousal values
                valence, arousal, _ = post.replace('[', '').replace(']', '').split(', ')
                tmp, emo = l.split('\t')[1:3]
                sep = tmp.rindex('_')
                # extract filename and utterance name
                filename = tmp[:sep]
                uterance_name = tmp[sep+1:]

                if(float(t_upp) - float(t_inf) > limit):
                    df_m = df_m.append({'session':sess, 'filename':filename, 'uterance_name':uterance_name, 't_inf':t_inf, 't_upp':t_upp, 'arousal':arousal, 'valence':valence, 'emotion':emo}, ignore_index=True)

df_m = df_m.astype({'arousal': float, 'valence':float})
print(df_m[['arousal', 'valence']].describe())
# print(len(df_m[df_m['arousal']>=4]))
# print(len(df_m[df_m['arousal']<=2]))
# print(len(df_m[df_m['valence']>=4]))
# print(len(df_m[df_m['valence']<=2]))

print(df_m.loc[(df_m['arousal']<=2) & (df_m['valence']<=2), ['arousal', 'valence']].describe())
print(df_m.loc[(df_m['arousal']<=2) & (df_m['valence'].between(2,4,inclusive=False)), ['arousal', 'valence']].describe())
print(df_m.loc[(df_m['arousal']<=2) & (df_m['valence']>=4), ['arousal', 'valence']].describe())
print(df_m.loc[(df_m['arousal'].between(2,4,inclusive=False)) & (df_m['valence']<=2), ['arousal', 'valence']].describe())
print(df_m.loc[(df_m['arousal'].between(2,4,inclusive=False)) & (df_m['valence'].between(2,4,inclusive=False)), ['arousal', 'valence']].describe())
print(df_m.loc[(df_m['arousal'].between(2,4,inclusive=False)) & (df_m['valence']>=4), ['arousal', 'valence']].describe())
print(df_m.loc[(df_m['arousal']>=4) & (df_m['valence']<=2), ['arousal', 'valence']].describe())
print(df_m.loc[(df_m['arousal']>=4) & (df_m['valence'].between(2,4,inclusive=False)), ['arousal', 'valence']].describe())
print(df_m.loc[(df_m['arousal']>=4) & (df_m['valence']>=4), ['arousal', 'valence']].describe())

fig, ax = plt.subplots()
plt.plot(-1, 0, 'wo', marker=".", markersize=0)
plt.plot(0, -1, 'wo', marker=".", markersize=0)
plt.plot(1, 0, 'wo', marker=".", markersize=0)
plt.plot(0, 1, 'wo', marker=".", markersize=0)
plt.plot(0, 0, 'ko', marker="+")
plt.plot((df_m['valence'].values-1)/2-1, (df_m['arousal'].values-1)/2-1, 'bo', label='labels', markersize=2)
ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False))
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.show()