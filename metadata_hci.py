import os
import pandas as pd

if(not os.path.isfile('metadata_hci.csv')):
    data_dir = 'data/hci_tagging/Sessions/'

    df = pd.DataFrame(columns = ['sess', 'arousal', 'valence'])

    cont = 0
    for sess in os.listdir(data_dir):
        with open(data_dir + sess + '/session.xml', 'r') as f:
            f.readline()
            l = f.readline()
            idx = l.index('feltArsl="')
            aro = int(l[idx+10:idx+11])
            assert 1 <= aro and aro <= 9
            idx = l.index('feltVlnc="')
            val = int(l[idx+10:idx+11])
            assert 1 <= val and val <= 9
            idx = l.index('sessionId="')
            idx_2 = l.index('" date="')
            sess_id = l[idx+11:idx_2]
            assert sess_id == sess

            df.loc[cont] = [sess_id, aro, val]
            cont +=1

    df.to_csv('metadata_hci.csv', index=False)