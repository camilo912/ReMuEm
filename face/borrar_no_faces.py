import os

data_folder = 'data/images/'
for file in [x for x in os.listdir(data_folder) if 'hci' in x and not 'face' in x]:
    cad = 'rm ' + data_folder + str(file)
    os.system(cad)