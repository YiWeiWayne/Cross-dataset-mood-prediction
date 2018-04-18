import os
from pyexcel_xls import get_data
import json
import librosa

dataset_name = 'AMG_1608'
sec_length = 29
sample_rate = 22050
output_sample_rate = 22050
save_path = '/mnt/data/Wayne'
if dataset_name == 'AMG_1608':
    dataset_path = save_path + '/Dataset/AMG1838_original/amg1838_mp3_original'
    label_path = save_path + '/Dataset/AMG1838_original/AMG1608/amg1608_v2.xls'
elif dataset_name == 'CH_818':
    dataset_path = save_path + '/Dataset/CH818/mp3'
    label_path = save_path + '/Dataset/CH818/label/CH818_Annotations.xlsx'
wav_path = save_path + '/Dataset/' + dataset_name + '_wav_' + str(output_sample_rate) + 'Hz'
# load xls data
data = get_data(label_path)
encodedjson = json.dumps(data)
decodejson = json.loads(encodedjson)
if dataset_name == 'AMG_1608':
    decodejson = decodejson['amg1608_v2']
elif dataset_name == 'CH_818':
    decodejson = decodejson['Arousal']

# transfer mp3 to wav file
if not os.path.exists(wav_path):
    os.makedirs(wav_path)
if True:
    for i in range(1, len(decodejson)):
        print(str(i).zfill(4))
        if dataset_name == 'AMG_1608':
            if os.path.exists(dataset_path + '/' + str(decodejson[i][2]) + '.mp3'):
                print(dataset_path + '/' + str(decodejson[i][2]) + '.mp3')
                y, sr = librosa.load(dataset_path + '/' + str(decodejson[i][2]) + '.mp3', sr=output_sample_rate)
                print(y.shape)
                print(str(sr))
        elif dataset_name == 'CH_818':
            for root, subdirs, files in os.walk(dataset_path):
                for f in files:
                    if os.path.splitext(f)[1] == '.MP3' or os.path.splitext(f)[1] == '.mp3':
                        if f[0:4].startswith(str(i) + '='):
                            print(dataset_path + '/' + f)
                            y, sr = librosa.load(dataset_path + '/' + f, sr=output_sample_rate)
                            print(y.shape)
                            print(str(sr))
        if y.shape[0] >= output_sample_rate*sec_length:
            librosa.output.write_wav(path=wav_path + '/' + str(i).zfill(4) + '@' + str(output_sample_rate) + '.wav',
                                     y=y[0:int(output_sample_rate*sec_length)], sr=output_sample_rate)
        else:
            print('Shorter: ' + str(y.shape[0]) + '/' + str(sample_rate*sec_length))
