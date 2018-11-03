#dataset_name =  ['AMG_1608', 'CH_818']
#feature_name =  ['melSpec_lw', 'pitch+lw', 'auto']

audio_path = '/mnt/data/Wayne/Dataset'
audio_npy_path = '/mnt/data/Wayne/#dataset_name/Train_X@220505Hz.npy'
feature_path = 'mnt/data/Wayne/#dataset_name/Train_X@220505Hz_#feature_name.npy'

0. Reformat the audio files

# Tansfer audio files into 22.05KHz and clip into 29 seconds
functions/Transfer_funcs.audio_to_wav()

# Generate combined audio files and labels for training
functions/Transfer_funcs.wav_to_npy()

1. Extract features

# Extract autocorrelation-based tempogram
feature_extraction/MATLAB-Tempogram-Toolbox_1.0/test_TempogramToolbox.m

# Extract pitch salience representation
feature_extraction/predict_on_audio.py
ex: python predict_on_audio.py '/mnt/data/Wayne/Dataset/AMG_1608_wav@22050Hz' 'pitch' '/mnt/data/Wayne/AMG_1608_pitch+lw@22050Hz'

# Extract log-mel spectrogram
functions/model_structure.extract_melspec()

# Transfer different features into the same npy format
feature_extraction/Extract_features.py

2. Pre-training

# Within-dataset experiment
Training/AMG1608_CV.py

# Cross-dataset experiment
Training/AMG1608_train.py

3. Adversarial discriminative domain adaptation

# Cross-dataset experiment
Training/WADDA_S_AMG1608_T_CH818.py

4. Testing

# Unconfirmed
Testing/Multi_fusion_pred_S_AMG1608_T_CH818_find_by_loss.py
