# Cross-dataset-mood-prediction

The Kears implementation of "Cross-Cultural Music Emotion Recognition by Adversarial Discriminative Domain Adaptation".

``` Citation: Yi-Wei Chen, Yi-Hsuan Yang, and Homer H. Chen, "Cross-cultural music emotion recognition by adversarial discriminative domain adaptation," Proc. IEEE Int. Conf. Machine Learning and Applications (ICMLA), December 2018. ```

## Effectiveness of adaptation:
1. Valence prediction: Our adaptation improves the performance for all features 
2. Arousal prediction: Our adaptation improves the performance for the timbre feature
* Table 2. Comparison of the R2 performance of our model with or without adaptation evaluated in the cross-dataset experiment (tested on CH818)
|         | Adaptation | Timbre | Pitch | Rhythm | Timbre + Pitch | Timbre + Rhythm | Rhythm + Pitch | Timbre + Pitch + Rhythm |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Valence | - | 0.03 | 0.08 | 0.04 | 0.08 | 0.05 | 0.08 | 0.09 |
|         | V | **0.21** | 0.18 | 0.06 | 0.22 | 0.22 | 0.17 | **0.23** |
| Arousal | - | 0.72 | 0.69 | 0.39 | 0.74 | 0.68 | 0.67 | 0.74 | 
|         | V | **0.73** | 0.65 | 0.28 | **0.76** | 0.65 | 0.49 | 0.71 |

## How to use
### 0. Reformat the audio files
* Tansfer audio files into 22.05KHz and clip into 29 seconds
Call function: Transfer_funcs.audio_to_wav()
* Generate combined audio files and labels for training
Call function: Transfer_funcs.wav_to_npy()

### 1. Extract features
* Extract log-mel spectrogram [1]
Call function: model_structure.extract_melspec()
* Extract pitch salience representation [2]
Use predict_on_audio.py
ex: python predict_on_audio.py '/mnt/data/Wayne/Dataset/AMG_1608_wav@22050Hz' 'pitch' '/mnt/data/Wayne/AMG_1608_pitch+lw@22050Hz'
* Extract autocorrelation-based tempogram [3]
Use MATLAB-Tempogram-Toolbox_1.0/test_TempogramToolbox.m
* Transfer different features into the same npy format
Use Extract_features.py

### 2. Pre-training
* Within-dataset experiment
Use AMG1608_CV.py
* Cross-dataset experiment
Use AMG1608_train.py

### 3. Adversarial discriminative domain adaptation
* Cross-dataset experiment
Use WADDA_S_AMG1608_T_CH818.py

### 4. Testing
Use Multi_fusion_pred_S_AMG1608_T_CH818_find_by_loss.py

## References
1. K. Choi, D. Joo, and J. Kim, "Kapre: On-gpu audio preprocessing layers for a quick implementation of deep neural network models with keras," arXiv preprint arXiv:1706.05781, 2017.
2.	R. M. Bittner et al., “Deep salience representations for f0 estimation in polyphonic music,” in Proc. Int. Soc. Music Information Retrieval, pp. 23–27, 2017.
3.	P. Grosche, M. Muller, and F. Kurth, “Cyclic tempogram—a midlevel tempo representation for music signals,” in IEEE Trans. Acoustics Speech Signal Process., pp. 5522–5525, 2010.

