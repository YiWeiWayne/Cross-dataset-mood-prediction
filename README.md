# Cross-Cultural Music Emotion Recognition by Adversarial Discriminative Domain Adaptation
(To be published in **ICMLA 2018**)

## Effectiveness of adaptation:
### 1. Valence prediction: Our adaptation improves the performance for all features 
### 2. Arousal prediction: Our adaptation improves the performance for the timbre feature
#### Table 2. Comparison of the R2 performance of our model with or without adaptation evaluated in the cross-dataset experiment (tested on CH818)
|         | Adaptation | Timbre | Pitch | Rhythm | Timbre + Pitch | Timbre + Rhythm | Rhythm + Pitch | Timbre + Pitch + Rhythm |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Valence | - | 0.03 | 0.08 | 0.04 | 0.08 | 0.05 | 0.08 | 0.09 |
|         | V | **0.21** | 0.18 | 0.06 | 0.22 | 0.22 | 0.17 | **0.23** |
| Arousal | - | 0.72 | 0.69 | 0.39 | 0.74 | 0.68 | 0.67 | 0.74 | 
|         | V | **0.73** | 0.65 | 0.28 | **0.76** | 0.65 | 0.49 | 0.71 |
