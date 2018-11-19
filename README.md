# Cross-dataset-mood-prediction

The Kears implementation of "Cross-Cultural Music Emotion Recognition by Adversarial Discriminative Domain Adaptation".

'''Yi-Wei Chen, Yi-Hsuan Yang, and Homer H. Chen, "Cross-cultural music emotion recognition by adversarial discriminative domain adaptation," Proc. IEEE Int. Conf. Machine Learning and Applications (ICMLA), December 2018.

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

## References
@inproceedings{choi2017kapre,
  title={Kapre: On-GPU Audio Preprocessing Layers for a Quick Implementation of Deep Neural Network Models with Keras},
  author={Choi, Keunwoo and Joo, Deokjin and Kim, Juho},
  booktitle={Machine Learning for Music Discovery Workshop at 34th International Conference on Machine Learning},
  year={2017},
  organization={ICML}
}
@inproceedings{Bittner:DeepSalience:ISMIR:17, Address = {Suzhou, China}, Author = {Bittner, R.M. and McFee, B. and Salamon, J. and Li, P. and Bello, J.P.}, Booktitle = {18th Int.~Soc.~for Music Info.~Retrieval Conf.}, Month = {Oct.}, Title = {Deep Salience Representations for $F_0$ Estimation in Polyphonic Music}, Year = {2017}}
