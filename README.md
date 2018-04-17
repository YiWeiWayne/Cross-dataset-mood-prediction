# Cross-dataset-mood-prediction

## 20180329 - Regressor result
#### Show in R square(Square of Pearson correlation coefficients)
#### Valence:
| Train | Test | SVM(paper's) | Compact_CNN |
| :---: | :---: | :---: | :---: |
| AMG | AMG | 0.14 | **0.27** |
|     | MER | **0.40** | 0.14 |
|     | CH  | **0.21** | 0.06 |

#### Arousal:
| Train | Test | SVM(paper's) | Compact_CNN |
| :---: | :---: | :---: | :---: |
| AMG | AMG | 0.73 | **0.78** |
|     | MER | 0.75 | **0.82** |
|     | CH  | 0.68 | **0.70** |

## 20180414 - Regressor result with adapted
#### Show in R square(Square of Pearson correlation coefficients)
#### Valence:
| Train | Test | SVM(paper's) | Compact_CNN | Compact_CNN(with adaptation) |
| :---: | :---: | :---: | :---: | :---: |
| AMG | CH  | **0.21** | 0.06 | 0.20 |

#### Arousal:
| Train | Test | SVM(paper's) | Compact_CNN | Compact_CNN(with adaptation) |
| :---: | :---: | :---: | :---: | :---: |
| AMG | CH  | 0.68 | 0.70 | **0.73** |

## 20180417 - Regressor result trained by full dataset
#### Show in R square(Square of Pearson correlation coefficients)
#### Valence:
| Train | Test | SVM(paper's) | Compact_CNN | Compact_CNN(with adaptation) |
| :---: | :---: | :---: | :---: | :---: |
| AMG | CH  | **0.21** | 0.05 | 0.20 |

#### Arousal:
| Train | Test | SVM(paper's) | Compact_CNN | Compact_CNN(with adaptation) |
| :---: | :---: | :---: | :---: | :---: |
| AMG | CH  | 0.68 | 0.72 | **0.73** |
