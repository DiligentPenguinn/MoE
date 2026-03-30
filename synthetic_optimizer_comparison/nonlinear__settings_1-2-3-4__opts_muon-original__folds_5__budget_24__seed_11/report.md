# Synthetic Optimizer Comparison Report

- Recommendation: The original optimizer family is the stronger default for this synthetic setup.
- Win counts: {'muon': 0, 'original': 4, 'tie': 0}
- Average Muon minus original accuracy: -4.815000000000005

## Setting s1

| Optimizer | Expert Num | CV Acc | CV Loss | Final Test Acc | Final Entropy | Mean Train Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| muon | 8 | 93.8812 | 0.3744 | 91.6412 | 0.6685 | 61.0212 |
| original | 8 | 99.9938 | 0.3133 | 99.9825 | 0.0129 | 39.5439 |

- `muon` specialization: dominant expert clusters=[1, 2, 1, 2, 1, 0, 1, 3], center margins=[3.0127, 2.3359, 0.8778, 0.1395, 0.7931, 0.0024, 0.0762, 2.8724]
- `muon` router peaks by cluster: [3, 0, 5, 7]
- `original` specialization: dominant expert clusters=[1, 3, 1, 3, 0, 3, 1, 2], center margins=[0.0037, 0.0018, 0.0001, 0.0002, 0.0002, 0.0041, 0.003, 0.0022]
- `original` router peaks by cluster: [0, 7, 4, 6]

## Setting s2

| Optimizer | Expert Num | CV Acc | CV Loss | Final Test Acc | Final Entropy | Mean Train Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| muon | 8 | 92.0187 | 0.3930 | 89.9525 | 0.8490 | 33.7830 |
| original | 8 | 99.1625 | 0.3216 | 98.3275 | 0.1286 | 59.9170 |

- `muon` specialization: dominant expert clusters=[3, 3, 0, 0, 2, 2, 3, 0], center margins=[0.2635, 0.3988, 0.0248, 0.3279, 0.7333, 0.1816, 0.6006, 0.4184]
- `muon` router peaks by cluster: [1, 0, 3, 2]
- `original` specialization: dominant expert clusters=[2, 3, 1, 2, 2, 3, 3, 1], center margins=[0.5552, 0.3227, 0.471, 1.6795, 0.4641, 0.1098, 0.701, 0.7823]
- `original` router peaks by cluster: [4, 7, 5, 1]

## Setting s3

| Optimizer | Expert Num | CV Acc | CV Loss | Final Test Acc | Final Entropy | Mean Train Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| muon | 8 | 98.6188 | 0.3271 | 98.7600 | 0.1891 | 62.9197 |
| original | 8 | 100.0000 | 0.3133 | 100.0000 | 0.0000 | 61.5417 |

- `muon` specialization: dominant expert clusters=[0, 1, 0, 3, 0, 2, 3, 0], center margins=[2.2372, 2.2917, 0.263, 2.0884, 0.0103, 2.3535, 0.6557, 0.7678]
- `muon` router peaks by cluster: [0, 1, 5, 3]
- `original` specialization: dominant expert clusters=[1, 0, 3, 1, 2, 0, 1, 1], center margins=[0.0161, 0.0039, 0.0118, 0.0008, 0.0003, 0.0015, 0.0098, 0.0098]
- `original` router peaks by cluster: [7, 5, 2, 3]

## Setting s4

| Optimizer | Expert Num | CV Acc | CV Loss | Final Test Acc | Final Entropy | Mean Train Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| muon | 8 | 96.8438 | 0.3448 | 97.3650 | 0.3069 | 62.6697 |
| original | 8 | 99.2687 | 0.3206 | 98.6688 | 0.1117 | 61.6914 |

- `muon` specialization: dominant expert clusters=[2, 2, 1, 1, 2, 3, 2, 3], center margins=[4.8407, 0.1303, 2.576, 1.4765, 2.2843, 0.3444, 0.9464, 1.2043]
- `muon` router peaks by cluster: [7, 2, 4, 3]
- `original` specialization: dominant expert clusters=[2, 2, 3, 0, 2, 1, 3, 2], center margins=[0.3125, 0.1758, 0.2406, 0.0453, 0.1421, 0.3181, 0.0234, 0.0749]
- `original` router peaks by cluster: [5, 1, 4, 2]
