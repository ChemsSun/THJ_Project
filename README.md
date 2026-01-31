This repository contains the **Implementation Of Skin Segmentation Using Game Theory (Simplified Approach)**. The project is organized as follows:
### Data2_FSD/ and Data3_SFA/ and Data4_HGR/ : 
Holds the pre processed datasets used in the project, including: Original images, Patch images, Ground-truth masks , CSV files
### Processing/:
Contain the code for the prepocessing of the data we did
### `Heuristic_RGB_Detector.py`
Implements the heuristic RGB-based skin detector used as the first classifier. 
This method relies on predefined RGB rules and serves as a baseline classifier in the system.
### HSV_Classifier/ :
Includes the full implementation of the **HSV-based skin classifier** for the 3 datasets.
This module performs patch-level skin detection using color thresholding in the HSV space and computes similarity metrics (α and β) for conflict resolution.
### ANN Classifier/ :
Contains the implementation of the Artificial Neural Network based classifier.
This folder includes 3 separate notebooks, each trained and evaluated on one of the 3 datasets used in the project.
### CSV_Results/ :
Contains all **CSV output files** of the classifiers, including:
- Patch-level labels
- Similarity metrics (α, β)
- Intermediate results used for fusion and game-theoretic resolution
These files are used as inputs for the conflict detection and resolution stages.
### Conflict_matrix/ :
Stores the conflict matrices computed for the 3 datasets.
A conflict matrix identifies patches where the classifiers disagree and prepares the data required for the game-theoretic resolution step.
### Game_Resolution/ :
Contains the final decision stage of the system:
- `ZeroSumGameSolver.ipynb`: implementation of the zero-sum game used to resolve conflicts between classifiers
- Final skin segmentation results and output images after conflict resolution
______________________________________________________

## Method Overview

- Image preprocessing and patch extraction
- Independent skin classification using Heuristic RGB Detector, HSV, and ANN
- Conflict detection between classifiers
- Zero-sum game-based resolution
- Final skin segmentation output
______________________________________________________
## Anex
Drive link for more results we have : https://drive.google.com/drive/folders/1SxsAOeEIIFKR48T9ArNSQx-0ExE9lp2G?usp=sharing
