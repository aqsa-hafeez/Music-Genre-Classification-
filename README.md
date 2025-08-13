# Music Genre Classification

## Overview

This project implements a music genre classification system using the GTZAN dataset. The goal is to classify audio tracks into one of 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock) based on extracted audio features such as MFCCs, spectral centroid, zero-crossing rate, and more.

The approach used here is **tabular data-based** with a Random Forest Classifier from scikit-learn. Audio features are extracted using Librosa, preprocessed, and fed into the model for multi-class classification.

### Bonus Features Explored
- **Tabular vs. Image-based Approaches**: This implementation focuses on tabular features. For an image-based approach (e.g., using spectrograms and CNNs), refer to potential extensions in the notebook or future branches.
- **Transfer Learning**: Not implemented in the current version but can be added for spectrogram images using pre-trained models like ResNet or VGG in Keras.

## Dataset
- **Source**: GTZAN Genre Collection (available on [Kaggle](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)).
- **Structure**: 1000 audio files (30 seconds each) across 10 genres, with 100 files per genre.
- **Genres**: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.

The dataset is loaded by walking through the directory structure, creating a DataFrame with file paths and labels.

## Features Extracted
Using Librosa, the following audio features are extracted for each track:
- Mean and variance of MFCCs (Mel-Frequency Cepstral Coefficients)
- Spectral Centroid (mean and variance)
- Spectral Rolloff (mean and variance)
- Zero Crossing Rate (mean and variance)
- Tempo (BPM)

These features are aggregated into a tabular format for model training.

## Model
- **Algorithm**: Random Forest Classifier (from scikit-learn)
- **Preprocessing**:
  - Label Encoding for genres.
  - Standard Scaling for features.
  - Train-Test Split (80-20).
- **Evaluation Metrics**:
  - Accuracy: ~48% (on test set).
  - Confusion Matrix (visualized with Seaborn heatmap).
  - Classification Report (precision, recall, F1-score per genre).

## Requirements
- Python 3.10+
- Libraries:
  ```
  pandas
  numpy
  matplotlib
  seaborn
  plotly
  librosa
  ipython
  scikit-learn
  ```

Install dependencies:
```
pip install -r requirements.txt
```

## Usage
1. **Download Dataset**: Place the GTZAN dataset in a directory (e.g., `Data/genres_original/`).
2. **Run the Notebook**:
   - Open `code.ipynb` in Jupyter Notebook or Jupyter Lab.
   - Update the `data_dir` path in the notebook to point to your dataset directory.
   - Execute cells sequentially to:
     - Load data.
     - Extract features.
     - Train and evaluate the model.
3. **Output**:
   - Feature DataFrame saved as `features.csv` (optional).
   - Model performance metrics printed.
   - Confusion matrix plotted.

Example command to run Jupyter:
```
jupyter notebook code.ipynb
```

## Results
- **Accuracy**: 0.48 (48%) on the test set.
- **Classification Report** (sample from notebook):

| Genre Label | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| 0 (blues)   | 0.50     | 0.55  | 0.52    | 20     |
| 1 (classical)| 0.74    | 0.85  | 0.79    | 20     |
| ...         | ...      | ...   | ...     | ...    |
| Average     | 0.47     | 0.48  | 0.47    | 200    |

- **Confusion Matrix**: Visualized in the notebook.

Genres like classical and metal perform better due to distinct features, while others (e.g., disco, hiphop) show lower scores, indicating room for improvement with more advanced models or features.

## Potential Improvements
- **Image-based Approach**: Convert audio to spectrograms and use a CNN (e.g., via Keras).
- **Transfer Learning**: Fine-tune pre-trained models on spectrograms.
- **Hyperparameter Tuning**: Use GridSearchCV for RandomForest.
- **Advanced Features**: Include Chroma features or more MFCC coefficients.
- **Augmentation**: Apply audio augmentations (e.g., noise, pitch shift) for better generalization.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
