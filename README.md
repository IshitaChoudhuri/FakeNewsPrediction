# Fake News Prediction
## README

### News Classification using Logistic Regression and TF-IDF Vectorization

This repository contains a Python script for classifying news articles as real or fake using a Logistic Regression model with TF-IDF vectorization. The code performs text preprocessing, feature extraction, model training, and evaluation.

### Setup Instructions
1. Install the required libraries by running:
   ```
   pip install numpy pandas nltk scikit-learn
   ```

2. Download NLTK stopwords by executing:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

3. Ensure the dataset `train.csv` is available in the same directory as the script.

### Code Overview
- The script loads the dataset, preprocesses text data by stemming and removing stopwords.
- It converts text data into numerical features using TF-IDF vectorization.
- Splits the data into training and testing sets.
- Trains a Logistic Regression model on the training data.
- Evaluates the model's accuracy on both training and testing sets.

### Usage
1. Run the script `prediction.ipynb`.
2. Follow the prompts to input an index for automated prediction.
3. View the predicted label (Real or Fake) and compare it with the actual label.

### Additional Functionalities
To enhance the code further, consider implementing:
- Cross-validation for robust model evaluation.
- Hyperparameter tuning for optimizing model performance.
- Confusion matrix and classification report generation.
- Feature importance analysis to identify key words influencing classification.
- Saving and loading trained models for future use.

Feel free to explore and expand upon this codebase to improve news classification accuracy and functionality.
