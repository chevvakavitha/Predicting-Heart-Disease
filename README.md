# Predicting Heart Disease
This project focuses on predicting heart disease using machine learning techniques. We analyze various features such as age, sex, blood pressure, cholesterol levels,
and other medical attributes to predict whether a person is likely to have heart disease. The goal is to build a predictive model that can
assist healthcare professionals in early diagnosis and intervention.
![image](https://github.com/user-attachments/assets/776621a2-dad9-48a5-8b8b-c66b62dc8d48)

## Table of Contents
- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [License](#license)

## Project Overview
Heart disease is one of the leading causes of death worldwide, and early detection is crucial for effective treatment. 
This project uses machine learning to predict the likelihood of heart disease based on several health-related features. 
By training a model on historical health data, we aim to provide a tool for predicting heart disease risk in individuals.

### The Key Steps in the Project:
1. Data preprocessing and cleaning.
2. Feature selection and engineering.
3. Model selection and training.
4. Model evaluation and performance metrics.
![image](https://github.com/user-attachments/assets/80944b02-a5ed-4c49-b0d5-8d51e32dea1d)

## Data Description
The dataset used in this project contains several attributes related to health and demographics, including:
- `Age`: Age of the patient.
- `Sex`: Gender of the patient (1 = male, 0 = female).
- `ChestPainType`: Type of chest pain experienced.
- `RestingBloodPressure`: Blood pressure at rest.
- `Cholesterol`: Serum cholesterol in mg/dl.
- `FastingBloodSugar`: Whether the fasting blood sugar is greater than 120 mg/dl (1 = true, 0 = false).
- `MaxHeartRate`: Maximum heart rate achieved during exercise.
- `ExerciseInducedAngina`: Whether the patient experienced angina (1 = yes, 0 = no).
- `Oldpeak`: Depression induced by exercise relative to rest.
- `Slope`: Slope of the peak exercise ST segment.
- `NumberOfMajorVessels`: Number of major vessels colored by fluoroscopy.
- `Thalassemia`: A blood disorder (normal, fixed defect, or reversable defect).
- `HeartDisease`: Target variable (1 = presence of heart disease, 0 = absence).

## Data Preprocessing
Data preprocessing is an essential step in preparing the dataset for modeling. The following preprocessing steps were performed:
1. **Missing Values Handling**: Imputed missing values using the median for numerical features and the mode for categorical features.
2. **Encoding Categorical Features**: Categorical variables like `ChestPainType`, `Thalassemia`, etc., were encoded using one-hot encoding.
3. **Feature Scaling**: Numerical features were standardized or normalized to ensure the model treats all features equally.
4. **Splitting the Dataset**: The dataset was split into training and testing sets to evaluate model performance.
![image](https://github.com/user-attachments/assets/86a8d3a0-dfd8-4a02-bdfe-055197729b2b)

## Modeling
Several machine learning models were used to predict heart disease, including:
1. **Logistic Regression**: A baseline model for binary classification.
2. **Decision Trees**: To understand feature importance and provide interpretable results.
3. **Random Forest**: A more complex ensemble model for better accuracy.
4. **Support Vector Machine (SVM)**: To handle higher-dimensional feature spaces.
5. **K-Nearest Neighbors (KNN)**: A simple yet effective classifier for this problem.

The models were trained on the training dataset and tuned for optimal performance using cross-validation.

## Evaluation
Model evaluation was performed using the following metrics:
- **Accuracy**: The proportion of correct predictions.
- **Precision**: The ratio of true positive predictions to the total positive predictions.
- **Recall**: The ratio of true positive predictions to the actual positives.
- **F1-Score**: The harmonic mean of precision and recall.
- **ROC-AUC**: The area under the Receiver Operating Characteristic curve, measuring the model’s ability to distinguish between classes.
![image](https://github.com/user-attachments/assets/0d42a57f-356d-4b27-aa61-adb79cb890c4)

Each model’s performance was compared, and the best-performing model was selected for final predictions.

## Usage
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/predicting-heart-disease.git
    cd predicting-heart-disease
    ```
2. Load your dataset into the project.
3. Run the heart disease prediction script:
    ```bash
    python heart_disease_prediction.py
    ```
4. The model's predictions will be displayed, along with performance metrics such as accuracy and AUC.

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/predicting-heart-disease.git
    ```
2. Navigate to the project directory:
    ```bash
    cd predicting-heart-disease
    ```
3. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
![image](https://github.com/user-attachments/assets/d03b0ab4-05d7-40e9-9785-3d7281f221f1)

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- imbalanced-learn

## Conclusion
In this project, we successfully built a predictive model to detect the presence of heart disease using machine learning techniques. By leveraging a variety of features such as age, sex, cholesterol levels, and heart rate, we were able to train several models to predict heart disease risk with high accuracy. The key steps included:

- **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numerical features ensured that the data was well-prepared for modeling.
- **Model Selection**: Various models, including Logistic Regression, Decision Trees, Random Forest, Support Vector Machines, and K-Nearest Neighbors, were trained and evaluated.
- **Model Evaluation**: Performance was assessed using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC, which provided insights into the models' ability to predict heart disease accurately.

### Key Insights:
- **Feature Importance**: Features like cholesterol levels, age, and maximum heart rate were found to be the most influential in predicting heart disease.
- **Model Performance**: The Random Forest model provided the best results in terms of accuracy and AUC, demonstrating its ability to handle complex, non-linear relationships in the data.

### Impact:
This model can serve as a valuable tool for healthcare professionals to identify individuals at high risk for heart disease. Early detection and intervention can help prevent severe health issues and improve patient outcomes.

### Future Work:
- **Advanced Feature Engineering**: Exploring additional health-related features, such as lifestyle factors or family history, could further improve prediction accuracy.
- **Model Optimization**: Hyperparameter tuning and model ensemble techniques could be explored to boost performance.
- **Real-time Prediction**: Deploying the model into a real-time healthcare application could enable immediate risk assessments for new patients.

Overall, this project highlights the potential of machine learning in healthcare, demonstrating how predictive models can support early diagnosis and enhance preventive healthcare strategies.


