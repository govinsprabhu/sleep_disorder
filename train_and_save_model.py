import pandas as pd
import numpy as np
import joblib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

def train_and_export():
    print("Loading data...")
    df = pd.read_csv('sleep_health.csv')
    
    # Split Blood Pressure
    df1 = pd.concat([df, df['Blood Pressure'].str.split('/', expand=True)], axis=1).drop('Blood Pressure', axis=1)
    df1 = df1.rename(columns={0: 'BloodPressure_Upper_Value', 1: 'BloodPressure_Lower_Value'})
    df1['BloodPressure_Upper_Value'] = df1['BloodPressure_Upper_Value'].astype(float)
    df1['BloodPressure_Lower_Value'] = df1['BloodPressure_Lower_Value'].astype(float)
    
    # Fill nan in Sleep Disorder with 'None' if needed, or check how they are handled. The original code says:
    # Unique values of Sleep Disorder: [nan 'Sleep Apnea' 'Insomnia']
    # Label Encoder will fail on NaN if not cast. The original code had `df1['Sleep Disorder'] = label_encoder.fit_transform(df1['Sleep Disorder'])`
    # Let's check original data. Let's fill NA with 'None' first!
    df1['Sleep Disorder'] = df1['Sleep Disorder'].fillna('None')

    # Apply Label Encoder
    label_encoders = {}
    
    for col in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
        le = preprocessing.LabelEncoder()
        df1[col] = le.fit_transform(df1[col])
        label_encoders[col] = le
        print(f"Classes for {col}: {le.classes_}")
        
    # Outlier Removal (as per original code)
    num_col = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level',
               'Heart Rate', 'Daily Steps', 'BloodPressure_Upper_Value', 'BloodPressure_Lower_Value']
    
    Q1 = df1[num_col].quantile(0.25)
    Q3 = df1[num_col].quantile(0.75)
    IQR = Q3 - Q1
    
    df1 = df1[~((df1[num_col] < (Q1 - 1.5 * IQR)) | (df1[num_col] > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    # Prepare X and y
    X = df1.drop(['Person ID', 'Sleep Disorder'], axis=1)
    y = df1['Sleep Disorder']
    
    # The original code added "Age_bin" but then dropped it. We didn't add it.
    
    print(f"Features: {X.columns.tolist()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    
    # Train
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier())
    ])
    
    param_grid = [
        {
            'clf': [KNeighborsClassifier()],
            'clf__n_neighbors': [3, 5, 7, 9],
        }
    ]
    
    print("Training model...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    acc = best_model.score(X_test, y_test)
    print(f"Accuracy on test set: {acc:.4f}")
    
    # Save Model and Encoders
    joblib.dump(best_model, 'model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    print("Model and Label Encoders saved.")

if __name__ == '__main__':
    train_and_export()
