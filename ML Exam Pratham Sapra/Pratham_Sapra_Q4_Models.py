import kagglehub
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Download dataset
path = kagglehub.dataset_download("anandshaw2001/netflix-movies-and-tv-shows")
df = pd.read_csv(path + "/netflix_titles.csv")

# Preprocessing
label_encoder = LabelEncoder()
df.dropna(inplace=True)  # Drop missing values
df['type'] = label_encoder.fit_transform(df['type'])
df['rating'] = label_encoder.fit_transform(df['rating'])
X = df[['type', 'release_year', 'rating']]
y = label_encoder.fit_transform(df['listed_in'].astype(str))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Store execution times
times = {}

# Train and evaluate models
for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    model.predict(X_test)
    end_time = time.time()
    times[name] = end_time - start_time
    print(f"{name} took {times[name]:.4f} seconds")

# Order by execution time
ordered_times = sorted(times.items(), key=lambda x: x[1])
print("\nExecution times in ascending order:")
for algo, t in ordered_times:
    print(f"{algo}: {t:.4f} seconds")
