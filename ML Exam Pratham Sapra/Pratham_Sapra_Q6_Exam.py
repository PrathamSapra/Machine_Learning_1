import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2


def data_loader(file_path):
    """Loads the dataset from the given CSV file path."""
    return pd.read_csv(file_path)


def data_cleaning(df):
    """Handles missing values, removes unnecessary columns, and encodes categorical features."""
    df.dropna(inplace=True)  # Drop rows with missing values

    if 'user_id' in df.columns:
        df.drop(columns=['user_id'], inplace=True)  # Drop unique identifier

    # Convert categorical columns to numerical using one-hot encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


def feature_selection(df):
    """Selects the best features using SelectKBest method."""
    X = df.drop(columns=['great_customer_class'])
    y = df['great_customer_class']

    selector = SelectKBest(chi2, k='all')  # Select best features
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return pd.DataFrame(X_new, columns=selected_features), y


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Trains different classifiers and prints their accuracies."""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='linear', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    accuracies = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[name] = accuracy
        print(f'{name} Accuracy: {accuracy:.4f}')

    return models, accuracies


def ensemble_learning(X_train, X_test, y_train, y_test, models):
    """Uses Voting Classifier for ensemble learning."""
    ensemble = VotingClassifier(estimators=[
        ('rf', models['Random Forest']),
        ('svm', models['SVM']),
        ('lr', models['Logistic Regression']),
        ('nb', models['Naive Bayes']),
        ('knn', models['KNN'])
    ], voting='hard')

    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Ensemble Model (Voting) Accuracy: {accuracy:.4f}')

    return accuracy


def main():
    """Main function to execute the pipeline."""
    file_path = 'https://raw.githubusercontent.com/subashgandyer/datasets/main/great_customers.csv'
    df = data_loader(file_path)
    df = data_cleaning(df)
    X, y = feature_selection(df)

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train models
    models, accuracies = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Ensemble Learning
    ensemble_acc = ensemble_learning(X_train, X_test, y_train, y_test, models)

    # Compare accuracies
    best_model = max(accuracies, key=accuracies.get)
    print(f'Best individual model: {best_model} with accuracy {accuracies[best_model]:.4f}')
    print(f'Ensemble Model Accuracy: {ensemble_acc:.4f}')


if __name__ == '__main__':
    main()
