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


def load_dataset(file_location):
    """Reads dataset from a CSV file."""
    return pd.read_csv(file_location)


def preprocess_data(dataframe):
    """Cleans data by handling missing values, dropping unnecessary columns, and encoding categorical variables."""
    dataframe.dropna(inplace=True)

    if 'customer_id' in dataframe.columns:
        dataframe.drop(columns=['customer_id'], inplace=True)

    categorical_features = dataframe.select_dtypes(include=['object']).columns
    dataframe = pd.get_dummies(dataframe, columns=categorical_features, drop_first=True)

    return dataframe


def select_important_features(dataframe):
    """Uses SelectKBest to choose significant features."""
    X = dataframe.drop(columns=['high_value_customer'])
    y = dataframe['high_value_customer']

    selector = SelectKBest(chi2, k='all')
    X_selected = selector.fit_transform(X, y)
    important_features = X.columns[selector.get_support()]

    return pd.DataFrame(X_selected, columns=important_features), y


def model_training(X_train, X_test, y_train, y_test):
    """Trains multiple classifiers and evaluates their accuracy."""
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='linear', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    model_accuracies = {}

    for model_name, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        model_accuracies[model_name] = accuracy
        print(f'{model_name} Accuracy: {accuracy:.4f}')

    return classifiers, model_accuracies


def voting_ensemble(X_train, X_test, y_train, y_test, classifiers):
    """Creates a Voting Classifier for ensemble learning."""
    ensemble_model = VotingClassifier(estimators=[
        ('rf', classifiers['Random Forest']),
        ('svm', classifiers['SVM']),
        ('lr', classifiers['Logistic Regression']),
        ('nb', classifiers['Naive Bayes']),
        ('knn', classifiers['KNN'])
    ], voting='hard')

    ensemble_model.fit(X_train, y_train)
    ensemble_predictions = ensemble_model.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
    print(f'Ensemble Model Accuracy: {ensemble_accuracy:.4f}')

    return ensemble_accuracy


def execute_pipeline():
    """Runs the full data processing, model training, and evaluation pipeline."""
    file_location = 'https://raw.githubusercontent.com/subashgandyer/datasets/main/great_customers.csv'
    dataset = load_dataset(file_location)
    dataset = preprocess_data(dataset)
    X, y = select_important_features(dataset)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifiers, model_accuracies = model_training(X_train, X_test, y_train, y_test)

    ensemble_accuracy = voting_ensemble(X_train, X_test, y_train, y_test, classifiers)

    best_model_name = max(model_accuracies, key=model_accuracies.get)
    print(f'Best Performing Model: {best_model_name} with Accuracy: {model_accuracies[best_model_name]:.4f}')
    print(f'Voting Ensemble Model Accuracy: {ensemble_accuracy:.4f}')


if __name__ == '__main__':
    execute_pipeline()