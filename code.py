# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Ask user for the CSV file input
file_path = input("Enter the path to the CSV file: ")

# Load the dataset
try:
    iris_df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error loading the dataset: {e}")
    exit()

# Display the first few rows of the dataset
print(iris_df.head())

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Perform basic EDA
sns.pairplot(iris_df, hue="Species", diag_kind='kde', markers=["o", "s", "D"], palette="husl")
plt.suptitle("Pair Plot of Iris Dataset", y=1.02)
plt.show()

# Split the dataset into features (X) and target variable (y)
X = iris_df.drop('Species', axis=1)
y = iris_df['Species']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
decision_tree_model = DecisionTreeClassifier(random_state=42)
random_forest_model = RandomForestClassifier(random_state=42)
logistic_regression_model = LogisticRegression(random_state=42)

# Train and evaluate Decision Tree model
decision_tree_model.fit(X_train, y_train)
y_pred_decision_tree = decision_tree_model.predict(X_test)

# Train and evaluate Random Forest model
random_forest_model.fit(X_train, y_train)
y_pred_random_forest = random_forest_model.predict(X_test)

# Train and evaluate Logistic Regression model
logistic_regression_model.fit(X_train, y_train)
y_pred_logistic_regression = logistic_regression_model.predict(X_test)

# Evaluate the models
models = {
    "Decision Tree": y_pred_decision_tree,
    "Random Forest": y_pred_random_forest,
    "Logistic Regression": y_pred_logistic_regression
}

for model_name, y_pred in models.items():
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    print(f"\nModel: {model_name} Evaluation:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    # Display classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)
