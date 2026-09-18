"""MLOps-805-B Experiment 3: Decision Tree classification on Titanic dataset."""

# 1. Import Libraries
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# 2. Load Dataset
def load_dataset(file_path: str) -> pd.DataFrame:
    """Load Titanic dataset from local CSV file."""
    return pd.read_csv(file_path)


# 3. Explore Dataset
def explore_dataset(df: pd.DataFrame) -> None:
    """Print required basic dataset exploration outputs."""
    print("=" * 60)
    print("DATASET EXPLORATION")
    print("=" * 60)

    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    duplicate_count = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicate_count}")

    print("\nBasic descriptive information:")
    print(df.describe(include="all"))


# 4. Preprocessing
def prepare_features_and_target(df: pd.DataFrame):
    """Prepare X and y after validating target and selecting useful features."""
    if "Survived" not in df.columns:
        raise ValueError("Target column 'Survived' not found in dataset.")

    target_column = "Survived"
    y = df[target_column]

    # Exclude identifier / irrelevant columns.
    excluded_columns = {
        "Survived": "Target column (cannot be used as input feature)",
        "PassengerId": "Identifier column (does not describe passenger characteristics)",
        "Name": "High-cardinality text field not required for this baseline model",
        "Ticket": "High-cardinality ticket code with many unique values",
        "Cabin": "Very high missing values (most entries are null)",
    }

    print("\nExcluded columns and reasons:")
    for column, reason in excluded_columns.items():
        if column in df.columns:
            print(f"- {column}: {reason}")

    selected_features = [
        "Pclass",
        "Gender",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    ]

    missing_selected = [col for col in selected_features if col not in df.columns]
    if missing_selected:
        raise ValueError(
            f"Expected feature columns are missing from dataset: {missing_selected}"
        )

    X = df[selected_features]
    return X, y


def create_preprocessing_pipeline() -> ColumnTransformer:
    """Create preprocessing for numeric and categorical columns."""
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Gender", "Embarked"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


# 5. Train-Test Split
# 6. Create Decision Tree
# 7. Train Model
# 8. Make Predictions
# 9. Evaluate Model
def train_and_evaluate_model(X: pd.DataFrame, y: pd.Series) -> None:
    """Split data, train model, predict, and print required evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = create_preprocessing_pipeline()

    model = DecisionTreeClassifier(random_state=42)

    decision_tree_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    decision_tree_pipeline.fit(X_train, y_train)
    y_pred = decision_tree_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print("Positive class: 1 (Survived)")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


# 10. Conclusion
def print_conclusion() -> None:
    """Print final assignment conclusion."""
    print(
        "\nThe Decision Tree Classifier was trained successfully on the Titanic "
        "dataset and evaluated on the test set."
    )


def main() -> None:
    dataset_path = "/home/runner/work/EX18/EX18/Titanic-Dataset.csv"
    df = load_dataset(dataset_path)

    explore_dataset(df)
    X, y = prepare_features_and_target(df)
    train_and_evaluate_model(X, y)
    print_conclusion()


if __name__ == "__main__":
    main()
