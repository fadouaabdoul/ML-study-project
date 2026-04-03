import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix



def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.dropna()

    df = df.drop("customerID", axis=1)

    return df



def preprocess_data(df):
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X = pd.get_dummies(X, drop_first=True)

    return X, y



def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)



def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test



def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model



def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def main():
    
    df = load_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df = clean_data(df)

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test = scale_data(X_train, X_test)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()