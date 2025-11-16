import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle

def create_model(data):
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']


    # scaler the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # testing the model
    y_pred = model.predict(X_test)
    acc = model.score(X_test, y_test)
    # print("Accuracy:", acc)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f'Accuracy: {acc} \nClassification report: {report}')

    return model,scaler




def get_clean_data():
    data = pd.read_csv("../data/breast_cancer_data.csv")
    data.drop(columns=['Unnamed: 32','id'],axis=1,inplace=True)
    data.diagnosis = data.diagnosis.map({'B':0, 'M':1})
    return data


def main():
    # function for cleaning the data
    data = get_clean_data()

    # function for model creation
    model, scaler = create_model(data)

    with open('../model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('../model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)




if __name__ == '__main__':
    main()

