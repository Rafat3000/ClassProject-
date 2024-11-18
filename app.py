import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler 
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def main(model_name,model,model_params,test_size , random_state):
    # load data
    df=pd.read_excel('glass (Imbalanced).xlsx')
    
    df.drop_duplicates()
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    scaler = MinMaxScaler()  
    scaled = scaler.fit_transform(df.values) 
    scaled_df = pd.DataFrame(scaled, columns=df.columns)
    X = df.drop('Class', axis=1)  # all columns ixcludet the target 
    y = df['Class']  # Target 
    
    

    # split data
    x_train , x_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)
    
    with mlflow.start_run(run_name=model_name):
       # log params
    
        mlflow.log_params({"test_size":test_size , "random_state":random_state})
        
        # loop over model params
        for param_name , param_value in model_params.items():
            mlflow.log_params({param_name:param_value})
        
        # build model
        model.set_params(**model_params)
        
        # train model
        model.fit(x_train , y_train)
        # predict
        y_pred = model.predict(x_test)
        # evaluate
        acc = accuracy_score(y_test , y_pred)
        mlflow.log_metric("accuracy" , acc)
         
        

if __name__ == "__main__":    
    # use expr
    mlflow.set_experiment("iris_classification10")

    models = [
        ("KNeighborsClassifier", KNeighborsClassifier(), {"n_neighbors": 3}),
        ("DecisionTreeClassifier", DecisionTreeClassifier(), {}),
        ("RandomForestClassifier", RandomForestClassifier(), {"n_estimators": 27, "criterion": 'entropy'}),
        ("AdaBoostClassifier", AdaBoostClassifier(algorithm="SAMME"), {"n_estimators": 50})
    ]



    # param
    train_size = 0.20
    random_state = 42

    for model_name , model , model_params in models:
        main(model_name,model,model_params,train_size,random_state)
