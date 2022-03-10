import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    csv_url = (
        'https://github.com/samfennelly/mlflowdemo/blob/main/titanic_train.csv?raw=true'
    )
    data = pd.read_csv(csv_url)
    data = data.fillna(0)


    train, test = train_test_split(data)

    train_x = train[["age","fare","body"]]
    test_x = test[["age","fare","body"]]
    train_y = train[["survived"]]
    test_y = test[["survived"]]

    model = LogisticRegression().fit(train_x, train_y)
    
    predictions = model.predict(test_x)

    rmse, mae, r2 = eval_metrics(test_y,predictions)
    
    

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    mlflow.sklearn.log_model(model, "model")
