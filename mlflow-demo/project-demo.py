# import os
# import warnings
# import sys
# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import ElasticNet
# from urllib.parse import urlparse
# import mlflow
# from mlflow.models import infer_signature

# import logging
# import dagshub

# # Set up DAGsHub and MLflow tracking URI only once, before starting any runs

# dagshub.init(repo_owner='masaba019', repo_name='mlops', mlflow=True)

# logging.basicConfig(level=logging.WARN)
# logger = logging.getLogger(__name__)

# def eval_metrics(actual, pred):
#     rmse = np.sqrt(mean_squared_error(actual, pred))
#     mae = mean_absolute_error(actual, pred)
#     r2 = r2_score(actual, pred)
#     return rmse, mae, r2

# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")
#     np.random.seed(40)

#     # Read the wine-quality csv file from the URL
#     csv_url = (
#         "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
#     )
#     try:
#         data = pd.read_csv(csv_url, sep=";")
#     except Exception as e:
#         logger.exception(
#             "Unable to download training & test CSV, check your internet connection. Error: %s", e
#         )

#     # Split the data into training and test sets. (0.75, 0.25) split.
#     train, test = train_test_split(data)

#     # The predicted column is "quality" which is a scalar from [3, 9]
#     train_x = train.drop(["quality"], axis=1)
#     test_x = test.drop(["quality"], axis=1)
#     train_y = train[["quality"]]
#     test_y = test[["quality"]]

#     alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
#     l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

#     with mlflow.start_run():
#         lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
#         lr.fit(train_x, train_y)

#         predicted_qualities = lr.predict(test_x)

#         (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

#         print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
#         print("  RMSE: %s" % rmse)
#         print("  MAE: %s" % mae)
#         print("  R2: %s" % r2)

#         mlflow.log_param("alpha", alpha)
#         mlflow.log_param("l1_ratio", l1_ratio)
#         mlflow.log_metric("rmse", rmse)
#         mlflow.log_metric("r2", r2)
#         mlflow.log_metric("mae", mae)
        
#     remote_server_uri = "https://dagshub.com/masaba019/mlops.mlflow"
#     mlflow.set_tracking_uri(remote_server_uri)

#     # Infer model signature
#     signature = infer_signature(train_x, lr.predict(train_x))

#     # Log the model
#     mlflow.sklearn.log_model(
#         sk_model=lr,
#         name="model-demo",
#         signature=signature,
#         input_example=train_x[:5],  # Sample input for documentation
#     )

import os
import warnings
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import logging
import dagshub

# Set up tracking BEFORE any MLflow operations
remote_server_uri = "https://dagshub.com/masaba019/mlops.mlflow"
mlflow.set_tracking_uri(remote_server_uri)
dagshub.init(repo_owner='masaba019', repo_name='mlops', mlflow=True)

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # Log parameters and metrics
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        # Save model to a pickle file
        with open("model.pkl", "wb") as f:
            pickle.dump(lr, f)

        # Log the pickle file as an artifact
        mlflow.log_artifact("model.pkl", "model")

        # Save and log model info for reference
        with open("model_info.txt", "w") as f:
            f.write(f"ElasticNet Model\n")
            f.write(f"Alpha: {alpha}\n")
            f.write(f"L1 Ratio: {l1_ratio}\n")
            f.write(f"Training RMSE: {rmse}\n")
            f.write(f"Training R2: {r2}\n")
        mlflow.log_artifact("model_info.txt")
