import os
import mlflow
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# leitura dos dados
data = pd.read_csv('https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health.csv')

# preparação dos dados
features_to_remove = data.columns[7:]
X = data.drop(features_to_remove, axis=1)
y = data["fetal_health"]

scaler = StandardScaler()
X_df = scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=X.columns)

# divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                    y,
                                                    random_state=42,
                                                    test_size=0.3)

# configuração do MLFlow
MLFLOW_TRACKING_URI = 'https://dagshub.com/renansantosmendes/mlops-arda-t2.mlflow'
MLFLOW_TRACKING_USERNAME = 'renansantosmendes'
MLFLOW_TRACKING_PASSWORD = 'cc41cc48f8e489dd5b87404dd6f9720944e32e9b'
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ativação do autolog
mlflow.sklearn.autolog(log_models=True,
                       log_input_examples=True,
                       log_model_signatures=True)

# treinamento
grd_clf = GradientBoostingClassifier(max_depth=15,
                                     n_estimators=150,
                                     learning_rate=0.2)

with mlflow.start_run(run_name='gradiente_bosting_github') as run:
    grd_clf.fit(X_train, y_train)

print('Modelo treinado com sucesso!')
