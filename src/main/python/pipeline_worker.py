import sys
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, root_mean_squared_error
import joblib

context = {
    "df": None,
    "target": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "transformers": {},
    "model": None,
    "metrics": {}
}

SCALERS = {
    "std": StandardScaler,
    "minmax": MinMaxScaler
}

ENCODERS = {
    "onehot": OneHotEncoder,
    "label": OrdinalEncoder
}

ALGORITHMS = {
    "RandomForest": RandomForestClassifier,
    "LinReg": LinearRegression,
    "LogReg": LogisticRegression
}

METRICS = {
    "acc": accuracy_score,
    "f1": f1_score,
    "pre": precision_score,
    "rec": recall_score,
    "mse": mean_squared_error,
    "rmse": root_mean_squared_error
}

def send_response(status, message, model_path=""):
    """Helper function, thath produces JSON in the Rascal Response-ADT format"""
    res = {
        "status": status,
        "message": message,
        "modelPath": model_path
    }
    print(json.dumps(res))
    sys.stdout.flush()

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
            
        try:
            request = json.loads(line)
            cmd = request.get("cmd")
            
            if cmd == "LOAD":
                path = request["path"]
                context["target"] = request["target"]
                
                if not os.path.exists(path):
                    send_response("ERROR", f"File not found: {path}")
                    continue
                    
                context["df"] = pd.read_csv(path)

                if context["target"] not in context["df"].columns:
                    send_response("ERROR", f"Specified target {context["target"]} is not a column in loaded CSV.")
                
                send_response("SUCCESS", f"CSV loaded successfully. Form: {context['df'].shape}")
            
            elif cmd == "SPLIT":
                ratio = float(request["ratio"]);
                random_state = int(request["randomState"]);
                df = context["df"]
                y_col = context["target"]
                
                if df is None:
                    send_response("ERROR", "No data loaded. Split not possible.")
                    continue
                    
                X = df.drop(columns=[y_col])
                y = df[y_col]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=random_state)
                context["X_train"] = X_train
                context["X_test"] = X_test
                context["y_train"] = y_train
                context["y_test"] = y_test
                
                send_response("SUCCESS", f"Data splitted (Train: {len(X_train)}, Test: {len(X_test)})")
            
            elif cmd == "SELECT":
                selected_features = request["features"]

                if context["X_train"] is not None:
                    missing = [f for f in selected_features if f not in context["X_train"].columns]
                    if missing:
                        send_response("ERROR", f"Columns not found in Dataframe: {missing}")
                        continue
                
                    context["X_train"] = context["X_train"][selected_features]
                    context["X_test"] = context["X_test"][selected_features]
                
                else:
                    if context["df"] is None:
                        send_response("ERROR", "No data loaded.")
                        continue

                    missing = [f for f in selected_features if f not in context["df"].columns]
                    if missing:
                        send_response("ERROR", f"Columns not found in Dataframe: {missing}")
                        continue
            
                    context["df"] = context["df"][selected_features]
                
                send_response("SUCCESS", f"Features selected successfully. Kept columns: {selected_features}")
            
            elif cmd == "TRANSFORM":
                action = request["action"]
                feature = request["feature"]
                method = request["method"]
                
                working_on_split = context["X_train"] is not None

                
                if action == "scale":
                    scaler = SCALERS[method]()
                    if working_on_split:
                        if feature not in context["X_train"].columns:
                            send_response("ERROR", f"Column not found in Dataframe: {feature}")
                            continue
                        context["X_train"][[feature]] = scaler.fit_transform(context["X_train"][[feature]])
                        context["X_test"][[feature]] = scaler.transform(context["X_test"][[feature]])
                        context["transformers"][f"scaler_{feature}"] = scaler
                    else:
                        if feature not in context["df"].columns:
                            send_response("ERROR", f"Column not found in Dataframe: {feature}")
                            continue
                        context["df"][[feature]] = scaler.fit_transform(context["df"][[feature]])
                        context["transformers"][f"scaler_{feature}"] = scaler
                        
                elif action == "fillna":
                    imputer = SimpleImputer(strategy=method)
                    if working_on_split:
                        context["X_train"][[feature]] = imputer.fit_transform(context["X_train"][[feature]])
                        context["X_test"][[feature]] = imputer.transform(context["X_test"][[feature]])
                        context["transformers"][f"imputer_{feature}"] = imputer
                    else:
                        context["df"][[feature]] = imputer.fit_transform(context["df"][[feature]])
                        context["transformers"][f"imputer_{feature}"] = imputer
                        
                elif action == "encode":
                    if method == "onehot":
                        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                        if working_on_split:
                            train_encoded = ohe.fit_transform(context["X_train"][[feature]])
                            test_encoded = ohe.transform(context["X_test"][[feature]])                         
                            new_cols = ohe.get_feature_names_out([feature])                          
                            train_df_encoded = pd.DataFrame(train_encoded, columns=new_cols, index=context["X_train"].index)
                            test_df_encoded = pd.DataFrame(test_encoded, columns=new_cols, index=context["X_test"].index)                           
                            context["X_train"] = pd.concat([context["X_train"].drop(columns=[feature]), train_df_encoded], axis=1)
                            context["X_test"] = pd.concat([context["X_test"].drop(columns=[feature]), test_df_encoded], axis=1)
                            context["transformers"][f"encoder_{feature}"] = ohe
                        else:
                            df_encoded = ohe.fit_transform(context["df"][[feature]])
                            new_cols = ohe.get_feature_names_out([feature])                        
                            df_encoded_pandas = pd.DataFrame(df_encoded, columns=new_cols, index=context["df"].index)                           
                            context["df"] = pd.concat([context["df"].drop(columns=[feature]), df_encoded_pandas], axis=1)                            
                            context["transformers"][f"encoder_{feature}"] = ohe        
                    else: #label
                        le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                        if working_on_split:
                            context["X_train"][[feature]] = le.fit_transform(context["X_train"][[feature]])
                            context["X_test"][[feature]] = le.transform(context["X_test"][[feature]])
                            context["transformers"][f"encoder_{feature}"] = le
                        else:
                            context["df"][[feature]] = le.fit_transform(context["df"][[feature]])
                            context["transformers"][f"encoder_{feature}"] = le 
                send_response("SUCCESS", f"Transformation {action}({method}) applied to {feature}.")

            elif cmd == "TRAIN":
                algo = request["algo"]
                hyperparams = request["hyperparameters"]
                model_dir = request["modelDir"]
                
                if context["X_train"] is None:
                    X = context["df"].drop(columns=[context["target"]])
                    y = context["df"][context["target"]]
                    context["X_train"], context["y_train"] = X, y
                
                if algo not in ALGORITHMS:
                    send_response("ERROR", f"Algorithm {algo} not supported.")
                    continue

                typed_params = {}
                for k, v in hyperparams.items():
                    if v in ("true", "false"):
                        typed_params[k] = (v == "true")
                        continue
                    try:
                        typed_params[k] = int(v)
                    except ValueError:
                        try:
                            typed_params[k] = float(v)
                        except ValueError:
                            typed_params[k] = v

                model = ALGORITHMS[algo](**typed_params)
                model.fit(context["X_train"], context["y_train"])
                context["model"] = model
                
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"{algo}_model.pkl")
                deployment_artifact = {
                    "model": context["model"],
                    "transformers": context["transformers"]
                }
                joblib.dump(deployment_artifact, model_path)
                
                send_response("SUCCESS", f"Model {algo} trained successfully.", model_path)
            
            elif cmd == "EVAL":

                metric = request["metric"]
                model = context["model"]

                if context["X_test"] is not None:
                    y_pred = model.predict(context["X_test"])
                    result = METRICS[metric](context["y_test"], y_pred)
                else:
                    y_pred = model.predict(context["X_train"])
                    result = METRICS[metric](context["y_train"], y_pred)
                
                context["metrics"][metric] = result
                send_response("SUCCESS", f"Evaluated model with metric {metric}: {result}")

            else:
                send_response("ERROR", f"Unknown command: {cmd}")
                
        except Exception as e:
            send_response("ERROR", f"Python exception: {str(e)}")

if __name__ == "__main__":
    main()