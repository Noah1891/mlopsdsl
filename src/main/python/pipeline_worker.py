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
from pathlib import Path

context = {
    "df": None,
    "y": None,
    "target": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "raw_features": [],
    "selected_features": [],
    "transformers": {},
    "transform_log": [],
    "first_transform": True,
    "baseline_df": None,
    "model": None,
    "path": None,
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

def send_response(status, message, file_path="", code=0):
    """Helper function, that produces JSON in the Rascal Response-ADT format"""
    res = {
        "status": status,
        "message": message,
        "modelFilePath": file_path,
        "code": code
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
                    send_response("ERROR", f"File not found: {path}", code=1)
                    continue
                    
                context["df"] = pd.read_csv(path)

                if context["target"] not in context["df"].columns:
                    send_response("ERROR", f"Specified target {context['target']} is not a column in loaded CSV.", code=2)
                    continue

                context["y"] = context["df"][context["target"]]
                context["df"] = context["df"].drop(columns=[context["target"]])
                context["raw_features"] = list(context["df"].columns)
                
                send_response("SUCCESS", f"CSV loaded successfully. Form: {context['df'].shape}")
            
            elif cmd == "SPLIT":
                ratio = float(request["ratio"]);
                random_state = int(request["randomState"]);
                X = context["df"]
                y = context["y"]
                
                if X is None:
                    send_response("ERROR", "No data loaded. Split not possible.")
                    continue
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=random_state)
                context["X_train"] = X_train
                context["X_test"] = X_test
                context["y_train"] = y_train
                context["y_test"] = y_test
                
                send_response("SUCCESS", f"Data splitted (Train: {len(X_train)}, Test: {len(X_test)})")
            
            elif cmd == "SELECT":
                context["selected_features"] = request["features"]

                if context["X_train"] is not None:
                    missing = [f for f in context["selected_features"] if f not in context["X_train"].columns]
                    if missing:
                        send_response("ERROR", f"Columns not found in Dataframe: {missing}", code=3)
                        continue
                
                    context["X_train"] = context["X_train"][context["selected_features"]]
                    context["X_test"] = context["X_test"][context["selected_features"]]
                
                else:
                    if context["df"] is None:
                        send_response("ERROR", "No data loaded.")
                        continue

                    missing = [f for f in context["selected_features"] if f not in context["df"].columns]
                    if missing:
                        send_response("ERROR", f"Columns not found in Dataframe: {missing}", code=3)
                        continue
            
                    context["df"] = context["df"][context["selected_features"]]

                kept = context["selected_features"]
                send_response("SUCCESS", f"Features selected successfully. Kept columns: {kept}")
            
            elif cmd == "TRANSFORM":
                action = request["action"]
                feature = request["feature"]
                method = request["method"]

                if context["first_transform"]:
                    source = context["X_train"] if context["X_train"] is not None else context["df"]
                    context["baseline_df"] = source.copy()
                    context["first_transform"] = False

                context["transform_log"] += [(action, feature, method)]
                
                working_on_split = context["X_train"] is not None
                
                if action == "scale":
                    scaler = SCALERS[method]()
                    if working_on_split:
                        if feature not in context["X_train"].columns:
                            send_response("ERROR", f"Column not found in Dataframe: {feature}", code=3)
                            continue
                        context["X_train"][[feature]] = scaler.fit_transform(context["X_train"][[feature]])
                        context["X_test"][[feature]] = scaler.transform(context["X_test"][[feature]])
                        context["transformers"][f"scaler_{feature}"] = scaler
                    else:
                        if feature not in context["df"].columns:
                            send_response("ERROR", f"Column not found in Dataframe: {feature}", code=3)
                            continue
                        context["df"][[feature]] = scaler.fit_transform(context["df"][[feature]])
                        context["transformers"][f"scaler_{feature}"] = scaler
                        
                elif action == "fillna":
                    imputer = SimpleImputer(strategy=method)
                    if working_on_split:
                        if feature not in context["X_train"].columns:
                            send_response("ERROR", f"Column not found in Dataframe: {feature}", code=3)
                            continue
                        context["X_train"][[feature]] = imputer.fit_transform(context["X_train"][[feature]])
                        context["X_test"][[feature]] = imputer.transform(context["X_test"][[feature]])
                        context["transformers"][f"imputer_{feature}"] = imputer
                    else:
                        if feature not in context["df"].columns:
                            send_response("ERROR", f"Column not found in Dataframe: {feature}", code=3)
                            continue
                        context["df"][[feature]] = imputer.fit_transform(context["df"][[feature]])
                        context["transformers"][f"imputer_{feature}"] = imputer
                        
                elif action == "encode":
                    if method == "onehot":
                        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                        if working_on_split:
                            if feature not in context["X_train"].columns:
                                send_response("ERROR", f"Column not found in Dataframe: {feature}", code=3)
                                continue
                            train_encoded = ohe.fit_transform(context["X_train"][[feature]])
                            test_encoded = ohe.transform(context["X_test"][[feature]])                         
                            new_cols = ohe.get_feature_names_out([feature])                          
                            train_df_encoded = pd.DataFrame(train_encoded, columns=new_cols, index=context["X_train"].index)
                            test_df_encoded = pd.DataFrame(test_encoded, columns=new_cols, index=context["X_test"].index)                           
                            context["X_train"] = pd.concat([context["X_train"].drop(columns=[feature]), train_df_encoded], axis=1)
                            context["X_test"] = pd.concat([context["X_test"].drop(columns=[feature]), test_df_encoded], axis=1)
                            context["transformers"][f"encoder_{feature}"] = ohe
                        else:
                            if feature not in context["df"].columns:
                                send_response("ERROR", f"Column not found in Dataframe: {feature}", code=3)
                                continue
                            df_encoded = ohe.fit_transform(context["df"][[feature]])
                            new_cols = ohe.get_feature_names_out([feature])                        
                            df_encoded_pandas = pd.DataFrame(df_encoded, columns=new_cols, index=context["df"].index)                           
                            context["df"] = pd.concat([context["df"].drop(columns=[feature]), df_encoded_pandas], axis=1)                            
                            context["transformers"][f"encoder_{feature}"] = ohe        
                    else: #label
                        le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                        if working_on_split:
                            if feature not in context["X_train"].columns:
                                send_response("ERROR", f"Column not found in Dataframe: {feature}", code=3)
                                continue
                            context["X_train"][[feature]] = le.fit_transform(context["X_train"][[feature]])
                            context["X_test"][[feature]] = le.transform(context["X_test"][[feature]])
                            context["transformers"][f"encoder_{feature}"] = le
                        else:
                            if feature not in context["df"].columns:
                                send_response("ERROR", f"Column not found in Dataframe: {feature}", code=3)
                                continue
                            context["df"][[feature]] = le.fit_transform(context["df"][[feature]])
                            context["transformers"][f"encoder_{feature}"] = le 
                send_response("SUCCESS", f"Transformation {action}({method}) applied to {feature}.")

            elif cmd == "TRAIN":
                algo = request["algo"]
                hyperparams = request["hyperparameters"]
                model_dir = request["modelDir"]

                if context["first_transform"]:
                    source = context["X_train"] if context["X_train"] is not None else context["df"]
                    context["baseline_df"] = source.copy()
                
                if context["X_train"] is None:
                    context["X_train"] = context["df"]
                    context["y_train"] = context["y"]
                
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

                try:
                    model = ALGORITHMS[algo](**typed_params)
                except TypeError as e:
                    send_response("ERROR", f"Hyperparameter included unexpected keyword: {str(e)}")
                    continue
                try:
                    model.fit(context["X_train"], context["y_train"])
                except ValueError as e:
                    send_response("ERROR",f"Model training failed: {str(e)}")
                    continue
                except TypeError as e:
                    send_response("ERROR",f"Hyperparameter has wrong type: {str(e)}")
                    continue
                context["model"] = model
                
                os.makedirs(model_dir, exist_ok=True)
                file_name = f"{algo}_model.pkl"
                model_file_path = Path(os.path.join(model_dir, file_name)).resolve()
                context["path"] = model_file_path
                deployment_artifact = {
                    "model": context["model"],
                    "transformers": context["transformers"],
                    "transform_log": context["transform_log"],
                    "selected_features": context["selected_features"] if context["selected_features"] else context["raw_features"],
                    "transformed_columns": list(context["X_train"].columns)
                }
                joblib.dump(deployment_artifact, model_file_path)
                
                send_response("SUCCESS", f"Model {algo} trained successfully.", str(model_file_path))
            
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

            elif cmd == "MONITOR":
                methods = request["methods"]
                features = request["features"]
                windows = request["windows"]
                thresholds = request["thresholds"]

                deployment_artifact = joblib.load(context["path"])
                deployment_artifact["baseline_df"] = context["baseline_df"][features]
                deployment_artifact["methods"] = methods
                deployment_artifact["windows"] = windows
                deployment_artifact["thresholds"] = thresholds
                joblib.dump(deployment_artifact, context["path"])
                send_response("SUCCESS", f"Monitored features selected successfully: {features}")

            else:
                send_response("ERROR", f"Unknown command: {cmd}")
                
        except Exception as e:
            send_response("ERROR", f"Python exception: {str(e)}")

if __name__ == "__main__":
    main()