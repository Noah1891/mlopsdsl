import joblib
import pandas as pd

deployment_artifact=joblib.load('LogReg_model.pkl')
deployment_artifact

transformers = deployment_artifact['transformers']
model = deployment_artifact['model']

age_df = pd.DataFrame([[22]], columns=['Age'])
age_imputed = transformers['imputer_Age'].transform(age_df)

pclass_df = pd.DataFrame([[3]], columns=['Pclass'])
pclass_encoded = transformers['encoder_Pclass'].transform(pclass_df)
pclass_cols = transformers['encoder_Pclass'].get_feature_names_out(['Pclass'])

sex_df = pd.DataFrame([['male']], columns=['Sex'])
sex_encoded = transformers['encoder_Sex'].transform(sex_df)
sex_cols = transformers['encoder_Sex'].get_feature_names_out(['Sex'])

fare_df = pd.DataFrame([[7.25]], columns=['Fare'])
fare_scaled = transformers['scaler_Fare'].transform(fare_df)

sample = pd.concat([
    pd.DataFrame(age_imputed, columns=['Age']),
    pd.DataFrame([[1, 0]], columns=['SibSp', 'Parch']),  # untouched features
    pd.DataFrame(fare_scaled, columns=['Fare']),
    pd.DataFrame(pclass_encoded, columns=pclass_cols),
    pd.DataFrame(sex_encoded, columns=sex_cols),
], axis=1)

# must match X_train's column order exactly:
sample = sample[model.feature_names_in_]
print(model.predict(sample))