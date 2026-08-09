import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def create_features(df):
    df = df.copy()
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalRooms'] = df['TotRmsAbvGrd'] + df['BsmtFinSF1'] / 100
    df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
    
    df['Age'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    
    df['SF_per_Room'] = df['GrLivArea'] / df['TotRmsAbvGrd'].clip(lower=1)
    df['LotArea_per_SF'] = df['LotArea'] / df['TotalSF'].clip(lower=1)
    
    df['Qual_Area'] = df['OverallQual'] * df['TotalSF']
    df['Qual_GrLivArea'] = df['OverallQual'] * df['GrLivArea']
    df['Year_Qual'] = df['YearBuilt'] * df['OverallQual'] / 100

    return df

df = pd.read_csv('train.csv')
y = df['SalePrice']
X_train = df.drop(['SalePrice', 'Id'], axis=1)

test = pd.read_csv('test.csv')
test_ids = test['Id']
X_test = test.drop('Id', axis=1)

x_train = create_features(X_train)
x_test = create_features(X_test)

print(f"Train shape: {x_train.shape}")
print(f"Test shape: {x_test.shape}")
print(f"Колонки совпадают - {all(x_train.columns == x_test.columns)}")

for col in x_train.columns:
    if x_train[col].dtype == 'object' or x_train[col].dtype == 'string':
        x_train[col] = x_train[col].fillna('None')
        x_test[col] = x_test[col].fillna('None')
    else:
        median_val = x_train[col].median()
        x_train[col] = x_train[col].fillna(median_val)
        x_test[col] = x_test[col].fillna(median_val)

categorial_features = x_train.select_dtypes(include=['object', 'string']).columns.tolist()
print()
print(f"Категориальные фичи: {len(categorial_features)}")

X_train, X_val, y_train, y_val = train_test_split(
    x_train, y, test_size=0.2, random_state=42
)

model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=8,
    subsample=0.78,
    min_data_in_leaf=5,
    random_seed=42,
    loss_function='RMSE',
    eval_metric='RMSE',
    verbose=100,
    early_stopping_rounds=150
)

model.fit(
    X_train, y_train,
    cat_features=categorial_features,
    eval_set=(X_val, y_val)
)

y_pred_val = model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
print()
print(f"RMSE на валид: ${rmse_val:,.2f}")

y_pred_test = model.predict(x_test)

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': y_pred_test
})
submission.to_csv('submis.csv', index=False)
print(submission.head())
