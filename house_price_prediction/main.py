import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, mean_squared_error
import matplotlib.pyplot as plt


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

# распределение цен (на графике видно, что оно нормальное (колокол) при логарифмировании)
plt.figure(figsize=(10, 6))
plt.hist(np.log1p(df['SalePrice']), bins=100, color='blue', alpha=0.7)
plt.title('Distribution of Sale Prices')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# смотрим зависимость площади от цены
# на графике видна линейная зав-сть, но есть 2 выброса, с большой площадью и низкой ценой
# от них надо избавиться, чтобы модель лучше нашла закономерность
plt.scatter(df['GrLivArea'], df['SalePrice'])
plt.title('GrLivArea vs SalePrice')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.grid(True)
plt.show()

# удалим эти плохие данные с помощью маски
price_per_sf = df['SalePrice'] / df['GrLivArea']
mask = (df['GrLivArea'] > 4000) & (price_per_sf < price_per_sf.median() * 0.5)
print(df.loc[mask, ['Id', 'GrLivArea', 'SalePrice']])
df = df[~mask].copy()
y = np.log1p(df['SalePrice'])
X_train = df.drop(['SalePrice', 'Id'], axis=1)

test = pd.read_csv('test.csv')
test_ids = test['Id']
X_test = test.drop('Id', axis=1)

x_train = create_features(X_train)
x_test = create_features(X_test)

print(f"Train shape: {x_train.shape}")
print(f"Test shape: {x_test.shape}")
print(f"Колонки совпадают - {all(x_train.columns == x_test.columns)}")

X_train, X_val, y_train, y_val = train_test_split(
    x_train, y, test_size=0.2, random_state=42
)

for col in X_train.columns:
    if X_train[col].dtype == 'object' or X_train[col].dtype == 'string':

        X_train[col] = X_train[col].fillna('None')
        x_test[col] = x_test[col].fillna('None')
        X_val[col] = X_val[col].fillna('None')
    else:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        x_test[col] = x_test[col].fillna(median_val)
        X_val[col] = X_val[col].fillna(median_val)

categorial_features = x_train.select_dtypes(include=['object', 'string']).columns.tolist()
print()
print(f"Категориальные фичи: {len(categorial_features)}")

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

y_pred_val = np.expm1(model.predict(X_val))
y_val_price = np.expm1(y_val)

rmsle_val = np.sqrt(mean_squared_log_error(y_val_price, y_pred_val))
rmse_val = np.sqrt(mean_squared_error(y_val_price, y_pred_val))
print()
print(f"RMSLE на валид: {rmsle_val:,.3f}")
print(f"RMSE на валид: {rmse_val:,.3f}")

# заново учимся на всех данных
best_iter = model.get_best_iteration()
X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])

final_model = CatBoostRegressor(
    iterations=best_iter + 1,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=8,
    subsample=0.78,
    min_data_in_leaf=5,
    random_seed=42,
    loss_function='RMSE', # т.к. на вход подаём логифрмы цен
    eval_metric='RMSE',
    verbose=100,
)
final_model.fit(
    X_full, y_full,
    cat_features=categorial_features,
)

y_pred_test = np.expm1(final_model.predict(x_test))

submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': y_pred_test
})
submission.to_csv('submis.csv', index=False)
print(submission.head())
