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


def fill_missing(df, stats=None):
    # В этом датасете нан часто значит, что объекта нет, а не что значение не заполнено
    # stats - статистики только с трейна (чтобы не было утечки в вал и тесте)
    # если stats = нан, считаем их из df и возвращаем вместе с результатом
    df = df.copy()
    compute_stats = stats is None
    if compute_stats:
        stats = {} # пересоздаём, чтобы он был уникальным при каждом вызове функции

    # нет объекта - категория none
    # PoolQC - кач-во бассейна, MiscFeature - доп особенность дома
    # Alley - тип подъезда/аллеи к дому, Fence - забор
    # MasVnrType - тип фасада, FireplaceQu - качество камина
    none_cat_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence',
        'MasVnrType', 'FireplaceQu',
    ]

    # гараж: (type,finish, qual, cond - тип, отделка, кач-во, сост.)
    # нан вместе с GarageArea = 0, значит гаража нет
    none_cat_cols += ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

    # подвал: (qual, cond, exposure, finType - кач-во, сост, выход наружу, тип отделки)
    # нан значит, что подвала нет
    none_cat_cols += [
        'BsmtQual', 'BsmtCond', 'BsmtExposure',
        'BsmtFinType1', 'BsmtFinType2',
    ]

    for col in none_cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    # нет объекта - число 0
    # GarageYrBlt, если гаража нет, то 0, а не медиана других
    # GarageArea / GarageCars - площадь и число машин
    # Bsmt * SF / Bsmt * Bath - площади и ванные в подвале
    # MasVnrArea - площадь фасада, PoolArea - площадь бассейна
    zero_num_cols = [
        'GarageYrBlt', 'GarageArea', 'GarageCars',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath',
        'MasVnrArea', 'PoolArea',
    ]
    for col in zero_num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # длина фасада участка по улице
    # в одном районе участки похожи, медиана по району лучше даст рез-т, чем общая
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        if compute_stats:
            stats['lotfrontage_by_nb'] = df.groupby('Neighborhood')['LotFrontage'].median()
            stats['lotfrontage_global'] = df['LotFrontage'].median()

        df['LotFrontage'] = df['LotFrontage'].fillna(df['Neighborhood'].map(stats['lotfrontage_by_nb']))
        df['LotFrontage'] = df['LotFrontage'].fillna(stats['lotfrontage_global'])

    # редкие категории с пропусками
    mode_cat_cols = [
        'Electrical', 'MSZoning', 'KitchenQual',
        'Exterior1st', 'Exterior2nd', 'SaleType', 'Utilities',
    ]
    for col in mode_cat_cols:
        # if col not in df.columns:
        #     continue
        if compute_stats:
            mode = df[col].mode()
            stats[f'mode_{col}'] = mode.iloc[0] if len(mode) else 'None'
        df[col] = df[col].fillna(stats[f'mode_{col}'])

    if 'Functional' in df.columns:
        df['Functional'] = df['Functional'].fillna('Typ')

    # оставшиеся
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype == 'object' or str(df[col].dtype) == 'string':
                df[col] = df[col].fillna('None')
            else:
                if compute_stats:
                    stats[f'median_{col}'] = df[col].median()
                fill_val = stats.get(f'median_{col}', df[col].median())
                if pd.isna(fill_val):
                    fill_val = 0
                df[col] = df[col].fillna(fill_val)

    return df, stats


def simple_fill_missing(train_df, other_dfs):
    train_df = train_df.copy()
    others = [d.copy() for d in other_dfs]

    for col in train_df.columns:
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            train_df[col] = train_df[col].fillna('None')
            for d in others:
                d[col] = d[col].fillna('None')
        else:
            median_val = train_df[col].median()
            if pd.isna(median_val):
                median_val = 0
            train_df[col] = train_df[col].fillna(median_val)
            for d in others:
                d[col] = d[col].fillna(median_val)

    return train_df, others


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

# делим выборку для валидации
X_tr, X_vali, y_tr, y_vali = train_test_split(
    X_train, y, test_size=0.2, random_state=42
)

# заполняем пропуски
X_tr, impute_stats = fill_missing(X_tr)
X_vali, _ = fill_missing(X_vali, stats=impute_stats)
x_test, _ = fill_missing(X_test, stats=impute_stats)

# создаём свои фичи
X_tr = create_features(X_tr)
X_vali = create_features(X_vali)
x_test = create_features(x_test)

print(f"Колонки совпадают - {all(X_tr.columns == x_test.columns)}")

categorial_features = X_tr.select_dtypes(include=['object', 'string']).columns.tolist()
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
    early_stopping_rounds=150,
)

model.fit(
    X_tr, y_tr,
    cat_features=categorial_features,
    eval_set=(X_vali, y_vali),
)

y_pred_val = np.expm1(model.predict(X_vali))
y_val_price = np.expm1(y_vali)
rmsle_val = np.sqrt(mean_squared_log_error(y_val_price, y_pred_val))
rmse_val = np.sqrt(mean_squared_error(y_val_price, y_pred_val))
print()
print(f"Best Iter: {model.get_best_iteration()}")
print(f"RMSLE на валид: {rmsle_val:,.3f}")
print(f"RMSE на валид: {rmse_val:,.3f}")

# финал, склеиваем уже заполненные куски
n_trees = model.get_best_iteration() + 1
X_full = pd.concat([X_tr, X_vali], axis=0)
y_full = pd.concat([y_tr, y_vali], axis=0)

final_model = CatBoostRegressor(
    iterations=n_trees,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=8,
    subsample=0.78,
    min_data_in_leaf=5,
    random_seed=42,
    loss_function='RMSE',  # т.к. на вход подаём логи цены
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
    'SalePrice': y_pred_test,
})
submission.to_csv('submis.csv', index=False)
print(submission.head())
