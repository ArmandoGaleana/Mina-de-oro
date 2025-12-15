train_df = pd.read_csv('/datasets/gold_recovery_train.csv')
test_df = pd.read_csv('/datasets/gold_recovery_test.csv')
full_df = pd.read_csv('/datasets/gold_recovery_full.csv')

display(train_df.head())
display(test_df.head())
display(full_df.head())

# Variables
C = train_df['rougher.output.concentrate_au']
F = train_df['rougher.input.feed_au']
T = train_df['rougher.output.tail_au']
recovery_real = train_df['rougher.output.recovery']

# Formula para la variable calculada y la real:
recovery_calc = ((C * (F - T)) / (F * (C - T))) * 100
recovery_real = train_df['rougher.output.recovery']

# Reemplazar valores infinitos por NaN
recovery_calc = recovery_calc.replace([np.inf, -np.inf], np.nan)

# Eliminar valores donde recovery_real o recovery_calc sea NaN
mask = (~recovery_real.isna()) & (~recovery_calc.isna())
recovery_real_clean = recovery_real[mask]
recovery_calc_clean = recovery_calc[mask]

mae = mean_absolute_error(recovery_real_clean, recovery_calc_clean)
print(f"MAE entre recuperación calculada y real: {mae:.4e}")

missing_cols = set(train_df.columns) - set(test_df.columns)

# Identificar columnas objetivo
target_columns = ['rougher.output.recovery', 'final.output.recovery']

# Eliminar columna de fecha si existe
for df in [train_df, test_df]:
    if 'date' in df.columns:
        df.drop(columns=['date'], inplace=True)

# Separar features y targets
features_train = train_df.drop(columns=target_columns, errors='ignore')
targets_train = train_df[target_columns]

# Asegurar mismas columnas en train y test
missing_cols = set(features_train.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0  # si faltan columnas en test, las rellenamos con ceros

features_test = test_df[features_train.columns]

# Manejo de valores faltantes
features_train = features_train.fillna(features_train.mean())
targets_train = targets_train.loc[features_train.index]  # sincronizar filas
features_test = features_test.fillna(
    features_train.mean())  # imputar con media del train

# Escalado
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

print("Tamaño train:", features_train_scaled.shape)
print("Tamaño test:", features_test_scaled.shape)
print("Targets disponibles:", targets_train.head())

for metal in ['au', 'ag', 'pb']:
    plt.figure(figsize=(10, 4))
    for stage in ['rougher.input', 'rougher.output', 'final.output']:
        col = f"{stage}.concentrate_{metal}" if f"{stage}.concentrate_{metal}" in train_df.columns else f"{stage}.concentrate.{metal}"
        if col in train_df.columns:
            plt.plot(train_df[col], label=stage)
    plt.title(f'Concentración de {metal.upper()} por etapa')
    plt.legend()
    plt.grid(True)
    plt.show()

    train_sizes = train_df['primary_cleaner.input.feed_size']
test_sizes = test_df['primary_cleaner.input.feed_size']

plt.hist(train_sizes, bins=50, alpha=0.6, label='Train')
plt.hist(test_sizes, bins=50, alpha=0.6, label='Test')
plt.legend()
plt.title('Distribución del tamaño de partículas')
plt.xlabel('Tamaño')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

for stage in ['rougher.input', 'rougher.output', 'final.output']:
    cols = [f"{stage}.concentrate_au",
            f"{stage}.concentrate_ag", f"{stage}.concentrate_pb"]
    if all(c in train_df.columns for c in cols):
        total = train_df[cols].sum(axis=1)
        total.hist(bins=100)
        plt.title(f'Total concentración en etapa: {stage}')
        plt.xlabel('Total')
        plt.ylabel('Frecuencia')
        plt.grid(True)
        plt.show()


def smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100


def final_smape(y_rougher_true, y_rougher_pred, y_final_true, y_final_pred):
    return 0.25 * smape(y_rougher_true, y_rougher_pred) + 0.75 * smape(y_final_true, y_final_pred)


# Convertir smape en scorer para usar en cross_val_score
smape_scorer = make_scorer(smape, greater_is_better=False)

# Features y targets
target_columns = ['rougher.output.recovery', 'final.output.recovery']

X_train = features_train_scaled
X_test = features_test_scaled
y_train = targets_train

# Revisar valores nulos e infinitos
print("NaNs en features_train:", np.isnan(features_train_scaled).sum())
print("NaNs en features_test:", np.isnan(features_test_scaled).sum())
print("NaNs en targets_train:\n", targets_train.isna().sum())

# Reemplazar infinitos en el dataset original
features_train = features_train.replace([np.inf, -np.inf], np.nan)
features_test = features_test.replace([np.inf, -np.inf], np.nan)

# Rellenar NaN con la media
features_train = features_train.fillna(features_train.mean())
features_test = features_test.fillna(features_train.mean())

# Escalar nuevamente
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

# Alinear targets con features
mask = ~targets_train.isna().any(axis=1)
features_train_scaled = features_train_scaled[mask]
targets_train = targets_train[mask]

print("Tamaño features:", features_train_scaled.shape)
print("Tamaño targets:", targets_train.shape)

# Modelos a evaluar
X_train = features_train_scaled
y_train = targets_train.copy()

# Crear máscara para eliminar filas con NaN en targets
mask = ~y_train.isna().any(axis=1)
X_train = X_train[mask]
y_train = y_train.loc[mask]

print("NaNs en X_train:", np.isnan(X_train).sum())
print("NaNs en y_train:\n", y_train.isna().sum())

# Modelos a evaluar
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    # Modelo rougher
    model.fit(X_train, y_train['rougher.output.recovery'])
    rougher_pred = model.predict(X_train)

    # Modelo final
    model.fit(X_train, y_train['final.output.recovery'])
    final_pred = model.predict(X_train)

    # Calcular métrica final sMAPE
    score = final_smape(
        y_train['rougher.output.recovery'], rougher_pred,
        y_train['final.output.recovery'], final_pred
    )

    results[name] = score

print("Resultados sMAPE:", results)

# Resultados
print("Resultados sMAPE final ponderado:")
for model, score in results.items():
    print(f"{model}: {score:.4f}")
