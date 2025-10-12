#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
dataset = pd.read_csv("C:/Users/jlopez04/Desktop/Facu/cuarto semestre/DMEyF/nuevabase.csv")


# In[2]:


dataset.head()


# ### OPTIMIZACION DE HIPERPARÁMETROS

# In[3]:


dataset["clase01"] = dataset["clase_ternaria"].isin(["BAJA+1", "BAJA+2"]).astype(int)


# In[4]:


# === CONFIGURACIÓN BASE ===
PARAM = {}

PARAM["experimento"] = 4070
PARAM["semilla_primigenia"] = 777781

# training y future
PARAM["train"] = [202101, 202102, 202103]
PARAM["future"] = [202104]

# hiperparámetros base de LightGBM
PARAM["lgb"] = {
    "num_iterations": 1000,
    "learning_rate": 0.012540013206725433,
    "num_leaves": 56,
    "max_depth": 7,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.6065424849326423,
    "bagging_fraction": 0.6868332757831337,
    "bagging_freq": 4,
    "lambda_l1": 0.003132685501561935,
    "lambda_l2": 4.7046025685189027e-08,
    "min_split_gain": 1.2999661394711282,
    "scale_pos_weight": 2.755429501539033,
    "max_bin": 31
}

PARAM["semilla_kaggle"] = 314159


# In[5]:


import numpy as np
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedShuffleSplit

# === VARIABLES ===
train_months = PARAM["train"]
eval_month   = PARAM["future"][0]
FEATURES = [c for c in dataset.columns if c not in ["numero_de_cliente", "foto_mes", "clase_ternaria", "clase01"]]

# === DATOS ===
mask_tr = dataset["foto_mes"].isin(train_months)
mask_ev = dataset["foto_mes"] == eval_month

X_tr_full = dataset.loc[mask_tr, FEATURES].values
y_tr_full = dataset.loc[mask_tr, "clase01"].astype(int).values
X_ev = dataset.loc[mask_ev, FEATURES].values
y_ev_baja2 = (dataset.loc[mask_ev, "clase_ternaria"] == "BAJA+2").astype(int).values

base = PARAM["lgb"]
seed = PARAM["semilla_primigenia"]

np.random.seed(seed)

# === FUNCIÓN DE GANANCIA LIMITADA A 15 000 ENVÍOS ===
def gain_metric(probas, y_true, gain_tp=780000, cost_fp=-20000, top_n=15000):
    """
    Calcula la ganancia considerando sólo los top_n clientes
    con mayor probabilidad de ser BAJA+2.
    """
    n = len(probas)
    top_n = min(top_n, n)  # por si hay menos de 15000 clientes
    order = np.argsort(-probas)[:top_n]
    gains = np.where(y_true[order] == 1, gain_tp, cost_fp)
    return float(np.sum(gains))

# === OBJETIVO OPTUNA ===
def objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": seed,
        "num_threads": -1,
        # Fijos desde tu base
        "learning_rate": base["learning_rate"],
        "max_depth": base["max_depth"],
        "bagging_freq": base["bagging_freq"],
        "bagging_fraction": base["bagging_fraction"],
        "feature_fraction": base["feature_fraction"],
        "lambda_l1": base["lambda_l1"],
        "lambda_l2": base["lambda_l2"],
        "min_split_gain": base["min_split_gain"],
        "scale_pos_weight": base["scale_pos_weight"],
        "max_bin": base["max_bin"],
        # A optimizar
        "num_leaves": trial.suggest_int("num_leaves", 32, 96),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 300, 3000)
    }

    # Split interno 80/20 para validación temprana
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, va_idx = next(sss.split(X_tr_full, y_tr_full))
    X_tr, y_tr = X_tr_full[tr_idx], y_tr_full[tr_idx]
    X_va, y_va = X_tr_full[va_idx], y_tr_full[va_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_va, label=y_va)

    model = lgb.train(
        params,
        train_set=dtrain,
        valid_sets=[dvalid],
        valid_names=["valid"],
        num_boost_round=1500,
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
    )

    # Predecimos el mes de evaluación y calculamos ganancia total
    prob_ev = model.predict(X_ev, num_iteration=model.best_iteration)
    total_gain = gain_metric(prob_ev, y_ev_baja2, top_n=16000)
    
    trial.set_user_attr("best_iteration", int(model.best_iteration))
    return total_gain

# === OPTIMIZACIÓN ===
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
study.optimize(objective, n_trials=12, show_progress_bar=True)

best = study.best_trial
bp = best.params

# === PARAMS OPTIMIZADOS ===
optimized_lgb = {
    "num_iterations": int(best.user_attrs["best_iteration"]),
    "learning_rate": base["learning_rate"],
    "num_leaves": int(bp["num_leaves"]),
    "max_depth": base["max_depth"],
    "min_data_in_leaf": int(bp["min_data_in_leaf"]),
    "feature_fraction": base["feature_fraction"],
    "bagging_fraction": base["bagging_fraction"],
    "bagging_freq": base["bagging_freq"],
    "lambda_l1": base["lambda_l1"],
    "lambda_l2": base["lambda_l2"],
    "min_split_gain": base["min_split_gain"],
    "scale_pos_weight": base["scale_pos_weight"],
    "max_bin": base["max_bin"]
}

print("Parámetros optimizados:", optimized_lgb)


# In[6]:


print("Mejor ganancia total:", study.best_value)
print("Ganancia promedio por cliente:", study.best_value / 15000)


# In[7]:


print("Parámetros optimizados:", optimized_lgb)


# In[8]:


campos_buenos = [c for c in dataset.columns if c not in ["clase_ternaria", "clase01"]]


# In[9]:


dataset["train"] = 0
dataset.loc[dataset["foto_mes"].isin(PARAM["train"]), "train"] = 1


# In[ ]:


# === CONFIGURACIÓN BASE === AHORA CON LOS PARAMETROS OPTIMOS
PARAM = {}

PARAM["experimento"] = 4070
PARAM["semilla_primigenia"] = 777781

# training y future
PARAM["train"] = [202101, 202102, 202103]
PARAM["future"] = [202104]

# hiperparámetros base de LightGBM
PARAM["lgb"] = {
  num_iterations: 1500
   learning_rate: 0.012540013206725433
   num_leaves: 74
   max_depth: 7
   min_data_in_leaf: 2516
   feature_fraction: 0.6065424849326423
   bagging_fraction: 0.6868332757831337
   bagging_freq: 4
   lambda_l1: 0.003132685501561935
   lambda_l2: 4.7046025685189027e-08
   min_split_gain: 1.2999661394711282
   scale_pos_weight: 2.755429501539033
   max_bin: 31
}

PARAM["semilla_kaggle"] = 314159


# In[10]:


#!pip install lightgbm
import lightgbm as lgb

# selecciono solo las filas de entrenamiento
train_mask = dataset["train"] == 1

dtrain = lgb.Dataset(
    data=dataset.loc[train_mask, campos_buenos].values,
    label=dataset.loc[train_mask, "clase01"].values
)


# In[11]:


import lightgbm as lgb
import numpy as np
import random

# fijo la semilla
np.random.seed(PARAM["semilla_primigenia"])
random.seed(PARAM["semilla_primigenia"])

# entreno el modelo
modelo = lgb.train(
    params={
        "objective": "binary",
        "max_bin": PARAM["lgb"]["max_bin"],
        "learning_rate": PARAM["lgb"]["learning_rate"],
        "num_iterations": PARAM["lgb"]["num_iterations"],
        "num_leaves": PARAM["lgb"]["num_leaves"],
        "max_depth": PARAM["lgb"]["max_depth"],
        "min_data_in_leaf": PARAM["lgb"]["min_data_in_leaf"],
        "feature_fraction": PARAM["lgb"]["feature_fraction"],
        "bagging_fraction": PARAM["lgb"]["bagging_fraction"],
        "bagging_freq": PARAM["lgb"]["bagging_freq"],
        "lambda_l1": PARAM["lgb"]["lambda_l1"],
        "lambda_l2": PARAM["lgb"]["lambda_l2"],
        "min_split_gain": PARAM["lgb"]["min_split_gain"],
        "scale_pos_weight": PARAM["lgb"]["scale_pos_weight"],
        "seed": PARAM["semilla_primigenia"]
    },
    train_set=dtrain
)


# In[12]:


importancias = modelo.feature_importance(importance_type="gain")
# Ojo: acá está en el mismo orden que en dtrain
columnas_reales = campos_buenos

df_importancias = pd.DataFrame({
    "variable": columnas_reales,
    "importancia": importancias
}).sort_values("importancia", ascending=False)

print(df_importancias.head(40))


# ### TESTEO EN ABRIL

# In[13]:


excluir_features = {"clase_ternaria", "clase01", "train"}
features = [c for c in dataset.columns if c not in excluir_features]


# In[14]:


mask_abril = dataset["foto_mes"] == 202104
X_abril = dataset.loc[mask_abril, features]


# In[15]:


proba_abril = modelo.predict(X_abril)


# In[20]:


prediccion_202104 = dataset.loc[mask_abril, ["numero_de_cliente","foto_mes"]].copy()
prediccion_202104["Predicted"] = (proba_abril > 0.0250).astype(int) 


# In[21]:


abril_real = dataset.loc[mask_abril, ["numero_de_cliente","foto_mes","clase_ternaria"]]

# merge
data_eval = abril_real.merge(prediccion_202104, on=["numero_de_cliente","foto_mes"], how="left")

# ganancia: +780000 si Predicted=1 y clase_ternaria=BAJA+2, -20000 si Predicted=1 y no es BAJA+2
data_eval["ganancia"] = np.where(
    data_eval["Predicted"] == 1,
    np.where(data_eval["clase_ternaria"] == "BAJA+2", 780000, -20000),
    0
)

ganancia_total = data_eval["ganancia"].sum()
contactados = (data_eval["Predicted"]==1).sum()
aciertos = ((data_eval["Predicted"]==1) & (data_eval["clase_ternaria"]=="BAJA+2")).sum()

print("Ganancia total:", f"{ganancia_total:,}".replace(",", "."))
print("Contactados:", f"{contactados:,}".replace(",", "."))
print("Aciertos:", f"{aciertos:,}".replace(",", "."))


# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# dataframe ordenado por probabilidad descendente
curva = pd.DataFrame({
    "numero_de_cliente": dataset.loc[mask_abril, "numero_de_cliente"],
    "proba": proba_abril,
    "es_baja2": (dataset.loc[mask_abril, "clase_ternaria"] == "BAJA+2").astype(int)
}).sort_values("proba", ascending=False).reset_index(drop=True)

# limitar a 30.000 envíos
curva = curva.iloc[:30_000].copy()

# ganancia marginal: +780k si es BAJA+2, -20k si no
curva["gan_marginal"] = np.where(curva["es_baja2"] == 1, 780_000, -20_000)

# acumulado
curva["gan_acum"] = curva["gan_marginal"].cumsum()

# punto óptimo dentro de esos 30k
idx_opt = curva["gan_acum"].idxmax()
k_opt   = idx_opt + 1
gan_max = curva.loc[idx_opt, "gan_acum"]

print(f"Mejor k (máx 30.000 envíos): {k_opt:,}")
print(f"Ganancia máxima: {gan_max:,}")

# --- graficar ---
plt.figure(figsize=(8,5))
plt.plot(curva.index+1, curva["gan_acum"])
plt.axvline(k_opt, color="red", linestyle="--", label=f"k óptimo = {k_opt}")
plt.axhline(gan_max, color="green", linestyle="--", label=f"Ganancia máx = {gan_max:,}")
plt.xlabel("Cantidad de envíos (limitado a 30k)")
plt.ylabel("Ganancia acumulada")
plt.title("Curva de envíos vs ganancia (máx 30k)")
plt.legend()
plt.show()


# In[ ]:


# predicciones (probabilidades) sobre dfuture
prediccion = modelo.predict(dfuture[campos_buenos].values)

# opcional: guardar en un DataFrame junto con los IDs
prediccion_df = dfuture[["numero_de_cliente", "foto_mes"]].copy()
prediccion_df["proba"] = prediccion

print(prediccion_df.head())


# ### TRAIN FINAL ENSAMBLE DE SEMILLAS

# In[1]:


import pandas as pd
dataset = pd.read_csv("C:/Users/jlopez04/Desktop/Facu/cuarto semestre/DMEyF/nuevabase.csv")


# In[2]:


dataset["clase01"] = dataset["clase_ternaria"].isin(["BAJA+1", "BAJA+2"]).astype(int)


# In[3]:


import numpy as np
import pandas as pd
import lightgbm as lgb

# ==========
# PARAMS (tus mismos valores)
# ==========
PARAM = {}
PARAM["experimento"] = 4070
PARAM["semilla_primigenia"] = 777781  # queda, pero NO se usa para entrenar: usamos las 5 que pediste
PARAM["train"]  = [202101, 202102, 202103, 202104]
PARAM["future"] = [202106]
PARAM["lgb"] = {
    "num_iterations": 1500,
    "learning_rate": 0.012540013206725433,
    "num_leaves": 74,
    "max_depth": 7,
    "min_data_in_leaf": 2516,
    "feature_fraction": 0.6065424849326423,
    "bagging_fraction": 0.6868332757831337,
    "bagging_freq": 4,
    "lambda_l1": 0.003132685501561935,
    "lambda_l2": 4.7046025685189027e-08,
    "min_split_gain": 1.2999661394711282,
    "scale_pos_weight": 2.755429501539033,
    "max_bin": 31
}
PARAM["semilla_kaggle"] = 314159

# ==========
# Semillas del ensamble (exactamente las que pediste)
# ==========
SEEDS = [777_781, 777_787, 777_817, 777_839, 777_857]

# ==========
# Preparación de datos
# ==========
excluir_features = {"clase_ternaria", "clase01", "train"}
features = [c for c in dataset.columns if c not in excluir_features]

mask_train = dataset["foto_mes"].isin(PARAM["train"])
mask_test  = dataset["foto_mes"].isin(PARAM["future"])

X_train = dataset.loc[mask_train, features]
y_train = (dataset.loc[mask_train, "clase_ternaria"] == "BAJA+2").astype(int)

X_test  = dataset.loc[mask_test, features]
id_test = dataset.loc[mask_test, ["numero_de_cliente", "foto_mes"]].copy()

# ==========
# Entrenamiento + predicción en ensamble de semillas
# ==========
num_boost_round = PARAM["lgb"].pop("num_iterations")  # usamos como control real

preds = []
models = []

for s in SEEDS:
    params = PARAM["lgb"].copy()
    # Respetamos todo y sólo fijamos semillas
    params.update({
        "seed": s,
        "feature_fraction_seed": s + 1,
        "bagging_seed": s + 2,
        "data_random_seed": s + 3,
        "verbose": -1,
        "objective": "binary",  # explícito (coincide con tus métricas/uso)
        "metric": "auc"
    })

    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, dtrain, num_boost_round=num_boost_round)
    models.append(model)

    preds.append(model.predict(X_test))  # prob de clase positiva

# Promedio de probabilidades (ensemblado)
proba_junio = np.mean(preds, axis=0)

# Salida final con IDs y score ensamblado
prediccion_202106 = id_test.copy()
prediccion_202106["prob_ensamble"] = proba_junio


# In[4]:


import pandas as pd

# models = lista de modelos entrenados en el bucle de semillas
all_importances = []

for m in models:
    imp = m.feature_importance(importance_type="gain")
    all_importances.append(imp)

# matriz (n_modelos x n_features)
importances_matrix = np.vstack(all_importances)

# promedio por feature
avg_importances = importances_matrix.mean(axis=0)

df_importancias = pd.DataFrame({
    "variable": features,      # las columnas que usaste
    "importancia": avg_importances
}).sort_values("importancia", ascending=False)

print(df_importancias.head(40))


# In[7]:


# armo el subset futuro (junio)
mask_future = dataset["foto_mes"].isin(PARAM["future"])
dfuture = dataset.loc[mask_future].copy()

# features definidos previamente
X_future = dfuture[features]

# predicciones del ensamble
preds = [m.predict(X_future.values) for m in models]
prediccion = np.mean(preds, axis=0)

# dataframe final con ids, mes y probabilidad
prediccion_df = dfuture[["numero_de_cliente", "foto_mes"]].copy()
prediccion_df["proba"] = prediccion

print(prediccion_df.head())


# In[10]:


# features de junio
X_future = dfuture[features]

# predicciones de cada modelo del ensamble
preds = [m.predict(X_future.values) for m in models]

# promedio de probabilidades
prediccion = np.mean(preds, axis=0)

# DataFrame con ids, mes y probabilidad de baja
tb_prediccion = dfuture[["numero_de_cliente", "foto_mes"]].copy()
tb_prediccion["prob"] = prediccion

# ordeno de mayor a menor probabilidad
tb_prediccion = tb_prediccion.sort_values("prob", ascending=False).reset_index(drop=True)

print(tb_prediccion.head())


# In[11]:


k_opt = 11820  # el que calculaste antes

# marco con 1 a los primeros k_opt clientes
tb_prediccion["Predicted"] = 0
tb_prediccion.loc[:k_opt-1, "Predicted"] = 1


# In[12]:


tb_prediccion[["numero_de_cliente", "Predicted"]].to_csv(
    r"C:\Users\jlopez04\Desktop\Facu\cuarto semestre\DMEyF\primera competencia\predicciones\prediccion_91.csv",
    index=False
)


# In[13]:


k_opt = 12820  # el que calculaste antes

# marco con 1 a los primeros k_opt clientes
tb_prediccion["Predicted"] = 0
tb_prediccion.loc[:k_opt-1, "Predicted"] = 1


# In[14]:


tb_prediccion[["numero_de_cliente", "Predicted"]].to_csv(
    r"C:\Users\jlopez04\Desktop\Facu\cuarto semestre\DMEyF\primera competencia\predicciones\prediccion_92.csv",
    index=False
)


# In[ ]:




