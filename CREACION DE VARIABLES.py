#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# leo el dataset
dataset = pd.read_csv("C:/Users/jlopez04/Desktop/Facu/cuarto semestre/DMEyF/competencia_01_crudo.csv")
# calculo el periodo0 consecutivo
dsimple = dataset.assign(
    pos = range(1, len(dataset) + 1),
    periodo0 = (dataset["foto_mes"] // 100) * 12 + (dataset["foto_mes"] % 100)
)[["pos", "numero_de_cliente", "periodo0"]]

# ordeno
dsimple = dsimple.sort_values(["numero_de_cliente", "periodo0"]).reset_index(drop=True)

# calculo topes
periodo_ultimo = dsimple["periodo0"].max()
periodo_anteultimo = periodo_ultimo - 1

# calculo los leads de orden 1 y 2 (shift con signo negativo = lead en R)
dsimple["periodo1"] = dsimple.groupby("numero_de_cliente")["periodo0"].shift(-1)
dsimple["periodo2"] = dsimple.groupby("numero_de_cliente")["periodo0"].shift(-2)

# assign most common class values = "CONTINUA"
dsimple["clase_ternaria"] = None
dsimple.loc[dsimple["periodo0"] < periodo_anteultimo, "clase_ternaria"] = "CONTINUA"

# calculo BAJA+1
mask_baja1 = (
    (dsimple["periodo0"] < periodo_ultimo) &
    (dsimple["periodo1"].isna() | (dsimple["periodo0"] + 1 < dsimple["periodo1"]))
)
dsimple.loc[mask_baja1, "clase_ternaria"] = "BAJA+1"

# calculo BAJA+2
mask_baja2 = (
    (dsimple["periodo0"] < periodo_anteultimo) &
    (dsimple["periodo0"] + 1 == dsimple["periodo1"]) &
    (dsimple["periodo2"].isna() | (dsimple["periodo0"] + 2 < dsimple["periodo2"]))
)
dsimple.loc[mask_baja2, "clase_ternaria"] = "BAJA+2"

# pego el resultado en el dataset original
dsimple = dsimple.sort_values("pos").reset_index(drop=True)
dataset["clase_ternaria"] = dsimple["clase_ternaria"].values


# In[2]:


dataset = dataset.sort_values(["foto_mes", "clase_ternaria", "numero_de_cliente"])

# equivalente a .N en data.table (conteo de filas por grupo)
resumen = (
    dataset.groupby(["foto_mes", "clase_ternaria"])
           .size()              # cuenta las filas
           .reset_index(name="N")  # renombra la columna
)

print(resumen)


# ### CONVIERTO PRINCIPALES VARIABLES A DOLAR

# In[3]:


import pandas as pd

# Cotización mensual del dólar
usd = pd.DataFrame({
    "foto_mes": [202101, 202102, 202103, 202104, 202105, 202106],
    "usd_ars": [86.25, 88.75, 91.50, 92.50, 93.75, 94.75]
})

# Merge con el dataset original
dataset = dataset.merge(usd, on="foto_mes", how="left")

# Variables que querés convertir de pesos a dólares
vars_a_convertir = [
    "mcaja_ahorro",
    "mprestamos_personales",
    "mcuentas_saldo",
    "mcaja_ahorro",
    "mcaja_ahorro_adicional",
    "mcuenta_corriente",
    "mcuenta_corriente_adicional",
    "mplazo_fijo_pesos",
    "minversion1_pesos",
    "Master_msaldopesos",
    "Master_mconsumospesos",
    "Master_madelantopesos",
    "Master_mpagospesos",
    "Visa_msaldopesos",
    "Visa_mconsumospesos",
    "Visa_madelantopesos",
    "Visa_mpagospesos",
    "Visa_msaldototal",
    "mtarjeta_visa_consumo",
    "ccomisiones_otras",
    "Visa_mpagominimo",
    "mtarjeta_master_consumo",
    "mrentabilidad_annual",
    "mcuenta_corriente",
    
]

# Reemplazar valores en pesos por dólares
for col in vars_a_convertir:
    if col in dataset.columns:
        dataset[col] = dataset[col] / dataset["usd_ars"]

# Eliminar la columna auxiliar de tipo de cambio
dataset = dataset.drop(columns=["usd_ars"])


# ### CREACION DE VARIABLES

# In[4]:


dataset.head()


# In[7]:


import pandas as pd

# Ordenamos por cliente y mes
dataset = dataset.sort_values(['numero_de_cliente', 'foto_mes'])

# === Engagement ===

# frecuencia mensual de transacciones
dataset['ctrx_mensual'] = dataset['ctrx_quarter'] / 3

# % de transacciones digitales (si está la columna)
if 'chomebanking_transacciones' in dataset.columns:
    dataset['pct_digital'] = dataset['chomebanking_transacciones'] / (dataset['ctrx_mensual'] + 1)

# dummy: ¿usa payroll?
if 'mpayroll' in dataset.columns:
    dataset['usa_payroll'] = (dataset['mpayroll'] > 0).astype(int)

# cantidad de productos básicos activos
productos = [
    'mprestamos_personales',
    'mcaja_ahorro',
    'mcuenta_corriente',
    'mtarjeta_visa_consumo',
    'mtarjeta_master_consumo'
]
productos_presentes = [p for p in productos if p in dataset.columns]
dataset['n_productos_activos'] = dataset[productos_presentes].gt(0).sum(axis=1)

# === Dispersión ===
vars_montos = [
    'mcuentas_saldo',
    'mcaja_ahorro',
    'mtarjeta_visa_consumo'
]
vars_presentes = [v for v in vars_montos if v in dataset.columns]

for v in vars_presentes:
    dataset[f'{v}_std_3m'] = dataset.groupby('numero_de_cliente')[v].transform(lambda x: x.rolling(3).std())
    dataset[f'{v}_cv_3m'] = dataset.groupby('numero_de_cliente')[v].transform(lambda x: x.rolling(3).std() / (x.rolling(3).mean() + 1))

# deltas mes a mes
dataset['delta_ctrx'] = dataset.groupby('numero_de_cliente')['ctrx_mensual'].diff()
if 'mcuentas_saldo' in dataset.columns:
    dataset['delta_saldo'] = dataset.groupby('numero_de_cliente')['mcuentas_saldo'].diff()

# === Flag de inactividad ===
dataset['inactivo'] = (dataset['ctrx_mensual'] == 0).astype(int)


# In[8]:


dataset.head()


# In[9]:


import pandas as pd
import numpy as np

def aguinaldo(df,
    lookback_months=12,
    dominance_threshold=0.70,
    min_onus_months=2,
    min_onus_trx=2,
    out_col="cobra_aguinaldo_en_banco_flag"
):
    req = ["numero_de_cliente","foto_mes","mpayroll","cpayroll_trx","mpayroll2","cpayroll2_trx"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    out = df.copy()
    out["foto_mes"] = out["foto_mes"].astype(str).str[:6].astype(int)
    for c in ["mpayroll","cpayroll_trx","mpayroll2","cpayroll2_trx"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out.sort_values(["numero_de_cliente","foto_mes"], inplace=True)
    out["onus_any"] = ((out["mpayroll"] > 0) | (out["cpayroll_trx"] > 0)).astype(int)

    # Cálculos vectorizados
    grp = out.groupby("numero_de_cliente", group_keys=False)

    out["hist_mpayroll"]      = grp["mpayroll"].cumsum().shift(1)
    out["hist_mpayroll2"]     = grp["mpayroll2"].cumsum().shift(1)
    out["hist_cpayroll_trx"]  = grp["cpayroll_trx"].cumsum().shift(1)
    out["hist_onus_any"]      = grp["onus_any"].cumsum().shift(1)

    # Rolling window (solo estas dos necesitan loops, pero se puede usar transform rolling)
    out["hist_onus_months_roll"] = (
        grp["onus_any"]
        .rolling(window=lookback_months, min_periods=1)
        .sum()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    out["hist_cpayroll_trx_roll"] = (
        grp["cpayroll_trx"]
        .rolling(window=lookback_months, min_periods=1)
        .sum()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # Dominancia y condiciones
    denom = (out["hist_mpayroll"] + out["hist_mpayroll2"]).replace(0, np.nan)
    out["hist_onus_share"] = (out["hist_mpayroll"] / denom).fillna(0.0)

    cond_share  = out["hist_onus_share"] >= dominance_threshold
    cond_months = out["hist_onus_months_roll"] >= float(min_onus_months)
    cond_trx    = out["hist_cpayroll_trx_roll"] >= float(min_onus_trx)

    out[out_col] = np.where(cond_share & cond_months & cond_trx, 1, 0).astype(np.int8)

    return out


# In[10]:


# Generar el flag de aguinaldo para cada cliente/mes
dataset = aguinaldo(dataset)

# Verificar las primeras filas
print(dataset[['numero_de_cliente','foto_mes','cobra_aguinaldo_en_banco_flag']].head())


# In[7]:


dataset.head()


# In[9]:


dataset.info()


# In[11]:


excluir = {"numero_de_cliente", "foto_mes", "clase_ternaria","hist_mpayroll","hist_mpayroll2","hist_cpayroll_trx",
          "hist_onus_any","hist_onus_months_roll","hist_cpayroll_trx_roll","hist_onus_share","cobra_aguinaldo_en_banco_flag"}
cols_to_lag = [c for c in dataset.columns if c not in excluir]

# ordenar por cliente y mes
dataset = dataset.sort_values(["numero_de_cliente", "foto_mes"])

# crear lag1 y delta1 para cada columna
for c in cols_to_lag:
    dataset[f"{c}_lag1"] = dataset.groupby("numero_de_cliente")[c].shift(1)
    dataset[f"{c}_delta1"] = dataset[c] - dataset[f"{c}_lag1"]


# In[12]:


dataset.head()


# In[13]:


# ahora sí podés exportar
dataset.to_csv(
    r"C:\Users\jlopez04\Desktop\Facu\cuarto semestre\DMEyF\primera competencia\predicciones\nuevabase.csv",
    index=False
)


# In[ ]:




