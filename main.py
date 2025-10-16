import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def analisarNulos(df):
    arquivo2 = open("nulos.csv", "w")

    valores = df.isnull().sum().sort_values(ascending=False)
    for i in range(len(valores)): 
        nome_coluna = valores.index[i]
        total_nulos = valores.iloc[i]
        arquivo2.write(f"{nome_coluna},{total_nulos}\n")
    arquivo2.close()

def analisarCorrelacao(X, Y):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=X, y=Y)
    plt.xlabel(X.name if hasattr(X, 'name') else 'X')
    plt.ylabel(Y.name if hasattr(Y, 'name') else 'Y')
    plt.title(f'Dispersão: {X.name} x {Y.name}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

df = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")
df_teste = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")

#Aplicando oneHotEncode
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
colunas_categoricas = ["MSZoning", "Street", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "KitchenQual", 
                       "SaleType", "SaleCondition", "BsmtCond", "GarageType", "PavedDrive", "GarageCond", "GarageQual", "PoolQC", "Functional", "Electrical", "Condition1", "Condition2"] 

#treino
valores_codificados = encoder.fit_transform(df[colunas_categoricas])
#teste
valores_codificados_teste = encoder.transform(df_teste[colunas_categoricas])

colunas_novas = encoder.get_feature_names_out(colunas_categoricas)

#treino
df_codificado = pd.DataFrame(valores_codificados, columns=colunas_novas)
#teste
df_codificado_teste = pd.DataFrame(valores_codificados_teste, columns=colunas_novas)

#para pegar qual ultima coluna do df original
df_codificado.index = df.index

#para pegar qual ultima coluna do df original de teste
df_codificado_teste.index = df_teste.index

# Junta com o original
df_completo = pd.concat([df, df_codificado], axis=1)

df_completo_teste = pd.concat([df_teste, df_codificado_teste], axis=1)

#normalizador
scaler = StandardScaler()

#Lida com as colunas do arquivo de treino
df_completo["AnosTotais"] = (df_completo["YearBuilt"] + df_completo["YearRemodAdd"]) ** 1.5
df_completo["TotalBanheirosPorao"] = df_completo["BsmtFullBath"] + df_completo["BsmtHalfBath"]
df_completo["TotalBanheiros"] = (df_completo["FullBath"] + df_completo["HalfBath"]) ** 1.5
df_completo["LotFrontageNew"] = df_completo["LotFrontage"].fillna(0)
df_completo["TamanhoPoraoAcabado"] = df_completo["TotalBsmtSF"] + df_completo["BsmtUnfSF"]
df_completo["TamanhoAndaresSomado"] = df_completo["1stFlrSF"] + df_completo["2ndFlrSF"] - df_completo["LowQualFinSF"]
df_completo["TamanhoAndaresSomado"] = df_completo["TamanhoAndaresSomado"] * df_completo["TamanhoAndaresSomado"]
df_completo["QualidadeTotal"] = (df_completo["OverallQual"]) ** 2
df_completo["CondicaoTotal"] = (df_completo["OverallCond"])
df_completo["AreaLivre"] = df_completo["GrLivArea"] ** 1.5
df_completo["AnoMesVenda"] = (df_completo["YrSold"] + df_completo["MoSold"]) ** 1.5

colunas_treino = ["MSSubClass", "LotArea", "LotFrontageNew", "QualidadeTotal", "CondicaoTotal", "MasVnrArea", "TotalBanheirosPorao", 
               "TotalBanheiros", "TamanhoPoraoAcabado", "GarageArea", "GarageCars", "TamanhoAndaresSomado", "AreaLivre", "TotRmsAbvGrd", "KitchenAbvGr", "MiscVal", "AnosTotais", "AnoMesVenda",] + list(colunas_novas)

X_treino = df_completo[colunas_treino].fillna(0)
Y_treino = df_completo[["SalePrice"]]
X_treino_normalizado = scaler.fit_transform(X_treino)

#Lida com as colunas do arquivo de teste
df_completo_teste["AnosTotais"] = (df_completo_teste["YearBuilt"] + df_completo_teste["YearRemodAdd"]) ** 1.5
df_completo_teste["TotalBanheirosPorao"] = df_completo_teste["BsmtFullBath"] + df_completo_teste["BsmtHalfBath"]
df_completo_teste["TotalBanheiros"] = (df_completo_teste["FullBath"] + df_completo_teste["HalfBath"]) ** 1.5
df_completo_teste["LotFrontageNew"] = df_completo_teste["LotFrontage"].fillna(0)
df_completo_teste["TamanhoPoraoAcabado"] = df_completo_teste["TotalBsmtSF"] + df_completo_teste["BsmtUnfSF"]
df_completo_teste["TamanhoAndaresSomado"] = df_completo_teste["1stFlrSF"] + df_completo_teste["2ndFlrSF"] - df_completo_teste["LowQualFinSF"]
df_completo_teste["TamanhoAndaresSomado"] = df_completo_teste["TamanhoAndaresSomado"] * df_completo_teste["TamanhoAndaresSomado"]
df_completo_teste["QualidadeTotal"] = (df_completo_teste["OverallQual"])**2
df_completo_teste["AreaLivre"] = df_completo_teste["GrLivArea"] ** 1.5
df_completo_teste["CondicaoTotal"] = (df_completo_teste["OverallCond"])
df_completo_teste["AnoMesVenda"] = (df_completo_teste["YrSold"] + df_completo_teste["MoSold"]) ** 1.5

colunas_teste = ["MSSubClass", "LotArea", "LotFrontageNew", "QualidadeTotal", "CondicaoTotal",  "MasVnrArea", "TotalBanheirosPorao", 
               "TotalBanheiros", "TamanhoPoraoAcabado", "GarageArea", "GarageCars", "TamanhoAndaresSomado", "AreaLivre", "TotRmsAbvGrd", "KitchenAbvGr", "MiscVal", "AnosTotais", "AnoMesVenda",] + list(colunas_novas)
X_teste =  df_completo_teste[colunas_teste].fillna(0)
X_teste_normalizado = scaler.transform(X_teste)


#Treina o modelo
model = LinearRegression()
model.fit(X_treino_normalizado, Y_treino)

#Prediz as saídas
y = model.predict(X_teste_normalizado)
arquivo = open("submissao.csv", "w")

X_teste_id = df_teste["Id"]


#cria e escreve o arquivo submissao
arquivo.write("Id,SalePrice\n")
for i in range(len(y)): 
    arquivo.write(f"{X_teste_id.iloc[i]},{y[i][0]}\n")

arquivo.close()
