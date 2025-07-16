# %%

import pandas as pd
# %%

df = pd.read_excel("dados/dados_cerveja.xlsx")
df.head()
# %%
#Prever qual a minha cerveja 
#Feature são as carcaterísticas
#Target é a variável resposta 
features = ["temperatura","copo","espuma","cor"]
target='classe'

X = df[features]
y = df[target]
# %%
#Transformando as variaveis categoricas, pois o modelo sklearn só acieta variaveis numericas

X = X.replace(
    {
        "mud":1, "pint":2,
        "sim":1,"não":0,
        "clara":0,"escura":1,
    }
)
X
# %%
#Criar o modelo

from sklearn import tree
#Defino o objeto de modelo
#Ajuste do modelo
model = tree.DecisionTreeClassifier()
model.fit(X=X, y = y)

# %%
#Mostrar o modelo
import matplotlib.pyplot as plt

plt.figure(dpi=400)
tree.plot_tree(model, feature_names=features,
               class_names=model.classes_,
               filled=True)