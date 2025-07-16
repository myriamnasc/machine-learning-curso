# %%

import pandas as pd
from sklearn import linear_model
from sklearn import tree
import matplotlib.pyplot as plt
# %%

df = pd.read_excel("dados/dados_cerveja_nota.xlsx")
df.head()
# %%

#Definição das variaveis

X = df[['cerveja']] #isso é uma matriz(dataframe)
y= df['nota'] #isso é um vetor(series)

#Criação e treino do modelo
reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(X,y)

#Coeficientes da reta
a,b = reg.intercept_, reg.coef_[0]

#@Predição com dados distintos. Evita pontos repetidos em X para deixar a linha reta mais "limpa" no gráfico
predict_reg= reg.predict(X.drop_duplicates())

#Plotando o gráfico
plt.plot(X['cerveja'],y,'o')
plt.grid(True)
plt.title("Relação Cerveja vs Nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

plt.plot(X.drop_duplicates()['cerveja'],predict_reg)

plt.legend(['Observado', f'y={a:.3f} + {b:.3f} x'])
# %%
#Arvore de decisão
arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X,y)

arvore_d2 = tree.DecisionTreeRegressor(random_state=42,max_depth=2)
arvore_d2.fit(X,y)
#Predicao
predict_arvore_full = arvore_full.predict(X.drop_duplicates())
predict_arvore_d2= arvore_d2.predict(X.drop_duplicates())

#Plotando todos os modelos juntos
plt.plot(X['cerveja'],y,'o')
plt.grid(True)
plt.title("Relação Cerveja vs Nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

plt.plot(X.drop_duplicates()['cerveja'],predict_reg)
plt.plot(X.drop_duplicates()['cerveja'],predict_arvore_full)
plt.plot(X.drop_duplicates()['cerveja'],predict_arvore_d2)

plt.legend(['Observado', f'y={a:.3f} + {b:.3f} x',
            'Árvore full',
            'Árvore Depth = 2',
            ])
# %%
#Visualizando  árvore com profundidade 2
plt.figure(dpi=400)

tree.plot_tree(arvore_d2,
               feature_names=['cerveja'],
               filled=True)

