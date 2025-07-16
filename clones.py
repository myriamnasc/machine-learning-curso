# %% 

import pandas as pd

# %%

df = pd.read_parquet("dados/dados_clones.parquet")
df.head()
# %%
#Case: entender pq ta perdendo batalhas
features = ["Massa(em kilos)","Estatura(cm)"]
target = 'Status '

X = df[features]
y = df[target]
# %%
from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X=X, y = y)
# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400)
tree.plot_tree(model, max_depth=3,feature_names=features,
               class_names=model.classes_,
               filled=True)
# %%
