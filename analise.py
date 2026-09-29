"""
**Explicação das Colunas:**

*   `ad_id`: Identificador único para cada anúncio.
*   `xyz_campaign_id`: Identificador da campanha principal à qual o anúncio pertence. Pode ser um ID interno da empresa XYZ.
*   `fb_campaign_id`: Identificador da campanha no Facebook.
*   `age`: Faixa etária do público-alvo do anúncio (ex: '30-34', '35-39').
*   `gender`: Gênero do público-alvo (M para Masculino, F para Feminino).
*   `interest`: Código numérico que representa a categoria de interesse do público-alvo.
*   `Impressions`: O número de vezes que o anúncio foi exibido.
*   `Clicks`: O número de cliques no anúncio.
*   `Spent`: O valor gasto na campanha para o anúncio específico.
*   `Total_Conversion`: O número total de conversões atribuídas ao anúncio (incluindo aprovações e outras).
*   `Approved_Conversion`: O número de conversões que foram aprovadas, indicando uma ação mais valiosa (como uma compra finalizada).
*   `CTR`: (Click-Through Rate) A taxa de cliques do anúncio, calculada como `Clicks / Impressions`.
*   `CPC`: (Cost Per Click) O custo por clique, calculado como `Spent / Clicks`.
*   
"""

#%% Importações

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
    make_scorer,
)

# Carregamento dos dados
url = "https://raw.githubusercontent.com/mGalarnyk/Python_Tutorials/refs/heads/master/Kaggle/Facebook/KAG_conversion_data.csv"
df = pd.read_csv(url)

print("Dimensões do dataset:", df.shape)
display(df.head())


# Engenharia de Atributos (Feature Engineering)
# Criando CTR (Click-Through Rate) e CPC (Cost per Click)
# Adicionamos um pequeno valor (1e-6) no denominador para evitar divisão por zero se houver impressões/cliques zerados
df["CTR"] = df["Clicks"] / (df["Impressions"] + 1e-6)
df["CPC"] = df["Spent"] / (df["Clicks"] + 1e-6)
df["CPA"] = df["Spent"] / (df["Approved_Conversion"]+ 1e-6)
# Taxa de conversão de cliques aprovados
df["Converted"] = (df["Approved_Conversion"] > 0).astype(int)
df["Conv_Rate"] = np.where(df['Clicks'] > 0, df['Approved_Conversion'] / df['Clicks'], 0)
df['Status_Conversao'] = np.where(df['Converted'] == 1, 'Convertido', 'Não Convertido')

# Gráfico de caixa (boxplot) para comparar os gastos por status de conversão
plt.figure(figsize=(8, 6))
sns.boxplot(x='Status_Conversao', y='Spent', data=df, palette='muted')
plt.title('Boxplot de Gastos (Spent) por Status de Conversão')
plt.xlabel('Status de Conversão')
plt.ylabel('Valor Gasto (Spent)')
plt.show()

# Matriz de correlação entre métricas principais
correlation_matrix = df[['CTR', 'CPC', 'CPA', 'Converted']].corr()
display(correlation_matrix)

# Mapa de calor da correlação
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlação entre CTR, CPC, CPA e Conversão')
plt.show()


y = df['Converted']

# Remoção de colunas que podem causar fuga de dados (Data Leakage) ou identificadores irrelevantes
cols_to_drop = [
    'Converted', 'Status_Conversao', 'ad_id', 'fb_campaign_id', 'xyz_campaign_id',
    'Approved_Conversion', 'Total_Conversion', 'Conv_Rate', 'CPA'
]
X = df.drop(columns=cols_to_drop)

# Transformação de variáveis categóricas através de One-Hot Encoding
categorical_cols = ['age', 'gender']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(X[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

X = X.drop(columns=categorical_cols)
X = pd.concat([X.reset_index(drop=True), encoded_df], axis=1)

# Divisão estratificada em conjuntos de treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Treinamento de uma Árvore de Decisão
tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)

# Treinamento de um modelo Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Treinamento de um modelo XGBoost
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

from sklearn.metrics import precision_recall_curve
import shap

# Cálculo do limiar ideal baseado na curva Precision-Recall
precisions, recalls, thresholds = precision_recall_curve(y_test_fe, y_pred_proba_fe)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
best_threshold = thresholds[np.argmax(f1_scores)]

# Explicação do modelo utilizando valores SHAP
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test_fe)


import joblib

# Dicionário com os componentes essenciais para inferência em produção
model_artefact = {
    'model': best_rf,
    'encoder': encoder,
    'cols_to_drop': cols_to_drop,
    'categorical_cols': categorical_cols,
    'best_threshold': best_threshold,
    'feature_names': list(X_fe.columns)
}

# Salvamento do modelo num ficheiro comprimido .pkl
joblib.dump(model_artefact, 'modelo_conversao_rf.pkl')
print("✅ Modelo guardado com sucesso no ficheiro 'modelo_conversao_rf.pkl'!")






