
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Criando nosso próprio DataFrame com situações hipotéticas (mas nem tão hipotetica pq eu vou em todos)
data = {
    'Adversario': ['Clássico', 'Série C', 'Clássico', 'Série B', 'Série C', 'Série B', 'Clássico', 'Série C', 'Série B', 'Clássico', 'Série C', 'Série B', 'Clássico', 'Série C', 'Série B'],
    'Estadio': ['Casa', 'Fora', 'Casa', 'Fora', 'Casa', 'Casa', 'Fora', 'Fora', 'Casa', 'Casa', 'Fora', 'Casa', 'Fora', 'Casa', 'Fora'],
    'Clima': ['Bom', 'Chovendo', 'Bom', 'Bom', 'Chovendo', 'Nublado', 'Bom', 'Chovendo', 'Bom', 'Nublado', 'Bom', 'Chovendo', 'Nublado', 'Bom', 'Bom'],
    'Horario': ['Noite', 'Tarde', 'Noite', 'Tarde', 'Noite', 'Tarde', 'Tarde', 'Noite', 'Noite', 'Noite', 'Tarde', 'Noite', 'Tarde', 'Noite', 'Tarde'],
    'Companhia': ['Amigos', 'Sozinho', 'Amigos', 'Amigos', 'Sozinho', 'Amigos', 'Sozinho', 'Sozinho', 'Amigos', 'Amigos', 'Sozinho', 'Sozinho', 'Amigos', 'Amigos', 'Sozinho'],
    'VouOuNao': ['Sim', 'Não', 'Sim', 'Sim', 'Não', 'Sim', 'Não', 'Não', 'Sim', 'Sim', 'Não', 'Não', 'Sim', 'Sim', 'Não']
}
df = pd.DataFrame(data)

# =============================================================================
# 2. ANÁLISE DESCRITIVA E TRATAMENTO DOS DADOS (eu sofri feito um desgraçado tá)
# =============================================================================
print("--- Passo 2: Análise e Tratamento ---")
# Análise Visual: Ver o impacto de cada variável na decisão
for coluna in df.columns[:-1]:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=coluna, hue='VouOuNao', data=df, palette='viridis')
    plt.title(f'Impacto de "{coluna}" na Decisão de Ir ao Jogo')
    plt.show()

# Tratamento: Converter texto em números
X_categorico = df.drop('VouOuNao', axis=1)
y = df['VouOuNao']
encoder = OrdinalEncoder()
X = encoder.fit_transform(X_categorico)
feature_names = list(X_categorico.columns)
print("Dados transformados em formato numérico para o modelo.\n")

# =============================================================================
# 3. CRIAÇÃO E TREINAMENTO DO MODELO (prefiro treinar llm do que isso kkkk)
# =============================================================================
print("--- Passo 3: Criação do Modelo ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
modelo_arvore = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo_arvore.fit(X_train, y_train)
print("Modelo treinado com sucesso!\n")

# =============================================================================
# 4. EXTRAÇÃO DAS REGRAS EM FORMATO IF-THEN (que cão esse tal de if-then)
# =============================================================================
print("--- Passo 4: Regras da Árvore (IF-THEN) ---")
regras_texto = export_text(modelo_arvore, feature_names=feature_names)
print("Regras extraídas do modelo:")
print(regras_texto)

# Visualização gráfica da árvore
plt.figure(figsize=(20,10))
plot_tree(
    modelo_arvore,
    feature_names=feature_names,
    class_names=sorted(y.unique()),
    filled=True,
    rounded=True
)
plt.title("Visualização da Árvore de Decisão - Jogo do Náutico")
plt.show()

# =============================================================================
# 5. AVALIAÇÃO DO MODELO COM A BASE DE TESTE (eu amo IA, porém essa parte de teste me pega muito)
# =============================================================================
print("\n--- Passo 5: Acurácia do Modelo ---")
y_pred = modelo_arvore.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo na base de teste: {acuracia:.2%}")
print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_test, y_pred))