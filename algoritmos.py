#-------------------------------------------------------------------------------------------------
#
#                            1. Carregamento e Seleção de Colunas
#
#-------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset do arquivo 'dataZAP.csv'
# Usamos sep=';' pois o arquivo está separado por ponto e vírgula
# engine='python' e on_bad_lines='skip' para lidar com possíveis erros na leitura do arquivo
df = pd.read_csv('dataZAP.csv', sep=';', engine='python', on_bad_lines='skip')

# Definir a lista de colunas relevantes para a análise
colunas_relevantes = [
    'listing.address.city',
    'listing.address.state',
    'listing.usableAreas',
    'listing.bedrooms',
    'listing.bathrooms',
    'listing.suites',
    'listing.parkingSpaces',
    'listing.furnished',
    'listing.pool',
    'listing.gym',
    'listing.pricingInfo.rentalPrice',
    'listing.pricingInfo.monthlyCondoFee',
    'listing.pricingInfo.yearlyIptu'
]

# Criar um novo DataFrame (df_analise) selecionando apenas as colunas relevantes
df_analise = df[colunas_relevantes].copy()

#-------------------------------------------------------------------------------------------------
#
#                           2. Limpeza e Tratamento Inicial dos Dados
#
#-------------------------------------------------------------------------------------------------

# Renomear as colunas para nomes mais curtos e intuitivos
df_analise.columns = [
    'cidade', 'estado', 'area_util', 'quartos', 'banheiros', 'suites',
    'vagas_garagem', 'mobiliado', 'piscina', 'academia',
    'preco_aluguel', 'taxa_condominio', 'iptu_anual'
]

# Converter colunas numéricas para o tipo apropriado, tratando erros como NaN
for col in ['area_util', 'banheiros', 'suites', 'vagas_garagem', 'taxa_condominio', 'iptu_anual']:
    df_analise[col] = pd.to_numeric(df_analise[col], errors='coerce')

# Preencher valores ausentes (NaN) com 0 em colunas onde ausência pode significar zero (ex: comodidades ou custos não especificados)
for col in ['taxa_condominio', 'iptu_anual', 'suites']:
    df_analise[col] = df_analise[col].fillna(0)

# Remover linhas onde 'preco_aluguel' ou 'area_util' são nulos ou zero, pois são essenciais para a análise de preço
initial_rows = df_analise.shape[0]
df_analise.dropna(subset=['preco_aluguel', 'area_util'], inplace=True)
df_analise = df_analise[(df_analise['preco_aluguel'] > 0) & (df_analise['area_util'] > 0)]
rows_after_initial_cleaning = df_analise.shape[0]

print(f"Dimensão do DataFrame Original (após renomear colunas): ({initial_rows}, {df_analise.shape[1]})")
print(f"Dimensão do DataFrame após limpeza inicial (removendo nulos/zeros essenciais): ({rows_after_initial_cleaning}, {df_analise.shape[1]})")

#-------------------------------------------------------------------------------------------------
#
#                     3. Tratamento de Outliers e Filtragem por Faixa de Preço
#
#-------------------------------------------------------------------------------------------------

# Definir os limites para remoção de outliers na área útil (usando o percentil 99)
limite_area = df_analise['area_util'].quantile(0.99)

# Definir os limites mínimo e máximo para a faixa de preço de aluguel, conforme solicitado
limite_preco_min = 300
limite_preco_max = 1300

print(f"O limite mínimo de preço para filtragem é: R$ {limite_preco_min:,.2f}")
print(f"O limite máximo de preço para filtragem é: R$ {limite_preco_max:,.2f}")
print(f"O limite de área (percentil 99) para remoção de outliers é: {limite_area:,.2f} m²")

# Armazenar a dimensão antes da filtragem por preço e área
rows_before_filtering = df_analise.shape[0]

# Filtrar o DataFrame para remover outliers de área útil e aplicar a nova faixa de preço de aluguel
df_analise = df_analise[
    (df_analise['preco_aluguel'] >= limite_preco_min) &
    (df_analise['preco_aluguel'] <= limite_preco_max) &
    (df_analise['area_util'] < limite_area)
].copy()

# Armazenar a dimensão após a filtragem
rows_after_filtering = df_analise.shape[0]

# Calcular e imprimir a quantidade de linhas removidas pela filtragem
rows_removed_by_filtering = rows_before_filtering - rows_after_filtering

print(f"\nDimensão do DataFrame antes da filtragem por preço e área: ({rows_before_filtering}, {df_analise.shape[1]})")
print(f"Dimensão do DataFrame após a filtragem por preço e área: ({rows_after_filtering}, {df_analise.shape[1]})")
print(f"Número de linhas removidas pela filtragem de preço e área: {rows_removed_by_filtering}")

#-------------------------------------------------------------------------------------------------
#
#                                4. Análise Estatística Descritiva
#
#-------------------------------------------------------------------------------------------------

# Select the numerical columns of interest from the cleaned DataFrame
numerical_cols = ['area_util', 'quartos', 'banheiros', 'vagas_garagem', 'preco_aluguel', 'taxa_condominio', 'iptu_anual']

# Generate descriptive statistics for the selected columns
statistical_summary = df_analise[numerical_cols].describe()

# Display the generated statistical summary
display(statistical_summary.style.format('{:,.2f}'))

#-------------------------------------------------------------------------------------------------
#
#                                5. Análise de Correlação
#
#-------------------------------------------------------------------------------------------------

# Define the numerical columns for correlation analysis
numerical_cols_corr = ['area_util', 'quartos', 'banheiros', 'vagas_garagem', 'preco_aluguel', 'taxa_condominio', 'iptu_anual']

# Calculate the correlation matrix
correlation_matrix_cleaned = df_analise[numerical_cols_corr].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_cleaned, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação entre Variáveis Numéricas (Após Limpeza de Outliers)')
plt.show()

#-------------------------------------------------------------------------------------------------
#
#                                6. Visualizações Gráficas
#
#-------------------------------------------------------------------------------------------------

# 1. Histograma do preço do aluguel após limpeza e filtragem
plt.figure(figsize=(14, 6))
sns.histplot(df_analise['preco_aluguel'], bins=50, kde=True)
plt.title('Distribuição do Preço do Aluguel (Dados Limpos e Filtrados)')
plt.xlabel('Preço do Aluguel (R$)')
plt.ylabel('Frequência')
plt.show()

# 2. Gráfico de dispersão: Preço do Aluguel vs. Área Útil após limpeza e filtragem
plt.figure(figsize=(14, 6))
sns.scatterplot(data=df_analise, x='area_util', y='preco_aluguel', alpha=0.6)
plt.title('Preço do Aluguel vs. Área Útil (Dados Limpos e Filtrados)')
plt.xlabel('Área Útil (m²)')
plt.ylabel('Preço do Aluguel (R$)')
plt.show()

# 3. Boxplot: Preço do Aluguel por Número de Quartos após limpeza e filtragem
plt.figure(figsize=(14, 7))
sns.boxplot(data=df_analise, x='quartos', y='preco_aluguel')
plt.title('Distribuição do Preço de Aluguel por Número de Quartos (Dados Limpos e Filtrados)')
plt.xlabel('Número de Quartos')
plt.ylabel('Preço do Aluguel (R$)')
plt.show()

# 4. Identificar as 10 cidades com mais anúncios no DataFrame filtrado
top_10_cidades = df_analise['cidade'].value_counts().nlargest(10).index.tolist()
# Filtrar o DataFrame para incluir apenas as 10 principais cidades
df_top_cidades = df_analise[df_analise['cidade'].isin(top_10_cidades)].copy()

# 5. Gráfico de barras: Preço Médio por Cidade nas Top 10 (Dados Limpos e Filtrados)
plt.figure(figsize=(15, 8))

# Calcular o preço médio por cidade e ordenar do maior para o menor
mean_price_by_city = df_top_cidades.groupby('cidade')['preco_aluguel'].mean().sort_values(ascending=False)
order_cities = mean_price_by_city.index

sns.barplot(data=df_top_cidades, x='preco_aluguel', y='cidade', estimator=np.mean, order=order_cities)
plt.title('Preço Médio do Aluguel nas 10 Principais Cidades (Dados Limpos e Filtrados)')
plt.xlabel('Preço Médio do Aluguel (R$)')
plt.ylabel('Cidade')
plt.show()

#-------------------------------------------------------------------------------------------------
#
#                                7. Interpretação dos Resultados
#
#-------------------------------------------------------------------------------------------------

# Re-calcular e exibir a matriz de correlação para os dados limpos e filtrados
print("--- Matriz de Correlação (Dados Limpos e Filtrados R$ 300-1.300) ---")
numerical_cols_corr = ['area_util', 'quartos', 'banheiros', 'vagas_garagem', 'preco_aluguel', 'taxa_condominio', 'iptu_anual']
correlation_matrix_cleaned_filtered = df_analise[numerical_cols_corr].corr()
display(correlation_matrix_cleaned_filtered.style.format('{:,.2f}'))

print("\n--- Resumo Estatístico Descritivo (Dados Limpos e Filtrados R$ 300-1.300) ---")
# Re-calcular e exibir o resumo estatístico para os dados limpos e filtrados
numerical_cols = ['area_util', 'quartos', 'banheiros', 'vagas_garagem', 'preco_aluguel', 'taxa_condominio', 'iptu_anual']
resumo_estatistico_limpo_filtrado = df_analise[numerical_cols].describe()
display(resumo_estatistico_limpo_filtrado.style.format('{:,.2f}'))

print("\n--- Principais Observações das Visualizações (Dados Limpos e Filtrados R$ 300-1.300) ---")
print("1. Histograma do Preço do Aluguel: A distribuição do preço dentro da faixa de R$ 300 a R$ 1.300 mostra a concentração de imóveis nesse intervalo.")
print("2. Dispersão Preço vs. Área: O gráfico de dispersão ilustra a relação entre área útil e preço, permitindo ver se a tendência positiva se mantém clara nesse segmento.")
print("3. Boxplot por Quartos: O boxplot mostra como a distribuição dos preços varia com o número de quartos na faixa de R$ 300 a R$ 1.300.")
print("4. Preço Médio por Cidade: O gráfico de barras destaca as diferenças nos preços médios de aluguel entre as 10 cidades com mais anúncios, considerando apenas imóveis na faixa de preço selecionada.")

print("\n--- Conclusões da Análise Exploratória Atualizada ---")
print("Após a limpeza, remoção de outliers e filtragem pela faixa de preço de R$ 300 a R$ 1.300:")
print(f"- O número de imóveis na análise agora é: {df_analise.shape[0]}")
print("- A matriz de correlação fornece insights sobre as relações lineares entre as variáveis DENTRO desta faixa de preço específica.")
print("- As visualizações ajudam a confirmar e entender melhor as relações observadas na correlação e a distribuição dos dados neste segmento do mercado.")
print("- Podemos identificar quais atributos (como quartos e área útil, e a localização) parecem ser os mais influentes no preço do aluguel dentro deste intervalo mais realista.")
print("Esta análise refinada é um passo importante para a construção de um modelo de predição focado neste segmento de mercado.")

# O foco desta célula é na interpretação textual e na exibição de resumos estatísticos/correlação.

#-------------------------------------------------------------------------------------------------
#
#                9. Preparação e Pré-processamento dos dados para Random Forest
#
#-------------------------------------------------------------------------------------------------

import pandas as pd

# 1. Separar a variável alvo 'preco_aluguel'
y = df_analise['preco_aluguel']

# 2. Identificar as variáveis preditoras (features) excluindo a variável alvo
X = df_analise.drop('preco_aluguel', axis=1)

# 3. Identificar as colunas categóricas para One-Hot Encoding
categorical_cols = ['cidade', 'estado', 'mobiliado', 'piscina', 'academia']

# Converter 'mobiliado', 'piscina', 'academia' para tipo booleano e depois para int (0 ou 1)
# Isso garante que get_dummies as trate corretamente se houver valores não-booleanos ou nulos tratados como objetos
for col in ['mobiliado', 'piscina', 'academia']:
    if col in X.columns:
        X[col] = X[col].astype(bool).astype(int)

# 4. Aplicar One-Hot Encoding às colunas categóricas
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"Dimensões do dataset de features (X) após One-Hot Encoding: {X.shape}")
print(f"Dimensões do dataset da variável alvo (y): {y.shape}")
print("As primeiras 5 linhas do dataset de features (X) após One-Hot Encoding:")
display(X.head())

import pandas as pd
import numpy as np # Included for completeness of previous steps, though not explicitly used in the final part

# Reconstruct df_analise from scratch, incorporating all previous steps
# 1. Load the dataset (from cell dc2047e6)
df = pd.read_csv('dataZAP.csv', sep=';', engine='python', on_bad_lines='skip')

# 2. Define and select relevant columns (from cell dc2047e6)
colunas_relevantes = [
    'listing.address.city',
    'listing.address.state',
    'listing.usableAreas',
    'listing.bedrooms',
    'listing.bathrooms',
    'listing.suites',
    'listing.parkingSpaces',
    'listing.furnished',
    'listing.pool',
    'listing.gym',
    'listing.pricingInfo.rentalPrice',
    'listing.pricingInfo.monthlyCondoFee',
    'listing.pricingInfo.yearlyIptu'
]
df_analise = df[colunas_relevantes].copy()

# 3. Rename columns (from cell 2ac9efa1)
df_analise.columns = [
    'cidade', 'estado', 'area_util', 'quartos', 'banheiros', 'suites',
    'vagas_garagem', 'mobiliado', 'piscina', 'academia',
    'preco_aluguel', 'taxa_condominio', 'iptu_anual'
]

# 4. Convert numerical columns and fill NaNs (from cell 2ac9efa1)
for col in ['area_util', 'banheiros', 'suites', 'vagas_garagem', 'taxa_condominio', 'iptu_anual']:
    df_analise[col] = pd.to_numeric(df_analise[col], errors='coerce')

for col in ['taxa_condominio', 'iptu_anual', 'suites']:
    df_analise[col] = df_analise[col].fillna(0)

# 5. Remove rows with null or zero 'preco_aluguel' or 'area_util' (from cell 2ac9efa1)
df_analise.dropna(subset=['preco_aluguel', 'area_util'], inplace=True)
df_analise = df_analise[(df_analise['preco_aluguel'] > 0) & (df_analise['area_util'] > 0)]

# 6. Outlier treatment and price range filtering (from cell dcb41711)
limite_area = df_analise['area_util'].quantile(0.99)
limite_preco_min = 300
limite_preco_max = 1300

df_analise = df_analise[
    (df_analise['preco_aluguel'] >= limite_preco_min) &
    (df_analise['preco_aluguel'] <= limite_preco_max) &
    (df_analise['area_util'] < limite_area)
].copy()

# Now, proceed with the original code for preparing data for Random Forest:
# 1. Separar a variável alvo 'preco_aluguel'
y = df_analise['preco_aluguel']

# 2. Identificar as variáveis preditoras (features) excluindo a variável alvo
X = df_analise.drop('preco_aluguel', axis=1)

# 3. Identificar as colunas categóricas para One-Hot Encoding
categorical_cols = ['cidade', 'estado', 'mobiliado', 'piscina', 'academia']

# Converter 'mobiliado', 'piscina', 'academia' para tipo booleano e depois para int (0 ou 1)
# Isso garante que get_dummies as trate corretamente se houver valores não-booleanos ou nulos tratados como objetos
for col in ['mobiliado', 'piscina', 'academia']:
    if col in X.columns:
        X[col] = X[col].astype(bool).astype(int)

# 4. Aplicar One-Hot Encoding às colunas categóricas
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"Dimensões do dataset de features (X) após One-Hot Encoding: {X.shape}")
print(f"Dimensões do dataset da variável alvo (y): {y.shape}")
print("As primeiras 5 linhas do dataset de features (X) após One-Hot Encoding:")
display(X.head())

#-------------------------------------------------------------------------------------------------
#
#                              10. Divisão do Dataset: Treino e Teste
#
#-------------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

#-------------------------------------------------------------------------------------------------
#
#                             11. Inicialização e Treinamento do Modelo
#
#-------------------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestRegressor

# 1. Instanciar o modelo RandomForestRegressor
# Usamos random_state para garantir a reprodutibilidade dos resultados
model = RandomForestRegressor(random_state=42)

# 2. Treinar o modelo com os dados de treino
model.fit(X_train, y_train)

print("Modelo RandomForestRegressor treinado com sucesso!")

#-------------------------------------------------------------------------------------------------
#
#                             12. Avaliação de Performance do Modelo
#
#-------------------------------------------------------------------------------------------------

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Utilizar o modelo treinado para fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# 2. Calcular as métricas de avaliação
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 3. Imprimir os valores calculados
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

#-------------------------------------------------------------------------------------------------
#
#                          13. Otimização de Hiperparâmetros (Fine-Tuning)
#
#-------------------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# 1. Definir uma grade de parâmetros para o RandomizedSearchCV
param_dist = {
    'n_estimators': np.arange(100, 1000, 100), # Número de árvores no "forest"
    'max_depth': [None] + list(np.arange(10, 110, 10)), # Profundidade máxima da árvore
    'min_samples_split': np.arange(2, 20, 2), # Número mínimo de amostras necessárias para dividir um nó interno
    'min_samples_leaf': np.arange(1, 10, 1) # Número mínimo de amostras necessárias para estar em um nó folha
}

# 2. Criar uma instância do RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# 3. Inicializar o RandomizedSearchCV
# n_iter: número de configurações de parâmetros diferentes a serem amostradas
# cv: número de folds para validação cruzada
# scoring: métrica para avaliar o desempenho do modelo (R-squared)
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist,
                                   n_iter=50, cv=5, scoring='r2', random_state=42, n_jobs=-1, verbose=1)

# 4. Ajustar o RandomizedSearchCV aos dados de treino
print("Iniciando a busca aleatória de hiperparâmetros...")
random_search.fit(X_train, y_train)
print("Busca aleatória concluída.")

# 5. Exibir os melhores parâmetros encontrados e o melhor score
print(f"Melhores parâmetros encontrados: {random_search.best_params_}")
print(f"Melhor score R-squared (validação cruzada): {random_search.best_score_:.4f}")

#-------------------------------------------------------------------------------------------------
#
#                           14. Avaliação Final e Comparativo de Performance
#
#-------------------------------------------------------------------------------------------------

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Obter o melhor modelo encontrado pelo RandomizedSearchCV
best_model = random_search.best_estimator_

# 2. Fazer previsões no conjunto de teste com o modelo otimizado
y_pred_optimized = best_model.predict(X_test)

# 3. Calcular as métricas de avaliação para o modelo otimizado
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
mse_optimized = mean_squared_error(y_test, y_pred_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

# 4. Imprimir os valores calculados para o modelo otimizado
print("\n--- Performance do Modelo Random Forest Otimizado ---")
print(f"Mean Absolute Error (MAE) Otimizado: {mae_optimized:.2f}")
print(f"Mean Squared Error (MSE) Otimizado: {mse_optimized:.2f}")
print(f"R-squared (R2) Otimizado: {r2_optimized:.2f}")

# Comparar com o modelo inicial (opcional, para visualização)
print("\n--- Performance do Modelo Random Forest Inicial ---")
print(f"Mean Absolute Error (MAE) Inicial: {mae:.2f}")
print(f"Mean Squared Error (MSE) Inicial: {mse:.2f}")
print(f"R-squared (R2) Inicial: {r2:.2f}")

#-------------------------------------------------------------------------------------------------
#
#                          16. Implementação do Segundo Modelo de Regressão
#
#-------------------------------------------------------------------------------------------------

from sklearn.ensemble import GradientBoostingRegressor

# 1. Instantiate the GradientBoostingRegressor model
# Set random_state=42 for reproducibility
gbr_model = GradientBoostingRegressor(random_state=42)

print("GradientBoostingRegressor model instantiated.")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# Reconstruct df_analise from scratch, incorporating all previous steps
# 1. Load the dataset (from cell dc2047e6)
df = pd.read_csv('dataZAP.csv', sep=';', engine='python', on_bad_lines='skip')

# 2. Define and select relevant columns (from cell dc2047e6)
colunas_relevantes = [
    'listing.address.city',
    'listing.address.state',
    'listing.usableAreas',
    'listing.bedrooms',
    'listing.bathrooms',
    'listing.suites',
    'listing.parkingSpaces',
    'listing.furnished',
    'listing.pool',
    'listing.gym',
    'listing.pricingInfo.rentalPrice',
    'listing.pricingInfo.monthlyCondoFee',
    'listing.pricingInfo.yearlyIptu'
]
df_analise = df[colunas_relevantes].copy()

# 3. Rename columns (from cell 2ac9efa1)
df_analise.columns = [
    'cidade', 'estado', 'area_util', 'quartos', 'banheiros', 'suites',
    'vagas_garagem', 'mobiliado', 'piscina', 'academia',
    'preco_aluguel', 'taxa_condominio', 'iptu_anual'
]

# 4. Convert numerical columns and fill NaNs (from cell 2ac9efa1)
for col in ['area_util', 'banheiros', 'suites', 'vagas_garagem', 'taxa_condominio', 'iptu_anual']:
    df_analise[col] = pd.to_numeric(df_analise[col], errors='coerce')

for col in ['taxa_condominio', 'iptu_anual', 'suites']:
    df_analise[col] = df_analise[col].fillna(0)

# 5. Remove rows with null or zero 'preco_aluguel' or 'area_util' (from cell 2ac9efa1)
df_analise.dropna(subset=['preco_aluguel', 'area_util'], inplace=True)
df_analise = df_analise[(df_analise['preco_aluguel'] > 0) & (df_analise['area_util'] > 0)]

# 6. Outlier treatment and price range filtering (from cell dcb41711)
limite_area = df_analise['area_util'].quantile(0.99)
limite_preco_min = 300
limite_preco_max = 1300

df_analise = df_analise[
    (df_analise['preco_aluguel'] >= limite_preco_min) &
    (df_analise['preco_aluguel'] <= limite_preco_max) &
    (df_analise['area_util'] < limite_area)
].copy()

# Separar a variável alvo 'preco_aluguel'
y = df_analise['preco_aluguel']

# Identificar as variáveis preditoras (features) excluindo a variável alvo
X = df_analise.drop('preco_aluguel', axis=1)

# Identificar as colunas categóricas para One-Hot Encoding
categorical_cols = ['cidade', 'estado', 'mobiliado', 'piscina', 'academia']

# Converter 'mobiliado', 'piscina', 'academia' para tipo booleano e depois para int (0 ou 1)
for col in ['mobiliado', 'piscina', 'academia']:
    if col in X.columns:
        X[col] = X[col].astype(bool).astype(int)

# Aplicar One-Hot Encoding às colunas categóricas
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# --- FIX: Fill any remaining NaN values in X before splitting ---
# The 'banheiros' and 'vagas_garagem' columns (among others) might still contain NaNs
# after pd.to_numeric(..., errors='coerce') if no explicit fillna was applied.
X = X.fillna(0)

# Divide the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the GradientBoostingRegressor model
gbr_model = GradientBoostingRegressor(random_state=42)

print("Training GradientBoostingRegressor model...")
gbr_model.fit(X_train, y_train)
print("GradientBoostingRegressor model trained successfully!")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Use the trained GradientBoostingRegressor model to make predictions on the test set
y_pred_gbr = gbr_model.predict(X_test)

# 2. Calculate the evaluation metrics for the Gradient Boosting model
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
r2_gbr = r2_score(y_test, y_pred_gbr)

# 3. Print the calculated values for the Gradient Boosting model
print("--- Performance do Modelo Gradient Boosting Regressor ---")
print(f"Mean Absolute Error (MAE) GBR: {mae_gbr:.2f}")
print(f"Mean Squared Error (MSE) GBR: {mse_gbr:.2f}")
print(f"R-squared (R2) GBR: {r2_gbr:.2f}")

print("\n--- Comparativo de Performance dos Modelos ---")
print("| Métrica                    | Random Forest Inicial | Random Forest Otimizado | Gradient Boosting Regressor |")
print("| :------------------------- | :-------------------- | :---------------------- | :-------------------------- |")

# Retrieve initial Random Forest metrics (from previous output or kernel state)
mae_initial_rf = 80.05
mse_initial_rf = 13411.78
r2_initial_rf = 0.42

# Retrieve optimized Random Forest metrics (from previous output or kernel state)
mae_optimized_rf = 79.85
mse_optimized_rf = 13227.05
r2_optimized_rf = 0.43

# Gradient Boosting Regressor metrics (from previous output)
# mae_gbr, mse_gbr, r2_gbr are already defined in the kernel state

print(f"| R^2 (Explicabilidade)      | {r2_initial_rf:<21.2f} | {r2_optimized_rf:<23.2f} | {r2_gbr:<27.2f} |")
print(f"| MAE (Erro Médio Absoluto)  | R$ {mae_initial_rf:<18.2f} | R$ {mae_optimized_rf:<20.2f} | R$ {mae_gbr:<24.2f} |")
print(f"| MSE (Erro Quadrático Médio)| {mse_initial_rf:<21.2f} | {mse_optimized_rf:<23.2f} | {mse_gbr:<27.2f} |")

