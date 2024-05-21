
# Dados Contínuos
### 1. **Teste t de Student**
   - **Aplicação**: Comparar as médias de dois grupos.
   - **Tipos**:
     - **Teste t para amostras independentes**: Quando os grupos são independentes (não relacionados).
     - **Teste t para amostras pareadas**: Quando os dados são pareados ou dependentes (por exemplo, antes e depois do tratamento no mesmo grupo de indivíduos).
     
```python
from scipy import stats

# Teste t para amostras independentes
t_statistic, p_value = stats.ttest_ind(grupo1, grupo2)

# Teste t para amostras pareadas
t_statistic, p_value = stats.ttest_rel(grupo1, grupo2)
```

### 2. **Análise de Variância (ANOVA)**
   - **Aplicação**: Comparar as médias de três ou mais grupos.
   - **Tipos**:
     - **ANOVA unidirecional (one-way)**: Um fator de agrupamento.
     - **ANOVA bidirecional (two-way)**: Dois fatores de agrupamento, podendo incluir interações.
```python
from scipy import stats
# One-way ANOVA
f_statistic, p_value = stats.f_oneway(grupo1, grupo2, grupo3)     
```    

### 3. **Teste de Mann-Whitney U (ou Wilcoxon rank-sum)**
   - **Aplicação**: Comparar duas amostras independentes quando a suposição de normalidade não é atendida.
   - **Alternativa ao**: Teste t para amostras independentes.

```python
from scipy import stats
u_statistic, p_value = stats.mannwhitneyu(grupo1, grupo2)
```

### 4. **Teste de Wilcoxon para amostras pareadas**
   - **Aplicação**: Comparar duas amostras pareadas quando a suposição de normalidade não é atendida.
   - **Alternativa ao**: Teste t para amostras pareadas.

```python
from scipy import stats
w_statistic, p_value = stats.wilcoxon(grupo1, grupo2)
```

### 5. **Teste de Kruskal-Wallis**
   - **Aplicação**: Comparar três ou mais grupos independentes quando a suposição de normalidade não é atendida.
   - **Alternativa ao**: ANOVA unidirecional.

```python
from scipy import stats
h_statistic, p_value = stats.kruskal(grupo1, grupo2, grupo3)
```

### 6. **Teste de Friedman**
   - **Aplicação**: Comparar três ou mais grupos dependentes (repetidas medidas) quando a suposição de normalidade não é atendida.
   - **Alternativa ao**: ANOVA de medidas repetidas.

```python
from scipy import stats
f_statistic, p_value = stats.friedmanchisquare(grupo1, grupo2, grupo3)
```

### 7. **Teste de Shapiro-Wilk**
   - **Aplicação**: Testar a normalidade dos dados.
   - **Importância**: Determinar se é adequado usar testes paramétricos (como o teste t e ANOVA) ou se deve optar por testes não paramétricos.

```python
from scipy import stats
stat, p_value = stats.shapiro(grupo1)
stat_2024, p_2024 = stats.shapiro(tempo_com_ia)
stat_2023, p_2023 = stats.shapiro(tempo_sem_ia)
```

### 8. **Teste de Kolmogorov-Smirnov**
   - **Aplicação**: Comparar a distribuição de uma amostra com uma distribuição teórica (como a normal) ou comparar duas amostras.
   - **Uso**: Verificação da normalidade ou homogeneidade de distribuições.

### Exemplos de aplicação:

- **Comparar a média do tempo gasto em dois anos diferentes (2024 vs. 2023)**: 
  - Se os dados forem normalmente distribuídos: **Teste t para amostras independentes**.
  - Se os dados não forem normalmente distribuídos: **Teste de Mann-Whitney U**.

- **Comparar a média do tempo gasto em três grupos diferentes (por exemplo, três anos consecutivos)**:
  - Se os dados forem normalmente distribuídos: **ANOVA unidirecional**.
  - Se os dados não forem normalmente distribuídos: **Teste de Kruskal-Wallis**.

- **Verificar se a distribuição do tempo gasto em um ano segue uma distribuição normal**: **Teste de Shapiro-Wilk**.

- **Comparar o tempo gasto antes e depois de um evento específico no mesmo grupo de indivíduos**:
  - Se os dados forem normalmente distribuídos: **Teste t para amostras pareadas**.
  - Se os dados não forem normalmente distribuídos: **Teste de Wilcoxon para amostras pareadas**.

## Dados Discretos

### 1. **Teste Qui-Quadrado de Independência**
   - **Aplicação**: Verificar se há associação entre duas variáveis categóricas.
   - **Uso**: Frequências observadas em uma tabela de contingência.

```python
from scipy import stats

# Exemplo de dados observados em uma tabela de contingência
observed = [[10, 20, 30], [6, 9, 17]]

# Teste Qui-Quadrado de Independência
chi2_statistic, p_value, dof, expected = stats.chi2_contingency(observed)
```

### 2. **Teste de McNemar**
   - **Aplicação**: Verificar mudanças em proporções para dados pareados.
   - **Uso**: Dados binários antes e depois de um tratamento no mesmo grupo de indivíduos.

```python
from statsmodels.stats.contingency_tables import mcnemar

# Exemplo de dados binários pareados
table = [[10, 5], [6, 9]]

# Teste de McNemar
result = mcnemar(table)
p_value = result.pvalue
```

## Dados Nominais

### 1. **Teste Qui-Quadrado de Bondade de Ajuste**
   - **Aplicação**: Verificar se a distribuição observada de uma variável categórica difere de uma distribuição teórica esperada.
   - **Uso**: Comparar frequências observadas com frequências esperadas.

```python
from scipy import stats

# Exemplo de dados observados e esperados
observed = [50, 30, 20]
expected = [40, 40, 20]

# Teste Qui-Quadrado de Bondade de Ajuste
chi2_statistic, p_value = stats.chisquare(observed, expected)
```

### 2. **Teste G de Independência**
   - **Aplicação**: Alternativa ao teste Qui-Quadrado de Independência, especialmente para amostras pequenas.
   - **Uso**: Tabelas de contingência para variáveis categóricas.

```python
import numpy as np
from scipy.stats import chi2

# Exemplo de dados observados em uma tabela de contingência
observed = np.array([[10, 20, 30], [6, 9, 17]])

# Calcular valores esperados
expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()

# Calcular G estatístico
G_statistic = 2 * np.sum(observed * np.log(observed / expected))

# Graus de liberdade
dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)

# Valor p
p_value = chi2.sf(G_statistic, dof)
```

## Dados Ordinais

### 1. **Teste de Mann-Whitney U (ou Wilcoxon rank-sum)**
   - **Aplicação**: Comparar duas amostras independentes ordinais.
   - **Alternativa ao**: Teste t para amostras independentes.

```python
from scipy import stats

# Teste Mann-Whitney U
u_statistic, p_value = stats.mannwhitneyu(grupo1, grupo2)
```

### 2. **Teste de Kruskal-Wallis**
   - **Aplicação**: Comparar três ou mais grupos independentes ordinais.
   - **Alternativa ao**: ANOVA unidirecional.

```python
from scipy import stats

# Teste de Kruskal-Wallis
h_statistic, p_value = stats.kruskal(grupo1, grupo2, grupo3)
```

### 3. **Teste de Wilcoxon para amostras pareadas**
   - **Aplicação**: Comparar duas amostras pareadas ordinais.
   - **Alternativa ao**: Teste t para amostras pareadas.

```python
from scipy import stats

# Teste de Wilcoxon para amostras pareadas
w_statistic, p_value = stats.wilcoxon(grupo1, grupo2)
```

### 4. **Teste de Friedman**
   - **Aplicação**: Comparar três ou mais grupos dependentes ordinais (repetidas medidas).
   - **Alternativa ao**: ANOVA de medidas repetidas.

```python
from scipy import stats

# Teste de Friedman
f_statistic, p_value = stats.friedmanchisquare(grupo1, grupo2, grupo3)
```

## Exemplos de aplicação:

- **Verificar se a distribuição de preferências por três marcas diferentes segue uma distribuição esperada**:
  - **Teste Qui-Quadrado de Bondade de Ajuste**.

- **Comparar a proporção de sucesso antes e depois de uma intervenção no mesmo grupo de indivíduos**:
  - **Teste de McNemar**.

- **Comparar a satisfação dos clientes em três diferentes lojas**:
  - Se os dados forem ordinais: **Teste de Kruskal-Wallis**.
  - Se os dados forem nominais: **Teste Qui-Quadrado de Independência**.

- **Verificar se há associação entre a categoria de produto e a decisão de compra**:
  - **Teste Qui-Quadrado de Independência**.
