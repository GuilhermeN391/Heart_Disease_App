# Modelo de Previsão de Doença Cardíaca

Este reposiório hospeda um modelo de aprendizado de máquina desenvolvido para detecção de pré-disposição ou existência de doença cardíaca no paciente, implementado usando PyTorch.

## Informações do projeto

- **Disciplina**: PPGEEC2318 - APRENDIZADO DE MÁQUINA
- **Turma**: T01 (2025.1)
- **Discente**: Guilherme Nascimento da Silva
- **Matrícula**: 20251011640
- **Docente**: Prof. Dr. Ivanovitch Medeiros Dantas da Silva ([GitHub](https://github.com/ivanovitchm))
- **Dataset**: [Heart Disease Prediction](https://www.kaggle.com/datasets/mfarhaannazirkhan/heart-dataset/data)
- **Inspirado em**: 
  - [Heart Disease Prediction](https://github.com/maxim-eyengue/Heart-Disease-App)
  - [PPGEEC2318 - Week05c](https://github.com/ivanovitchm/PPGEEC2318/blob/main/lessons/week05/week05c.ipynb)
  - [ppgeecmachinelearning - Week02](https://github.com/ivanovitchm/ppgeecmachinelearning/tree/main/lessons/week_02/sources)

## Model Card

Para apresentar as informações importantes do modelo, esse Model Card foi desenvolvido seguindo os princípios do artigo [Model Cards for Model Reporting](https://arxiv.org/pdf/1810.03993) para documentação de projetos envolvendo *Machine learning*.

### Detalhes do Modelo

- **Desenvolvido por**: Guilherme Nascimento da Silva
- **Data de desenvolvimento**: 05/2025
- **Tipo de modelo**: Regressão Logística Binária implementada com PyTorch
- **Versão**: 1.0
- **Framework**: PyTorch
- **Número de features**: 20 (features originais + features derivadas)
- **Dispositivo utilizado**: CPU
- **Hyperparâmetros**:
  - Batch size: 64
  - Learning rate: 0,01
  - Número de épocas: 40

### Uso Pretendido

- **Uso primário**: Detectar pacientes propensos a possuírem doenças cardíadas;
- **Usuários pretendidos**: Pacientes em geral, com a presença ou não de comorbidades;
- **Casos fora do escopo**: Não deve ser usado como único mecanismo para indicação de doenças cardíacas.

### Fatores

- **Fatores relevantes**: Buscando indicar a propensão a doença, fatores como idade, nível de colesterol, presença de angina (dor no peito), pressão arterial, frequência cardiaca, glicemina em jejum;
- **Fatores de avaliação**: O modelo foi avaliado baseado nas métricas aplicadas ao threshold de 0,5.

### Métricas

As métricas de avaliação foram escolhidas considerando o forte desbalanceamento dos dados:

- **Acurácia**: Proporção de predições corretas;
- **Precisão**: Proporção de verdadeiros positivos entre os casos classificados como positivos;
- **Recall**: Proporção de fraudes corretamente identificadas;
- **Especificidade**: Proporção de transações legítimas corretamente identificadas;
- **F1-Score**: Média harmônica entre precisão e recall;
- **AUC-ROC**: Área sob a curva ROC.

### Informações do Dataset

- O Dataset completo consiste na combinação de cinco datasets públicos com informações relacionadas a doenças cardíacas, com um total de 2181 catalogações:
  <ul>
      <li> 📝 Heart Attack Analysis & Prediction Dataset: 304 records from Rahman, 2021</li>
      <li> 📝 Heart Disease Dataset: 1,026 records from Lapp, 2019</li>
      <li> 📝 Heart Attack Prediction (Dataset 3): 295 records from Damarla, 2020</li>
      <li> 📝 Heart Attack Prediction (Dataset 4): 271 records from Anand, 2018</li>
      <li> 📝 Heart CSV Dataset: 290 records from Nandal, 2022</li>
  </ul>
- Sua versão base conta com 13 features que irão se converter nas variáveis para o desenvolvimento do modelo:
  - **age**: idade do paciente [em anos, valor numérico];
  - **sex**: gênero do paciente [1: Masculino, 0: Feminino];
  - **cp**: Tipo de dor no peito [0: Angina típica, 1: Angina atípica, 2: Dor não anginosa, 3: Assintomático];
  - **trestbps**: Pressão arterial em repouso [mm Hg, valor numérico];
  - **chol**: Colesterol total no sangue do paciente [mg/dl, valor numérico];
  - **fbs**: índice de glicemia [1: caso o açucar no sangue > 120 mg/dl, 0: caso contrário];
  - **restecg**: Resultados do eletrocardiograma do paciente [0: Normal, 1: Anormalidade da onda ST-T (inversões da onda T e/ou elevação ou decaimento do segmento ST > 0,05 mV), 2: Apresentando uma provável ou definitiva hipertrofia ventricular esquerda pelos critérios de Estes];
  - **thalachh**: Frequência cardíaca máxima medida [Valor entre 60 and 202];
  - **exang**: Presença de angina recorrente de exercícios físicos [1: Sim, 0: Não];
  - **oldpeak**: Decaímento do segmento ST induzido por exercício físico em relação ao repouso [Decaímento relatado através de um valor numérico];
  - **slope**: Variação do segmento ST em  pico do exercício [0: Elevação, 1: constante, 2: Decaímento ];
  - **ca**: Número de vasos principais (artérias, veias e capilares) coloridos por fluoroscopia: [0, 1, 2, 3];
  - **thal**: Tipo de Talessemia (condição genética de defeito na Síntese da Hemoglobina); [1: Normal, 2: Defeito corrigido, 3: Defeito reversível];
  - **target**: variável de indicação do risco de doenças cardíadcas [1: Doente ou maior propensão a doenças cardíacas, 0: Normal ou chances baixas de doenças cardíacas].

Após o tratamento de dados, com os passos indicados pelo [Heart Disease Prediction](https://github.com/maxim-eyengue/Heart-Disease-App), o Dataset total para treinamento e teste do modelo consiste em **519** amostras que serão divididas em **80%** para treino e **20%** para teste, com valores totais para cada grupo apresentados em seguida. Valor o qual foi decorrente da presença alta de catalogações de pacientes repedidas, além da presença de dados inconsistentes indicado através do tratamento feito que foram desconsiderados para a implementação do modelo.

### Resumo do Dataset original

| Aspecto | Informação |
|---------|------------|
| Total de cadastros | 2.181 |
| Features | 13  |
| Classe alvo | "target"  |
| Valores nulos | 1.231 (4.0%)|
| Classe normal (0) | 1.099 (50,39%) |
| Classe fraude (1) | 1.082 (49,61%) |

### Dados de Avaliação

- **Conjunto de dados**: Conjunto de teste (20% do dataset original);
- **Tamanho**: 104 instâncias;
- **Pré-processamento**: Remoção de outliers usando pandas.DataFrame.fillna e normalização com StandardScaler.

### Dados de Treinamento

- **Conjunto de dados**: Arquivos CSV contendo as informações dos pacientes;
- **Tamanho**: 415 instâncias após remoção de outliers;
- **Características**: As 13 features originais estratificadas, resultando na adição de 7 features, totalizando 20 features;
- **Desbalanceamento original**: 1,35:1 (normais:doentes).

## Visualizações

As seguintes visualizações estão disponíveis no notebook:

1. **Distribuição das classes**  
   ![Distribuição da classe alvo](images/class_distribution.png)
   
## Tabelas Adicionais

### Top 5 Features Correlacionadas com o alvo

| Feature | Correlação |
|---------|------------|
| thal | 	0,533 |
| slope | 0,400 |
| cp | 0,378 |
| ca | 0,352 |
| thalachh | 0,242 |


## Principais Observações da Análise Exploratória

Observações feitas com a Análise Exploratória de Dados da biblioteca Pandas profiling.

- **Classe alvo balanceada**: 49,61% dos pacientes tem indicativo de doença;
- **Uma feature com alta correlação**: a feature "thal" isolada é forte preditora de doença; 
