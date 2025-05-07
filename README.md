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

Este model card segue os princípios propostos no artigo [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) para documentação responsável de modelos de machine learning.

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
  - Learning rate: 0.01
  - Número de épocas: 40

### Uso Pretendido

- **Uso primário**: Detectar pacientes propensos a possuírem doenças cardíadas;
- **Usuários pretendidos**: Pacientes em geral, com a presença ou não de comorbidades;
- **Casos fora do escopo**: Não deve ser usado como único mecanismo para indicação de doenças cardíacas.

### Fatores

- **Fatores relevantes**: Buscando indicar a propensão a doença, fatores como idade, nível de colesterol, presença de angina (dor no peito), pressão arterial, frequência cardiaca, glicemina em jejum;
- **Fatores de avaliação**: O modelo foi avaliado com diferentes thresholds de classificação para balancear entre precisão e recall
