# Integrantes - Checkpoint 5

| Nome | RM | 
|--------|------|
| Sofia Petruk | 556585 |
| Samuel Yariwake | 556461 |
| Lucas Fialho | 557884 |
| Júlia Monteiro | 557023 |
| João Amorim | 559213 |
| Luiz Kamada | 557652 |

# Treinamento de Redes Neurais com Keras

Este repositório contém a implementação completa dos exercícios de treinamento de redes neurais com Keras para a disciplina de Disruptive Architectures: IoT, IoB e Generative AI.

## Visão Geral do Projeto

O projeto consiste em dois exercícios práticos que demonstram a aplicação de redes neurais em problemas de classificação e regressão, comparando o desempenho com modelos tradicionais de machine learning.

### Exercício 1: Classificação Multiclasse
- **Dataset**: Wine Dataset (UCI Repository)
- **Objetivo**: Classificar vinhos em 3 classes diferentes
- **Tipo**: Problema de classificação multiclasse

### Exercício 2: Regressão
- **Dataset**: California Housing Dataset (Scikit-learn)
- **Objetivo**: Prever o valor médio das casas
- **Tipo**: Problema de regressão


## Exercício 1 - Classificação de Vinhos

### Especificações Técnicas
- **Arquitetura da Rede Neural**: 2 camadas ocultas com 32 neurônios cada
- **Função de Ativação**: ReLU (camadas ocultas), Softmax (saída)
- **Camada de Saída**: 3 neurônios (3 classes)
- **Função de Perda**: categorical_crossentropy
- **Otimizador**: Adam

### Modelos de Comparação
- Random Forest Classifier
- Logistic Regression

### Dataset
O Wine Dataset contém informações químicas de vinhos de três cultivares diferentes. O dataset possui:
- **178 amostras** de vinhos
- **13 características** químicas (álcool, acidez, fenóis, etc.)
- **3 classes** diferentes de vinhos

### Resultados Obtidos

| Modelo | Acurácia | Observações |
|--------|----------|-------------|
| Rede Neural | 94.4% | Boa capacidade de generalização |
| Random Forest | 97.2% | Melhor desempenho geral |
| Logistic Regression | 91.7% | Performance sólida para modelo linear |

### Análise dos Resultados
O Random Forest apresentou o melhor desempenho para este dataset, o que é esperado considerando:
- O dataset possui um número relativamente pequeno de amostras (178)
- As características já são bem estruturadas e informativas
- O Random Forest é naturalmente resistente ao overfitting
- Não requer normalização dos dados

A rede neural apresentou performance competitiva, demonstrando sua capacidade de aprender padrões complexos mesmo em datasets menores.

## Exercício 2 - Regressão de Preços de Casas

### Especificações Técnicas
- **Arquitetura da Rede Neural**: 3 camadas ocultas (64, 32, 16 neurônios)
- **Função de Ativação**: ReLU (camadas ocultas), Linear (saída)
- **Camada de Saída**: 1 neurônio (valor contínuo)
- **Função de Perda**: Mean Squared Error (MSE)
- **Otimizador**: Adam

### Modelos de Comparação
- Linear Regression
- Random Forest Regressor

### Dataset
O California Housing Dataset contém informações sobre habitação na Califórnia. O dataset possui:
- **20,640 amostras** de distritos
- **8 características** (renda mediana, idade das casas, localização, etc.)
- **Target**: Valor mediano das casas (em centenas de milhares de dólares)

### Resultados Obtidos

| Modelo | RMSE | MAE | R² Score | Erro Prático |
|--------|------|-----|----------|--------------|
| Rede Neural | 0.743 | 0.526 | 0.606 | ~$52.6k por casa |
| Linear Regression | 0.746 | 0.533 | 0.603 | ~$53.3k por casa |
| Random Forest | 0.484 | 0.328 | 0.825 | ~$32.8k por casa |

### Análise dos Resultados
O Random Forest obteve o melhor desempenho para este problema de regressão:
- **RMSE mais baixo**: 0.484 vs 0.743 (rede neural)
- **R² mais alto**: 0.825 vs 0.606 (rede neural)
- **Erro prático menor**: ~$33k vs ~$53k por casa

Fatores que contribuíram para o sucesso do Random Forest:
- Capacidade superior de capturar relações não-lineares
- Robustez contra outliers
- Habilidade de lidar com características de diferentes escalas
- Menor tendência ao overfitting

## Como Executar os Experimentos

### No Google Colab
1. Abra o Google Colab (colab.research.google.com)
2. Faça upload dos notebooks `.ipynb`
3. Execute as células sequencialmente
4. Os resultados serão salvos automaticamente

### Tempo de Execução Estimado
- **Exercício 1**: 2-3 minutos
- **Exercício 2**: 3-5 minutos
- **Total**: Aproximadamente 5-8 minutos

## Metodologia Aplicada

### Pré-processamento
1. **Divisão dos dados**: 80% treino, 20% teste
2. **Normalização**: StandardScaler para redes neurais
3. **Estratificação**: Manutenção da proporção de classes (classificação)
4. **Codificação**: One-hot encoding para variáveis categóricas

### Treinamento
1. **Validação**: 20% dos dados de treino para validação
2. **Early Stopping**: Prevenção de overfitting
3. **Batch Size**: Otimizado para cada dataset
4. **Épocas**: Determinadas por convergência

### Avaliação
1. **Métricas múltiplas**: Acurácia, precisão, recall (classificação)
2. **Métricas de erro**: RMSE, MAE, R² (regressão)
3. **Comparação direta**: Mesmo conjunto de teste para todos os modelos
4. **Interpretação prática**: Erros convertidos para valores monetários

## Conclusões Gerais

### Lições Aprendidas
1. **Tamanho do dataset importa**: Datasets menores podem favorecer modelos mais simples
2. **Normalização é crucial**: Especialmente para redes neurais e regressão linear
3. **Random Forest é robusto**: Consistentemente bom desempenho em ambos os problemas
4. **Redes neurais são flexíveis**: Boa adaptação a diferentes tipos de problemas

### Recomendações Práticas
- **Para produção**: Considerar Random Forest pela interpretabilidade e robustez
- **Para experimentação**: Redes neurais oferecem mais flexibilidade arquitetural
- **Para datasets pequenos**: Modelos mais simples podem ser mais eficazes
- **Para datasets grandes**: Redes neurais tendem a se destacar


# 🍌 Classificação do Grau de Maturação de Bananas com YOLOv8

## 🎯 Objetivo do Projeto
O objetivo deste projeto é desenvolver um modelo de **classificação de imagens** capaz de identificar o **grau de maturação de bananas**, classificando-as em diferentes estágios:  
**freshripe**, **freshunripe**, **overripe**, **ripe**, **rotten** e **unripe**.  

Essa solução visa **automatizar o processo de avaliação da qualidade das frutas**, podendo ser aplicada em contextos como:
- Agricultura de precisão 🍃  
- Controle de estoque em distribuidores e mercados 🏬  
- Sistemas de inspeção automatizados 🧠  

---

## 🧰 Ferramentas Utilizadas
- **Google Colab** — ambiente de execução e treinamento do modelo.  
- **Ultralytics YOLOv8** — framework de visão computacional utilizado para classificação.  
- **Roboflow** — utilizado para preparar, organizar e exportar o dataset de forma padronizada.  

---

## ⚙️ Hiperparâmetros e Configurações Principais

| Parâmetro | Valor Utilizado | Descrição |
|------------|----------------|------------|
| `model` | `yolov8n-cls.pt` | Modelo base de classificação da Ultralytics |
| `epochs` | **30** | Número de épocas de treino — ajustado para melhor desempenho |
| `batch` | 4 | Tamanho do lote de imagens processadas por iteração |
| `imgsz` | 320 | Tamanho das imagens utilizadas no treino |
| `optimizer` | `auto` | Otimizador padrão configurado automaticamente |
| `device` | GPU (Tesla T4 - Colab) | Aceleração do treinamento |

> 🧠 Após o aumento para **30 epochs**, o modelo apresentou resultados mais consistentes e confiança mais alta nas previsões.

---

## 📊 Resultados e Desempenho

- O modelo obteve **bom desempenho em testes de validação**, classificando corretamente as imagens.  
- Para cada imagem, o resultado exibe:
  - **Classe prevista** (ex: `freshunripe`)
  - **Confiança da predição** (ex: `99.99%`)
- O modelo mostrou excelente capacidade de distinguir entre estágios próximos, como *freshunripe* e *unripe*.  

📈 Com o aumento de épocas e ajustes nos parâmetros, espera-se:
- Maior **acurácia geral**
- Melhor **generalização** para novas imagens
- Redução de erros entre classes visualmente semelhantes

---

## 🗂️ Dataset Utilizado
O dataset utilizado foi o **Banana Ripeness Classification Dataset**, disponível publicamente no Roboflow Universe:

🔗 [Banana Ripeness Classification – Roboflow](https://universe.roboflow.com/roboflow-universe-projects/banana-ripeness-classification)

O conjunto contém imagens de bananas em diferentes estágios de maturação, separadas em **treino** e **validação**, com pastas por classe.

---

## 📁 Estrutura do Projeto




banana_dataset/
├── train/
│ ├── freshripe/
│ ├── freshunripe/
│ ├── overripe/
│ ├── ripe/
│ ├── rotten/
│ └── unripe/
└── valid/
├── freshripe/
├── freshunripe/
├── overripe/
├── ripe/
├── rotten/
└── unripe/





