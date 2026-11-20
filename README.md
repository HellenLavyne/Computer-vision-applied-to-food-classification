# Classificação de Imagens com Extração de Características — Fruits-360

Projeto desenvolvido como avaliação (AV2) da disciplina de **Laboratório de Ciência de Dados**. O objetivo é construir um pipeline completo de visão computacional clássica e aprendizado de máquina tradicional para classificar imagens de frutas, vegetais, nozes, sementes e cogumelos.

> **Restrição Metodológica:** Seguindo as diretrizes do projeto, não são utilizadas redes neurais convolucionais (CNNs), técnicas de aprendizado profundo (*Deep Learning*) ou modelos pré-treinados. Toda a extração de características e classificação é baseada em técnicas clássicas de PDI e algoritmos tradicionais de *Machine Learning*.

---

## 📊 Dataset Utilizado

* **Nome:** [Fruits-360 (100x100)](https://github.com/fruits-360/fruits-360-100x100)
* **Versão:** `2026.10.27.0`
* **Autoria e Citação:** Mihai Oltean, *Fruits-360 dataset*, 2017–
* **Licença do Dataset:** [CC BY-SA 4.0](https://creativecommons.org)

### Características Principais do Conjunto de Dados
* **Total de imagens:** 200.201 imagens coloridas (`100x100` pixels).
* **Divisão original:** 150.179 imagens para treinamento (`Training`) e 50.022 imagens para teste (`Test`).
* **Número de classes:** 278 categorias distintas (incluindo diferentes variedades da mesma fruta armazenadas como classes separadas).
* **Aquisição e fundo:** As amostras foram filmadas em um motor de baixa rotação (3 rpm) com fundo branco, posteriormente segmentado via algoritmo de preenchimento por inundação (*flood fill*) pelo autor original.

---

## 🗂️ Estrutura do Projeto e Metodologia

O pipeline foi dividido em etapas sequenciais e modulares para garantir clareza metodológica e reprodutibilidade:

1. **`01_analise_e_preparacao_do_dataset.ipynb`**: Exploração das classes, verificação de integridade/duplicatas, análise de balanceamento e definição da estratégia de divisão sem vazamento de dados (*data leakage*).
2. **`02_pre_processamento_das_imagens.ipynb`**: Aplicação e justificativa de operações de melhoria e padronização (como conversão de espaços de cores, filtros de redução de ruído e ajuste de contraste).
3. **`03_segmentacao.ipynb`**: Avaliação e aplicação de técnicas tradicionais de separação entre objeto e fundo (limiarização, operações morfológicas, contornos ou agrupamento).
4. **`04_extracao_caracteristicas_e_dataframe.ipynb`**: Transformação das imagens em vetores numéricos combinando descritores de **cor** (histogramas, médias/desvios), **textura** (LBP, matriz de coocorrência), **forma** (momentos de Hu, circularidade, área) e **gradientes/bordas** (HOG).
5. **`05_treinamento_e_avaliacao_dos_classificadores.ipynb`**: Comparação entre pelo menos dois algoritmos tradicionais (ex: KNN, SVM, Random Forest, Árvore de Decisão ou Regressão Logística) frente a um classificador *baseline*, utilizando validação cruzada e padronização no treino.
6. **`06_analise_dos_resultados.ipynb`**: Avaliação quantitativa (Acurácia, Precisão, Recall, F1-score por classe e F1 macro, Matriz de Confusão) e discussão crítica sobre o impacto de cada etapa nos erros e acertos do modelo.
* **`projeto_final.ipynb`**: Notebook consolidado contendo o fluxo completo e os resultados finais do estudo.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.11
* **Ambiente:** Jupyter Notebook / VS Code
* **Bibliotecas:** NumPy, Scikit-Learn, OpenCV, Matplotlib

---

## 🚀 Como Configurar o Ambiente

Para instruções detalhadas sobre como clonar os repositórios, criar o ambiente virtual (`venv`) e instalar as dependências via `requirements.txt`, consulte o nosso guia interno em **[SETUP.md](./SETUP.md)**.

---

## 👩‍💻 Equipe

* **Dayanne**
* **Hellen**
* **Suiany**
