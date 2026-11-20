# |              Etapa              |           Bibliotecas principais              |
# |---|---|
# | 1. Carregamento das imagens     |           OpenCV, Pillow, NumPy               |
# | 2. Análise exploratória         |           Pandas, Matplotlib, Seaborn         |
# | 3. Amostragem de 25%            |           Pandas, NumPy, Scikit-learn         |
# | 4. Pré-processamento            |           OpenCV, NumPy                       |
# | 5. Segmentação                  |           OpenCV, Scikit-image                |
# | 6. Extração de características  |           Scikit-image, OpenCV, NumPy         |
# | 7. Construção do dataframe      |           Pandas                              |
# | 8. Treinamento                  |           Scikit-learn                        |
# | 9. Avaliação                    |           Scikit-learn, Matplotlib, Seaborn   |
# | 10. Salvamento dos resultados   |           Joblib, Pandas                      |


# | Característica          | Scikit-image          | OpenCV |
# |---|---|---|
# | Função de leitura       | `io.imread()`         | `cv2.imread()` |
# | Resultado               | Matriz NumPy          | Matriz NumPy |
# | Falha de leitura        | Pode gerar exceção    | Normalmente retorna `None` |
# | Espaço de cores padrão  | RGB                   | BGR |
# | Tipo de retorno         | `ndarray`             | `ndarray` |


# Scikit-image (skimage)

# É uma biblioteca voltada ao processamento de imagens científicos e computacional.

# Algumas funcionalidades:
# - Leitura de imagens.
# - Conversão de espaços de cores.
# - Redimensionamento.
# - Aplicação de filtros.
# - Segmentação.
# - Extração de características.
# - Análise de propriedades das imagens.
# OpenCV (cv2)

# É uma biblioteca bastante utilizada em visão computacional.

# Algumas funcionalidades:
# - Leitura e escrita de imagens.
# - Redimensionamento.
# - Conversão de espaços de cores.
# - Aplicação de filtros.
# - Detecção de bordas.
# - Segmentação.
# - Extração de características.
# - Operações morfológicas.
# Perceba que existe uma sobreposição de funcionalidades. Isso é interessante porque permite investigar como diferentes ferramentas executam operações semelhantes.

# Assim, poderíamos responder a uma pergunta de pesquisa interessante:

# ####################################################################################
# Como a utilização de diferentes 
# bibliotecas de processamento de imagens influencia o 
# desempenho de modelos de classificação de frutas e vegetais?
# #####################################################################################

# Essa comparação pode acrescentar valor científico ao trabalho, desde que seja conduzida de maneira controlada.




# | Indicador                   | Scikit-image | OpenCV |
# |---                          |---:          |---:    |
# | Total de imagens analisadas | ...          | ...    |
# | Imagens carregadas          | ...          | ...    |
# | Falhas de leitura           | ...          | ...    |
# | Taxa de sucesso (%)         | ...          | ...    |
# | Tempo de execução (s)       | ...          | ...    |


# Uma métrica que podemos calcular é a taxa de sucesso de leitura:
# 
# Taxa de sucesso = Imagens carregadas / Total de imagens analisadas * 100
# Essa métrica permitirá comparar a capacidade de leitura das bibliotecas.
    
# Atenção: taxa de sucesso de leitura não é acurácia de classificação.
# Também precisaremos garantir que ambas sejam testadas sobre os mesmos arquivos, nas mesmas condições.
















































