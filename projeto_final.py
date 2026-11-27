
{
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "venv (3.11.9)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}


####################################################################################################
#------------------------------ETAPA 01 - Análise e Preparação do dataset -------------------------#
####################################################################################################
# Importações 
from pathlib import Path
from collections import Counter
import pandas as pd
import cv2
import hashlib
import random
import matplotlib.pyplot as plt


print("="*50)
print("ETAPA 01 - ANÁLISE E PREPARAÇÃO DO DATASET")
print("="*50)
print("1.1 - Identificação das classes e quantidade de imagens por classe")
print("="*50)

print(Path.cwd())
caminho_dataset = Path(r"fruits-360-100x100")
caminho_treino = caminho_dataset / "Training"
caminho_teste = caminho_dataset / "Test"

print(caminho_dataset.exists())
print(caminho_treino.exists())
print(caminho_teste.exists())


# ---- IDENTIFICAÇÃO DAS CLASSES ----

classes_treino = []
classes_teste = []

for pasta in caminho_treino.iterdir():
    if pasta.is_dir():
        print(pasta.name)
        classes_treino.append(pasta.name)

for pasta in caminho_teste.iterdir():
    if pasta.is_dir():
        #print(pasta.name)
        classes_teste.append(pasta.name)

comparacao = set(classes_treino).difference(set(classes_teste))
comparacao_inversa = set(classes_teste).difference(set(classes_treino))


print(f"Quantidade de classes em treino: {len(classes_treino)}")
print(f"\nQuantidade de classes em teste: {len(classes_teste)}")
print(f"\nClasses faltantes no conjunto de teste com base no treino:\n{comparacao}")
print(f"\nClasses faltantes no conjunto de treino com base no teste:\n{comparacao_inversa}")


classes_treino_normalizadas = [classe.casefold() for classe in classes_treino]
classes_teste_normalizadas = [classe.casefold() for classe in classes_teste]

comparacao = set(classes_treino_normalizadas).difference(set(classes_teste_normalizadas))
comparacao_inversa = set(classes_teste_normalizadas).difference(set(classes_treino_normalizadas))

print(f"\nClasses faltantes no conjunto de teste com base no treino:\n{comparacao}")
print(f"\nClasses faltantes no conjunto de treino com base no teste:\n{comparacao_inversa}")


# ---- CONTAGEM DAS IMAGENS POR CLASSE ----

pasta = caminho_treino / "Apple Braeburn 1"
imagens = list(pasta.glob("*.jpg"))
print(pasta.exists())
print(f"Quantidade de imagens da classe Apple Braeburn 1:\n{len(imagens)}")


# ---- CONTAGEM DAS IMAGENS POR CLASSE ----
dados_treino = []
dados_teste = []

def contagem_imagens(caminho, dados):
    for pasta in caminho.iterdir():
        if pasta.is_dir():
            imagens = list(pasta.glob("*.jpg"))
            quantidade = len(imagens)
            registro = {"classe": pasta.name, 
                        "quantidade": quantidade}
            dados.append(registro)
    return dados

df_treino = pd.DataFrame(contagem_imagens(caminho_treino, dados_treino))
df_teste = pd.DataFrame(contagem_imagens(caminho_teste, dados_teste))
print(f"---- Contagem das imagens por classe do conjunto de treino----")
df_treino.head()


print(f"---- Contagem das imagens por classe do conjunto de teste----\n")
df_teste.head()


print(50*("="))
print("------- INORMAÇÕES DO CONJUNTO DE TREINO ---------")
print(50*("="))

print(f"\nTamanho do df:{df_treino.shape}")
print(f"\n====== Informações: ======")
df_treino.info()
print(f"\n====== Total de imagens: =======")
print(df_treino['quantidade'].sum())
print(f"\n====== Estatísticas: ======")
df_treino.describe()


print(50*("="))
print("------- INORMAÇÕES DO CONJUNTO DE TESTE ---------")
print(50*("="))

print(f"\nTamanho:{df_teste.shape}")
print(f"\n====== Informações: ======")
df_teste.info()
print(f"\n====== Total de imagens: =======")
print(df_teste['quantidade'].sum())
print(f"\n====== Estatísticas: ======")
df_treino.describe()

# Analisando se há alguma classe sem registro
df_treino.loc[df_treino["quantidade"] == 0]
df_teste.loc[df_teste["quantidade"] == 0]

########################################################################################
# 1.2 - Verificação da integridade dos dados
#########################################################################################

print("="*50)
print("1.2 - Verificação da integridade dos dados")
print("="*50)

# Identificando possíveis erros ao carregar as imagens - OpenCV
def CarregarImagem_CV2(caminho):
    try:
        imagem_opencv = cv2.imread(str(caminho))
        if imagem_opencv is None:
            return False
        else:
            return True
    except Exception:
        return False

## Verificação do conjunto de treinamento
resultados_CV2_treino = []

for pasta in caminho_treino.iterdir():
    if pasta.is_dir():
        for caminho_imagem in pasta.glob("*.jpg"):
            resultado = CarregarImagem_CV2(caminho_imagem)
            registro = {"classe": pasta.name, 
                        "imagem": caminho_imagem.name,
                        "leitura_sucesso": resultado}
            resultados_CV2_treino.append(registro)


## Verificação do conjunto de teste
resultados_CV2_teste = []

for pasta in caminho_teste.iterdir():
    if pasta.is_dir():
        for caminho_imagem in pasta.glob("*.jpg"):
            resultado = CarregarImagem_CV2(caminho_imagem)
            registro = {"classe": pasta.name, 
                        "imagem": caminho_imagem.name, 
                        "leitura_sucesso": resultado}
            resultados_CV2_teste.append(registro)            


# Validando os resultados
print("="*50)
print(f"Imagens analisadas no treinamento: {len(resultados_CV2_treino)}")
print(f"Imagens analisadas no teste: {len(resultados_CV2_teste)}")
print(f"Resultados do treino: \n{resultados_CV2_treino[:5]}")
print(f"Resultados do teste: \n{resultados_CV2_teste[:5]}")
print("="*50)


# Armazenando os resultados em dataframes
df_resultados_CV2_treino = pd.DataFrame(resultados_CV2_treino)
df_resultados_CV2_teste = pd.DataFrame(resultados_CV2_teste)

# Verificando a estrutura dos Dataframes
df_resultados_CV2_treino.head()
print("="*50)
print(f"Tamanho do df de treino: {df_resultados_CV2_treino.shape}")
print("="*50)
print(df_resultados_CV2_treino.info())
print("="*50)


df_resultados_CV2_teste.head()
print("="*50)
print(f"Tamanho do df de teste: {df_resultados_CV2_teste.shape}")
print("="*50)
print(df_resultados_CV2_teste.info())
print("="*50)

# Imagens carregadas corretementeno treinamento
print("="*50)
sucesso_treino = df_resultados_CV2_treino["leitura_sucesso"].sum()
print(f"Imagens carregadas corretemente no treinamento: {sucesso_treino}")

# Imagens carregadas corretementeno teste
sucesso_teste = df_resultados_CV2_teste["leitura_sucesso"].sum()
print(f"Imagens carregadas corretement no teste: {sucesso_teste}")
print("="*50)


# Imagens carregadas incorretementeno treino
print("="*50)
falhas_treino = (df_resultados_CV2_treino["leitura_sucesso"] == False).sum()
print(f"Falhas de leitura no treinamento: {falhas_treino}")

# Imagens carregadas incorretementeno teste
falhas_teste = (df_resultados_CV2_teste["leitura_sucesso"] == False).sum()
print(f"Falhas de leitura no teste: {falhas_teste}")
print("="*50)


# calculando a taxa de sucesso - treino
print("="*50)
taxa_sucesso_treino = (sucesso_treino / len(df_resultados_CV2_treino)) * 100
print(f"Taxa de sucesso no treinamento: {taxa_sucesso_treino:.2f}%")

# teste
taxa_sucesso_teste = (sucesso_teste / len(df_resultados_CV2_teste)) * 100
print(f"Taxa de sucesso no teste: {taxa_sucesso_teste:.2f}%")
print("="*50)


# verificando falhas - treino
imagens_falhas_treino = df_resultados_CV2_treino.loc[df_resultados_CV2_treino["leitura_sucesso"] == False]
imagens_falhas_treino

# teste
imagens_falhas_teste = df_resultados_CV2_teste.loc[df_resultados_CV2_teste["leitura_sucesso"] == False]
imagens_falhas_teste

#############################################
# 1.3 - Verificação de Imagens duplicadas
#############################################

print("="*50)
print("1.3 - Verificação de Imagens duplicadas")
print("="*50)

# calculando hash
def Calcula_hash(caminho_imagem):
        imagem_binary = caminho_imagem.read_bytes()     # lê o conteúdo de um arquivo e retorna seus dados em binário (bytes).
        obj_hash = hashlib.sha256()                     # Criamos o objeto vazio.         
        obj_hash.update(imagem_binary)                  # Fornecemos os bytes por meio de update().
        hash_hexa = obj_hash.hexdigest()                # Converte o resultado do hash em uma representação hexadecimal.
        return hash_hexa


# Teste para duas imagens
# pasta = caminho_treino / "Almonds 1"
# imagem_1 = pasta / "r0_0_100.jpg"
# imagem_2 = pasta / "r0_1_100.jpg"
# hash_1 = Calcula_hash(imagem_1)
# hash_2 = Calcula_hash(imagem_2)
# print(hash_1)
# print(hash_2)


# Calculando os hashes das imagens de treinamento
hashes_treino = []
for pasta in caminho_treino.iterdir():
    if pasta.is_dir():
        for caminho_imagem in pasta.glob("*.jpg"):
            hash_imagem = Calcula_hash(caminho_imagem)
            registro = {"classe": pasta.name,
                        "imagem": caminho_imagem.name,
                        "hash": hash_imagem,
                        "conjunto": "Training"}
            hashes_treino.append(registro)



# Calculando os hashes das imagens de teste
hashes_teste = []
for pasta in caminho_teste.iterdir():
    if pasta.is_dir():
        for caminho_imagem in pasta.glob("*.jpg"):
            hash_imagem = Calcula_hash(caminho_imagem)
            registro = {"classe": pasta.name,
                        "imagem": caminho_imagem.name,
                        "hash": hash_imagem,
                        "conjunto": "Test"}
            hashes_teste.append(registro)



# Transformando os resultados em DataFrames
df_hashes_treino = pd.DataFrame(hashes_treino)
df_hashes_teste = pd.DataFrame(hashes_teste)

print("Imagens analisadas no treinamento:", len(df_hashes_treino))
print("Imagens analisadas no teste:", len(df_hashes_teste))


# Visualizando os registros
print("Visualizando os registros")
print(df_hashes_treino.head())
print(df_hashes_teste.head())

print("Verificando a estrutura")
df_hashes_treino.info()
df_hashes_teste.info()

# Verificando duplicatas - treino
duplicatas_treino = df_hashes_treino.duplicated(subset="hash")
print("Quantidade de duplicatas no treinamento:", duplicatas_treino.sum())

df_duplicatas_treino = df_hashes_treino.loc[
    df_hashes_treino.duplicated(subset="hash", keep=False)
]

print(df_duplicatas_treino)


# Verificando duplicatas - teste
duplicatas_teste = df_hashes_teste.duplicated(subset="hash")
print("Quantidade de duplicatas no teste:", duplicatas_teste.sum())

df_duplicatas_teste = df_hashes_teste.loc[
    df_hashes_teste.duplicated(subset="hash", keep=False)
]
print(df_duplicatas_teste)


# Identificando duplicatas entre treinamento e teste
hashes_unicos_treino = set(df_hashes_treino["hash"])
hashes_unicos_teste = set(df_hashes_teste["hash"])

hashes_comuns = hashes_unicos_treino.intersection(hashes_unicos_teste)
print("Quantidade de hashes presentes em ambos os conjuntos:", len(hashes_comuns))

df_duplicatas_entre_conjuntos = pd.concat(
    [df_hashes_treino, df_hashes_teste]
)

df_duplicatas_entre_conjuntos = df_duplicatas_entre_conjuntos.loc[
    df_duplicatas_entre_conjuntos["hash"].isin(hashes_comuns)
]

print(df_duplicatas_entre_conjuntos)


##########################################################
# 1.4 - Verificando possíveis problemas de rotulagem
##########################################################

print("="*50)
print("1.4 - Verificando possíveis problemas de rotulagem")
print("="*50)

# Selecionando 10 classes aleatórias 
classes_amostra = random.sample(classes_treino, 10)
print(f"Classes selecionadas:\n{classes_amostra}")

# Selecionando imagens aleatórias das classes
imagens_amostra = []
for classe in classes_amostra:
    pasta = caminho_treino / classe
    imagens = list(pasta.glob("*.jpg"))
    imagens_selecionadas = random.sample(imagens, 5)
    for imagem in imagens_selecionadas:
        imagens_amostra.append({"classe": classe,
                                "caminho": imagem})


# Visualizando as imagens selecionadas

fig, axes = plt.subplots(10, 5, figsize=(15, 25))

for i, registro in enumerate(imagens_amostra):
    imagem = cv2.imread(str(registro["caminho"]))
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    linha = i // 5
    coluna = i % 5
    axes[linha, coluna].imshow(imagem)
    axes[linha, coluna].set_title(registro["classe"], fontsize=9)
    axes[linha, coluna].axis("off")
plt.tight_layout()
plt.show()


##########################################################################
# 1.5 - Analizando o balanciamento das classes
##########################################################################

print("="*50)
print("1.5 - Analizando o balanciamento das classes")
print("="*50)


# Classes com maior quantidade de imagens no treinamento
df_treino_ordenado = df_treino.sort_values(by="quantidade", ascending=False)
print("Classes com maior quantidade de imagens no treinamento:")
print(df_treino_ordenado.head(10))

# Classes com menor quantidade de imagens no treinamento
df_treino_ordenado_crescente = df_treino.sort_values(by="quantidade", ascending=True)
print("Classes com menor quantidade de imagens no treinamento:")
print(df_treino_ordenado_crescente.head(10))

print("Estatísticas do treinamento:")
print(df_treino["quantidade"].describe())


# Classes com maior quantidade de imagens no teste
df_teste_ordenado = df_teste.sort_values(by="quantidade", ascending=False)
print("Classes com maior quantidade de imagens no teste:")
print(df_teste_ordenado.head(10))

# Classes com menor quantidade de imagens no teste
df_teste_ordenado_crescente = df_teste.sort_values(by="quantidade", ascending=True)
print("Classes com menor quantidade de imagens no teste:")
print(df_teste_ordenado_crescente.head(10))


print("Estatísticas do teste:")
print(df_teste["quantidade"].describe())


# Comparando as quantidades de imagens por classe
df_comparacao = pd.merge(df_treino, df_teste, 
                         on="classe", suffixes=("_treino", "_teste"))
print("Quantidade de imagens por classe:")
print(df_comparacao.head())


# Calculando a proporção de treinamento e teste

df_comparacao["total"] = (
    df_comparacao["quantidade_treino"] +
    df_comparacao["quantidade_teste"]
)

df_comparacao["proporcao_treino"] = (
    df_comparacao["quantidade_treino"] /
    df_comparacao["total"]
)

df_comparacao["proporcao_teste"] = (
    df_comparacao["quantidade_teste"] /
    df_comparacao["total"]
)
print("Proporção de treinamento e teste:")
print(df_comparacao.head())

# Histograma do conjunto de treinamento
plt.figure(figsize=(10, 6))

plt.hist(
    df_treino["quantidade"],
    bins=20,
    edgecolor="black"
)

plt.title("Distribuição da quantidade de imagens por classe - Treinamento")
plt.xlabel("Quantidade de imagens por classe")
plt.ylabel("Número de classes")

plt.grid(axis="y", alpha=0.3)

plt.show()


# Histograma do conjunto de teste
plt.figure(figsize=(10, 6))

plt.hist(
    df_teste["quantidade"],
    bins=20,
    edgecolor="black"
)

plt.title("Distribuição da quantidade de imagens por classe - Teste")
plt.xlabel("Quantidade de imagens por classe")
plt.ylabel("Número de classes")

plt.grid(axis="y", alpha=0.3)

plt.show()


# Selecionando as 10 classes com maior quantidade de imagens
top_10 = df_treino_ordenado.head(10)

plt.figure(figsize=(12, 6))

plt.bar(
    top_10["classe"],
    top_10["quantidade"]
)

plt.title("10 classes com maior quantidade de imagens - Treinamento")
plt.xlabel("Classes")
plt.ylabel("Quantidade de imagens")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.show()



maior_quantidade = df_treino["quantidade"].max()
menor_quantidade = df_treino["quantidade"].min()
diferenca = maior_quantidade - menor_quantidade

print("Maior quantidade:", maior_quantidade)
print("Menor quantidade:", menor_quantidade)
print("Diferença:", diferenca)

razao = maior_quantidade / menor_quantidade
print("Razão entre a maior e a menor classe:", razao)


####################################################################
# 1.6 - Exemplos de imagens e principais dificuldades do modelo
####################################################################

print("="*50)
print("1.6 - Exemplos de imagens e principais dificuldades do modelo")
print("="*50)

# Visualizando classes relacionadas a maçãs
classes_maça = [
    classe for classe in classes_treino
    if "Apple" in classe
]
print(classes_maça)

# Visualizando classes relacionadas a peras
classes_pera = [
    classe for classe in classes_treino
    if "Pear" in classe
]
print(classes_pera)

# Visualizando classes relacionadas a cerejas
classes_cereja = [
    classe for classe in classes_treino
    if classe.startswith("Cherry")
]
print(classes_cereja)


# Visualizando diferentes classes do mesmo grupo
# Escolha manual de classes para comparação

classes_visualizacao = [
    "Apple Braeburn 1",
    "Apple Granny Smith 1",
    "Apple Golden 1",
    "Apple Red 1",
    "Apple Pink Lady 1"
]

imagens_visualizacao = []

for classe in classes_visualizacao:
    pasta = caminho_treino / classe
    imagens = list(pasta.glob("*.jpg"))
    imagem_selecionada = random.choice(imagens)
    imagens_visualizacao.append({
        "classe": classe,
        "caminho": imagem_selecionada
    })


# Selecionar uma imagem aleatória de cada classe
def selecionar_imagens(classes, caminho_base):
    imagens_selecionadas = []
    for classe in classes:
        pasta = caminho_base / classe
        imagens = list(pasta.glob("*.jpg"))
        imagem = random.choice(imagens)
        imagens_selecionadas.append({
            "classe": classe,
            "caminho": imagem
        })

    return imagens_selecionadas


# Criar os grupos de análise

classes_macas = [
    "Apple Braeburn 1",
    "Apple Granny Smith 1",
    "Apple Golden 1",
    "Apple Pink Lady 1",
    "Apple Red Delicious 1"
]

classes_peras = [
    "Pear Abate 1",
    "Pear Forelle 1",
    "Pear Kaiser 1",
    "Pear Williams 1",
    "Pear Red 1"
]

classes_cerejas = [
    "Cherry 1",
    "Cherry Rainier 1",
    "Cherry Sour 1",
    "Cherry Wax Red 1",
    "Cherry Wax Yellow 1"
]

# Selecionando as imagens 
imagens_macas = selecionar_imagens(
    classes_macas,
    caminho_treino
)

imagens_peras = selecionar_imagens(
    classes_peras,
    caminho_treino
)

imagens_cerejas = selecionar_imagens(
    classes_cerejas,
    caminho_treino
)


# Criar uma função para visualizar as imagens

def visualizar_imagens(imagens_selecionadas, titulo):
    fig, axes = plt.subplots(
        1,
        len(imagens_selecionadas),
        figsize=(15, 4)
    )
    fig.suptitle(titulo, fontsize=16)
    for i, registro in enumerate(imagens_selecionadas):
        imagem = cv2.imread(str(registro["caminho"]))
        imagem = cv2.cvtColor(
            imagem,
            cv2.COLOR_BGR2RGB
        )
        axes[i].imshow(imagem)
        axes[i].set_title(
            registro["classe"],
            fontsize=9
        )
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


visualizar_imagens(
    imagens_macas,
    "Comparação visual entre variedades de maçãs"
)

visualizar_imagens(
    imagens_peras,
    "Comparação visual entre variedades de peras"
)

visualizar_imagens(
    imagens_cerejas,
    "Comparação visual entre variedades de cerejas"
)

##################################################################
# 1.7 - Divisão dos dados e prevenção de vazamento de dados
##################################################################

print("="*50)
print("1.7 - Divisão dos dados e prevenção de vazamento de dados")
print("="*50)

# Verificando hashes compartilhados entre treinamento e teste
hashes_treino = set(df_hashes_treino["hash"])
hashes_teste = set(df_hashes_teste["hash"])
hashes_comuns = hashes_treino.intersection(hashes_teste)
print("Quantidade de hashes compartilhados:", len(hashes_comuns))


# Investigando nomes de arquivos relacionados
pasta = caminho_treino / "Apple Braeburn 1"
imagens = list(pasta.glob("*.jpg"))
for imagem in imagens[:20]:
    print(imagem.name)


# Criando uma estrutura com os caminhos das imagens
def criar_dataframe_imagens(caminho_base):
    registros = []
    for pasta in caminho_base.iterdir():
        if pasta.is_dir():
            for imagem in pasta.glob("*.jpg"):
                registros.append({
                    "classe": pasta.name,
                    "caminho": imagem
                })
    df = pd.DataFrame(registros)
    return df


df_imagens_treino = criar_dataframe_imagens(caminho_treino)
df_imagens_teste = criar_dataframe_imagens(caminho_teste)

print("Treinamento:")
print(df_imagens_treino.shape)

print("\nTeste:")
print(df_imagens_teste.shape)

print(df_imagens_treino.head())


# Selecionando 25% das imagens de treinamento
df_treino_reduzido = (
    df_imagens_treino
    .groupby("classe", group_keys=False)
    .sample(frac=0.25, random_state=42)
)

print("Quantidade de imagens selecionadas:", len(df_treino_reduzido))


# Selecionando 25% das imagens de teste
df_teste_reduzido = (
    df_imagens_teste
    .groupby("classe", group_keys=False)
    .sample(frac=0.25, random_state=42)
)

print("Quantidade de imagens selecionadas:", len(df_teste_reduzido))


print("===== DATASET REDUZIDO =====")

print("\nTreinamento:")
print(df_treino_reduzido.shape)

print("\nTeste:")
print(df_teste_reduzido.shape)

print("\nTotal:")
print(
    len(df_treino_reduzido) +
    len(df_teste_reduzido)
)


classes_reduzidas_treino = set(
    df_treino_reduzido["classe"]
)

classes_reduzidas_teste = set(
    df_teste_reduzido["classe"]
)

print("Classes no treinamento:", len(classes_reduzidas_treino))
print("Classes no teste:", len(classes_reduzidas_teste))

# verificamos se as classes são iguais
print(
    "Classes ausentes no treinamento:",
    set(classes_treino) - classes_reduzidas_treino
)

print(
    "Classes ausentes no teste:",
    set(classes_teste) - classes_reduzidas_teste
)


# Verificando o balanciamento após a redução
contagem_reduzida_treino = (
    df_treino_reduzido["classe"]
    .value_counts()
    .sort_index()
)

contagem_reduzida_teste = (
    df_teste_reduzido["classe"]
    .value_counts()
    .sort_index()
)

print("Estatísticas do treinamento reduzido:")
print(contagem_reduzida_treino.describe())
print("\nEstatísticas do teste reduzido:")
print(contagem_reduzida_teste.describe())

# Visualizando a distribuição após a redução 
# Treino
plt.figure(figsize=(10, 5))
plt.hist(
    contagem_reduzida_treino,
    bins=20,
    edgecolor="black"
)
plt.title("Distribuição de imagens por classe - Treinamento reduzido")
plt.xlabel("Quantidade de imagens por classe")
plt.ylabel("Número de classes")
plt.grid(axis="y", alpha=0.3)
plt.show()

# Teste
plt.figure(figsize=(10, 5))
plt.hist(
    contagem_reduzida_teste,
    bins=20,
    edgecolor="black"
)
plt.title("Distribuição de imagens por classe - Teste reduzido")
plt.xlabel("Quantidade de imagens por classe")
plt.ylabel("Número de classes")
plt.grid(axis="y", alpha=0.3)
plt.show()


# Salvando a seleção
df_treino_reduzido.to_csv(
    "treino_reduzido.csv",
    index=False
)

df_teste_reduzido.to_csv(
    "teste_reduzido.csv",
    index=False
)

