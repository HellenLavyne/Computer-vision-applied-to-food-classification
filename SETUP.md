# Guia de Configuração do Ambiente (Equipe)

Siga este passo a passo para configurar o projeto no seu computador e manter tudo sincronizado.

## 1. Clonar os repositórios
No terminal, baixe o repositório do nosso projeto e a pasta do dataset:

```bash
git clone <URL_DO_REPOSITORIO_DA_EQUIPE>
```
Após entrar no diretório:

```bash
git clone <URL_DO_REPOSITORIO_DO_DATASET>
```

## 2. Configurar o VS Code
Abra a pasta do projeto (`AV2`) no VS Code, acesse a aba de **Extensões** (`Ctrl + Shift + X`) e instale:
* **Python** (Microsoft)
* **Jupyter** (Microsoft)

## 3. Criar e ativar o ambiente virtual (`venv`)
No terminal do VS Code, crie o ambiente virtual:

```bash
python -m venv venv
```

Ative o ambiente virtual:

```bash
.\venv\Scripts\Activate
```

> **Nota:** Se ocorrer erro de permissão no Windows, execute o comando abaixo antes de tentar ativar novamente:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

## 4. Instalar as dependências
Com o `(venv)` ativo em verde no terminal, atualize o `pip` e instale as bibliotecas do projeto:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Conectar o Kernel do Jupyter
1. Abra o arquivo `.ipynb` da etapa em que você vai trabalhar (ex: `01_analise_e_preparacao_do_dataset.ipynb` ou `projeto_final.ipynb`).
2. No canto superior direito, clique em **Selecionar Kernel** > **Ambientes Python**.
3. Escolha a opção que aponta para a pasta `venv` do projeto.

## 6. Organização e Fluxo de trabalho no GitHub
O projeto está dividido em arquivos numerados de `01` a `06` para que cada integrante trabalhe na sua etapa sem gerar conflitos de código, reunindo os resultados finais em `projeto_final.ipynb`.

Antes de começar a editar qualquer código, sempre baixe as alterações mais recentes da equipe:

```bash
git pull origin main
```

Após finalizar suas alterações, envie para o repositório:

```bash
git add .
git commit -m "Descreva brevemente o que você fez"
git push origin main
```

Lembre-se sempre de salvar (`Ctrl + S`) cada arquivo modificado antes de enviar para o repositório.

O arquivo `annotations.py` será utilizado para nós registrarmos informações/modificações/explicações e/ou o que acharmos importante. 

Vamos lá! 😎👩🏻‍💻

![alt text](image.png)
