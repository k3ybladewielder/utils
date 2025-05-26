# **How to use**

* **Criar um ambiente virtual:**

  ```bash
  python3 -m venv env
  ```

* **Ativar o ambiente virtual:**

  * Linux/macOS:

    ```bash
    source env/bin/activate
    ```
  * Windows (cmd):

    ```cmd
    env\Scripts\activate
    ```
  * Windows (PowerShell):

    ```powershell
    env\Scripts\Activate.ps1
    ```

* **Instalar os pacotes do `requirements.txt`:**

  ```bash
  pip install -r requirements.txt
  ```

* **Instalar o Jupyter no ambiente virtual:**

  ```bash
  pip install jupyter ipykernel
  ```

* **Adicionar o ambiente virtual como kernel no Jupyter:**

  ```bash
  python -m ipykernel install --user --name=env --display-name "Python (env)"
  ```

* **Iniciar o Jupyter Notebook:**

  ```bash
  jupyter notebook
  ```
