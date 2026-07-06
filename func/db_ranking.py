import pandas as pd
from sqlalchemy import create_engine
import pymysql # Driver para MySQL

# --- CONFIGURAÇÃO DO MYSQL ---
DB_USER = 'root'
DB_PASSWORD = 'filacro6869'
DB_HOST = 'localhost' # Ou o IP/Nome do seu servidor
DB_PORT = 3306
DB_NAME = 'predicao' # O nome do seu banco de dados
TABLE_NAME = 'ranking' # O nome da tabela onde o ranking será salvo
# ----------------------------

def create_mysql_engine():
    """Cria e retorna o objeto SQLAlchemy Engine para a conexão MySQL."""
    # O formato da string de conexão é:
    # 'mysql+<driver>://<user>:<password>@<host>:<port>/<db_name>'
    DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    try:
        print("Conectando ao banco")
        engine = create_engine(DB_URL)
        return engine
    except Exception as e:
        print(f"Erro ao criar o Engine do MySQL: {e}")
        return None

def save_ranking_to_mysql(df_ranking, table_name):
    """Salva o DataFrame de ranking na tabela MySQL (adicionando novas linhas)."""

    engine = create_mysql_engine()

    if engine is None:
        print("Não foi possível salvar os dados, a conexão falhou.")
        return

    print(f"\nTentando adicionar {len(df_ranking)} linhas na tabela '{table_name}'...")

    try:
        # MUDANÇA CRUCIAL: Usar 'append' para adicionar as novas linhas sem apagar o histórico
        df_ranking.to_sql(
            name=table_name,
            con=engine,
            if_exists='append', # <--- ALTERADO PARA ADICIONAR AO INVÉS DE DELETAR
            index=False,
            chunksize=1000
        )
        print(f"Sucesso! O ranking foi adicionado à tabela '{table_name}'.")

    except Exception as e:
        # Se ocorrer um erro, pode ser por tentativa de inserir um ID duplicado (chave primária)
        print(f"Erro ao salvar no MySQL. Verifique se o ID já existe ou se há incompatibilidade de colunas: {e}")

def load_latest_ranking_from_mysql(table_name):
    """
    Carrega o ranking mais recente (pela data) da tabela MySQL.
    """
    engine = create_mysql_engine()

    if engine is None:
        return pd.DataFrame()

    try:
        # 1. Encontra a data de execução mais recente na tabela.
        query_max_date = f"SELECT MAX(Data) FROM {table_name}"
        max_date_df = pd.read_sql(query_max_date, con=engine)
        latest_date = max_date_df.iloc[0, 0]

        if latest_date is None:
            print("Tabela de ranking vazia.")
            return pd.DataFrame()

        # 2. Carrega todos os registros para essa data mais recente.
        # Usa o formato de string para a data na query SQL (YYYY-MM-DD)
        latest_date_str = latest_date.strftime('%Y-%m-%d')
        query_latest_ranking = f"SELECT * FROM {table_name} WHERE Data = '{latest_date_str}' ORDER BY Risco DESC"

        df_latest_ranking = pd.read_sql(query_latest_ranking, con=engine)
        print(f"Sucesso! Carregado {len(df_latest_ranking)} registros do ranking da data {latest_date_str}.")
        return df_latest_ranking

    except Exception as e:
        print(f"Erro ao carregar ranking do MySQL: {e}")
        return pd.DataFrame()
