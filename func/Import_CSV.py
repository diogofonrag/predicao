import pandas as pd
import requests
import csv
from django.http import HttpResponse
import io

def importarquivo(Data_A, Data_P, Matricula, arquivo):
    List_info = []

    def tratamento(Data):
        print("Transformando data em string")
        str_data_a = str(Data)
        sep_01 = "/"
        sep_02 = " "
        tata_data = str_data_a[8:10] + sep_01 + str_data_a[5:7] + sep_01 + str_data_a[0:4] + sep_02 + str_data_a[11:13] + str_data_a[14:16]
        return tata_data


    print("  Iniciando a importação do arquivo csv ")
    colunas = [0, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 32, 33, 34, 35, 40]
    List_info = pd.read_csv(arquivo,
                            sep=';',
                            index_col=False,
                            usecols = colunas,
                            encoding='utf-8',
                        )
    print(List_info)
    List_info.info()
    formato = ".csv"
    caminho = "Stme_app/templates/www_stme/"
    local = caminho + "STME" + formato
    colunas_01 = ['ID', 'UF_A', 'SUBREGIAO', 'GRA', 'SIGLA_LOCALIDADE_A', 'NOME_LOCALIDADE_A', 'SIGLA_ESTACAO_A', 'NOME_ESTACAO_A', 'RAMIFICACAO', 'IDENT_CABO_ELEM_REDE', 'SEGMENTO_PROGRAMACAO', 'AREA_TECNICA', 'TIPO_OS', 'MODELO', 'FABRICANTE', 'TIPO_EQUIPAMENTO', 'COD_MATRICULA_TEC', 'NOME_AGENTE', 'CRITICIDADE', 'PERIODICIDADE', 'DIA_SEMANA', 'COS', 'ENLACE', 'TAMANHO_ENLACE_KM', 'ID_PRS_RTU', 'BARRAMENTO', 'TIPO_CABO']
    List_info = List_info[colunas_01]
    List_info.insert(21, column = "21", value=tratamento(Data_A))
    List_info.insert(22, column = "22", value=tratamento(Data_P))
    List_info.insert(29, column = "29", value=Matricula)
    buffer = io.StringIO()
    Lista = List_info.to_csv(buffer, sep=';', header=False, index=False)
    #Lista = List_info.to_csv(buffer, local, sep=';', header=False, index=False)
    buffer.seek(0)
    response = HttpResponse(buffer, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="STME_modificado.csv"'
    return response
