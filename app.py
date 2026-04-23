import streamlit
import streamlit  as st
import pandas as pd
import seaborn as sns
import numpy as np
import os
import sys
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.patches import Rectangle


from narwhals.selectors import categorical
#import tkinter as tk
from statsmodels.stats.multicomp import MultiComparison
import math

import matplotlib.pyplot as plt
from bleach import clean
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import anderson
from scipy.stats import shapiro, levene
from scipy.stats import anderson
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from streamlit import selectbox
from statsmodels.stats.diagnostic import acorr_ljungbox
import itertools







st.set_page_config(
    page_title="ANOVA App",
    page_icon="🧠",
    layout="centered",
)


st.write('**Análise consciente de dados**')
st.write('Email: jose.g.oliveira@ufv.br')








tab1, tab2, tab3 = st.tabs(["Pré-processamento e Análise descritiva", "Gráficos", "Pressupostos-ANOVA/ANOVA/Post-hoc teste "])

with tab1:
    st.title("Aplicativo anova")
    st.write("O objetivo é fazer uma **Anova** com no máximo 2 fatores para DIC.")
    meu_label = "Envie seu arquivo CSV"
    arquivo = st.file_uploader(meu_label, type="csv")# criopu um upload
    #data = pd.read_csv(arquivo, encoding='UTF-8', sep=';')# data em dataFrame
    modelo = st.radio('Você deseja ver o modelo de entrada da tabela?', ['Sim', 'Não '])
    if modelo == 'Sim':
        st.image("tabela.png", caption="Modelo de tabela", width=300)
        st.subheader('Configuração da **Planilha** ')
        st.warning('Células vazias devem ser preenchidas com **NA**')
        st.warning('Evitar colocar **pontuações** nos nomes das **variáveis**')
        st.warning('Evitar colocar **pontuações**  nos *níveis* das variáveis.')
        st.warning('Seguir o modelo de preenchimento da **planilha** acima.')
        st.warning('Os eventos são **dependentes**, então não esquecer de colocar   **sim**  em cada etapa.')

    if arquivo is  None:
        st.warning('Aguardando a escolha dos dados ')

    else:
        st.success(f"O arquivo selecionado foi: {arquivo.name}")

        data = pd.read_csv(arquivo, encoding='UTF-8', sep=';')
        data_copia = data.copy()

        escolha = st.radio("Você deseja ver seus dados ?", ["Sim", "Não"]).upper().strip()
        if escolha == 'SIM':
            st.dataframe(data)

        variavel = st.radio('Quantas variáveis categóricas você deseja analisar?', [1,2], horizontal = True)

        data1 = data.to_dict()
        chaves = data1.keys()
        chaves1 = list(chaves)


        escolhas = []
        if variavel == 1:
            categorica= st.selectbox('Escolha as variável categórica',['Selecione']+ chaves1, key = '1')
            if categorica != 'Selecione':
                st.success(f"Você escolheu a variável categórica: {categorica}") # escolha essa primeiro

            continua = st.selectbox('Escolha a variável contínua',['Selecione'] + chaves1, key = '2')
            if continua != 'Selecione': #Escolha essa depois que a primeira é escolhida
                assert continua in data.columns, f"Coluna {continua} não encontrada"
                st.success(f"Você escolheu a variável contínua: {continua}")
                #bamos
                
              

                if categorica != 'Selecione' and continua != 'Selecione':
                    escolhas.append(categorica)
                    escolhas.append(continua)
                    data = data[escolhas]# escolhi e armazenei as variáveis que quero trabalhar
                    st.write(data)

                    data_na = data.isna().sum()
                    #fazer uma função aqui! def retirarana(data na , data) e chamo novamente em outro lugar
                    if data_na.sum() == 0:
                        st.write(f'Você Não tem **NA** nas  variáveis de seus dados  ')
                        st.dataframe(data_na)
                    else:
                        st.write(f'Você  tem **NA** nas  variáveis de seus dados  ')
                        st.dataframe(data_na)
                        st.write('Você gostaria de retira  as **NAs** ou substituir por valores médios?')
                        escolha_2 = st.radio("Você deseja ?", ["Substituir por Valores médios", "Retirar Na"])
                        if escolha_2 == "Substituir por Valores médios":
                            data = data.fillna(data.median(numeric_only=True))
                            st.write('Dados com valores médios substituidos no lugar de NA')
                            st.dataframe(data)
                        else:
                            data = data.dropna(axis=0)
                            st.dataframe(data)  # manter o mes


                    #somente se essa condição for respeitada, então fazemos a anpalise
                    data_grouped = data.groupby(categorica)[continua].describe()
                    st.write(f"Análise descritiva da variável {continua}")
                    st.dataframe(data_grouped)
                    cv = data.loc[:,continua].values# transforma em array numpy  e pega os valores, para o cálculo
                    #st.write(cv)
                    #cálculo do cv
                    cv2 =  np.std(cv) / np.mean(cv) * 100
                    st.write(f"CV% = {cv2}")

                    #aqui 
                    # ============================================================
                    # Z-SCORE (POR TRATAMENTO – APENAS VISUALIZAÇÃO)
                    # ============================================================
                    st.subheader('Z-score (por tratamento)')
                    
                    data = data.copy()
                    
                    data['zscore'] = (
                        data
                        .groupby(categorica)[continua]
                        .transform(lambda x: (x - x.mean()) / x.std())
                    )
                    
                    st.write(data)
                    
                    # KDE
                    fig2, ax = plt.subplots()
                    sns.kdeplot(data=data, x='zscore', fill=True, alpha=0.3)
                    ax.set_title("Curva de KDE do Z-score (por tratamento)")
                    ax.axvline(0, color='red', linestyle='dashed', linewidth=1)
                    st.pyplot(fig2)
                    
                    # Boxplot
                    fig, ax = plt.subplots()
                    sns.boxplot(data=data, x='zscore', y=categorica, ax=ax)
                    sns.stripplot(data=data, x='zscore', y=categorica,
                                  color='black', jitter=True, alpha=0.5, ax=ax)
                    ax.set_title("Boxplot do Z-score por tratamento")
                    st.pyplot(fig)
                    
                    # ============================================================
                    # OUTLIERS (IQR POR TRATAMENTO)
                    # ============================================================
                    # ============================================================
                    # OUTLIERS (IQR POR TRATAMENTO) — VERSÃO FUNCIONAL
                    # ============================================================
                    st.subheader('Outliers (IQR por tratamento)')
                    
                    def limites_iqr(x):
                        Q1 = x.quantile(0.25)
                        Q3 = x.quantile(0.75)
                        IQR = Q3 - Q1
                        return pd.Series({
                            'LI': Q1 - 1.5 * IQR,
                            'LS': Q3 + 1.5 * IQR
                        })
                    
                    # calcula apenas os limites (SEM levar a coluna continua)
                    limites = (
                        data.groupby(categorica)[continua].agg(
                        LI=lambda x: x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25)),
                        LS=lambda x: x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25))
                    )
                    .reset_index()
                    )
                    
                    # merge seguro (continua NÃO some)
                    data_iqr = data.merge(limites, on=categorica, how='left')
                    
                    # marca outliers
                    data_iqr['outlier'] = (
                        (data_iqr[continua] < data_iqr['LI']) |
                        (data_iqr[continua] > data_iqr['LS'])
                    )
                    
                    st.write('Limites de outliers por tratamento')
                    st.dataframe(limites)
                    
                    outliers = data_iqr[data_iqr['outlier']]
                    
                    if outliers.empty:
                        st.success('Nenhum outlier identificado dentro dos tratamentos.')
                    else:
                        st.warning('Outliers identificados por tratamento:')
                        st.dataframe(outliers)

                    
                    # ============================================================
                    # DECISÃO DO USUÁRIO (ETAPA FUNDAMENTAL)
                    # ============================================================
                    escolha_3 = st.radio(
                        "Você deseja retirar os outliers?",
                        ["SIM", "Não"],
                        horizontal=True
                    )
                    
                    if escolha_3 == 'SIM':
                        data = (
                            data_iqr
                            .loc[~data_iqr['outlier']]
                            .drop(columns=['LI', 'LS', 'outlier'])
                        )
                    
                        st.success('Os outliers foram removidos com sucesso.')
                    
                        escolha_4 = st.radio(
                            "Você gostaria de ver os dados sem outliers?",
                            ['Sim', 'Não']
                        )
                    
                        if escolha_4 == 'Sim':
                            st.dataframe(data)
                    
                        escolha_5 = st.radio(
                            'Você deseja ver novamente os gráficos de Z-score?',
                            ['Sim', 'Não'],
                            horizontal=True
                        )
                    
                        if escolha_5 == 'Sim':
                            st.subheader('Z-score (dados sem outliers)')
                    
                            data['zscore'] = (
                                data
                                .groupby(categorica)[continua]
                                .transform(lambda x: (x - x.mean()) / x.std())
                            )
                    
                            fig2, ax = plt.subplots()
                            sns.kdeplot(data=data, x='zscore', fill=True, alpha=0.3)
                            ax.axvline(0, linestyle='--')
                            ax.set_title("Curva KDE do Z-score (sem outliers)")
                            st.pyplot(fig2)
                    
                            fig, ax = plt.subplots()
                            sns.boxplot(data=data, x='zscore', y=categorica, ax=ax)
                            sns.stripplot(data=data, x='zscore', y=categorica,
                                          color='black', alpha=0.5, ax=ax)
                            ax.set_title("Boxplot do Z-score (sem outliers)")
                            st.pyplot(fig)
                    
                    else:
                        st.info('Os outliers foram mantidos conforme decisão do usuário.')
                    
                    # ============================================================
                    # ANÁLISE DESCRITIVA FINAL (DEPENDE DA ESCOLHA)
                    # ============================================================
                    st.write('Análise descritiva dos seus dados')
                    
                    data_grouped = data.groupby(categorica)[continua].describe()
                    st.dataframe(data_grouped)
                    
                    cv = data[continua].values
                    cv2 = np.std(cv) / np.mean(cv) * 100
                    st.write(f"CV% = {cv2:.2f}")
                    
                    st.warning(
                        'Se quiser continuar a análise, clique na aba 2 acima '
                        '**Pressupostos da ANOVA**'
                    )


                #aqui
                with tab2:
                    st.header('Análise exploratória')
                    st.subheader('Gráfico boxplot')

                    Eixo_y = data.columns[1]
                    print(Eixo_y)
                    Axis_x = data.columns[0]



                    # colocando gráfico um ao lado do outro
                    col1, col2 = st.columns(2)

                    with tab2:
                        st.header('Análise exploratória')
                        st.subheader('Gráfico boxplot')



                        # colocando gráfico um ao lado do outro
                        col1, col2 = st.columns(2)

                        with col1:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.boxplot(x=Axis_x, y=Eixo_y,  palette="Set2", data=data, ax=ax)
                            sns.despine(offset=10, trim=True)
                            st.pyplot(fig)

                        with col2:
                            st.subheader('Gráfico de barras')
                            fig3, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(x=Axis_x, y=Eixo_y,  palette="Set2", errorbar='sd', width=0.5,
                                        data=data, ax=ax)
                            plt.ylim(0)
                            # sns.despine(offset=10, trim=True)
                            st.pyplot(fig3)

                        escolha_6 = st.radio('Você gostaria de alterar os gráficos?', ['Sim', 'Não '])
                        if escolha_6 == 'Sim':

                            escolha_7 = st.radio(f"Você gostaria de alterar o nível da variável categórica: {categorica}?",
                                                 ['Sim', 'Não'])

                            if escolha_7 == 'Sim':
                                data_grouped = data[categorica].unique()
                                lista = list(data_grouped)
                                tamanho = len(lista)
                                ordem_desejada = []
                                for k in range(tamanho):
                                    selecionado = st.selectbox(f'Escolha a ordem do nível {1+k} ', ['Selecione'] + lista,
                                                         key=f'ordem_{1+k}')
                                    ordem_desejada.append(selecionado )

                                    # Verifica se todos os níveis foram selecionados corretamente
                                    if 'Selecione' not in ordem_desejada and len(set(ordem_desejada)) == len(lista):
                                        nome_eixo_y = st.text_input("Digite o nome que você quer para o eixo Y:",
                                                                    value=Eixo_y)
                                        nome_eixo_x = st.text_input("Digite o nome que você quer para o eixo X:",
                                                                    value=Axis_x)
                                        # Criar um slider somente para valores máximos:
                                        max_valor = data[continua].max()
                                        valor_inicial = max_valor  # arredonda para o próximo inteiro

                                        ymax = st.number_input(
                                            label="Valor máximo do eixo Y (escala)",
                                            min_value=0.000000000,
                                            max_value=1000000.00,
                                            value=float(valor_inicial),
                                            step=0.01
                                        )

                                        font_opcao = ["serif",  "sans-serif",   "monospace",   "Arial", "Helvetica","Verdana" ,"Tahoma", "Calibri","DejaVu Sans","Geneva","Roboto","Times New Roman","Georgia","Garamond","Cambria","DejaVu Serif",
    "Computer Modern"]

                                        font1 = st.selectbox('Escolha a fonte dos eixos e rótulos', font_opcao, key = '87')

                                        options = ["Blues", "BuGn", "Set1", "Set2", "Set3", "viridis", "magma", "Pastel1",
                                                   "Pastel2", "colorblind", "Accent", "tab10", "tab20", "tab20b", 'tab20c',
                                                   "Paired"]

                                        cor_padrão = "Set2"
                                        cores = st.selectbox('Escolha a cor de interesse:', ['Cores'] + options, index=0)
                                        st.success(f"Você escolheu: {cores}.")
                                        if not cores:
                                            cores = 'Set2'
                                        if cores == 'Cores':
                                            cores = cor_padrão

                                        # Criar um slider somente para valores máximos:
                                        max_valor = data[continua].max()
                                        valor_inicial = max_valor  # arredonda para o próximo inteiro






                                        st.subheader('Gráfico de barras')
                                        fig3, ax = plt.subplots(figsize=(10, 6))
                                        sns.barplot(x=Axis_x, y=Eixo_y,  order=ordem_desejada, palette=cores,
                                                    errorbar='sd',
                                                    width=0.5,linewidth = 1, edgecolor = 'black', data=data, ax=ax)
                                        ax.set_ylabel(nome_eixo_y, fontsize=14, weight='bold')
                                        ax.set_xlabel(nome_eixo_x, fontsize=14, weight='bold')
                                        ax.set_ylim(0, ymax)#ax.spines['left'].set_linewidth(3)
                                        ax.set_ylim(0, ymax)  # ax.spines['left'].set_linewidth(3)
                                        cor = 'black'
                                        tom = 'bold'
                                        # Modificar as espinhas inferior e esquerda, colorindo-as
                                        # Esconder as espinhas superior e direita
                                        ax.spines['top'].set_visible(False)
                                        ax.spines['right'].set_visible(False)

                                        ax.spines['bottom'].set_linewidth(1)
                                        ax.spines['bottom'].set_color('black')
                                        ax.spines['left'].set_linewidth(1)
                                        ax.spines['left'].set_color('black')
                                        ax.tick_params(axis='y', labelsize=17, colors=cor)#tamanho dos números
                                        ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, fontweight='bold',
                                                           fontfamily=font1 )#tamangho das letras do rótulo
                                        ax.set_ylabel(nome_eixo_y, fontsize=18, weight='bold', family=font1 )#tamanho dos nomes das variáveis y
                                        ax.set_xlabel(nome_eixo_x, fontsize
                                        =18, weight='bold', family=font1 )#tamanho dos nomes das variáveis x

                                        # sns.despine(offset=10, trim=True)
                                        st.pyplot(fig3)

                                        # Salvar a figura em um arquivo PNG
                                        fig3.savefig(f"Gráfico de interação {categorica} e {continua}_barplot.png", dpi=300,
                                                      bbox_inches='tight')  # Salva a figura como .png

                                        # Cria um botão para download
                                        with open(f"Gráfico de interação {categorica} e {continua}_barplot.png", "rb") as f:
                                            st.download_button(
                                                label="Baixar o gráfico",  # Nome do botão
                                                data=f,  # Dados do arquivo
                                                file_name=f"Gráfico de interação {categorica} e {continua}_barplot.png",
                                                # Nome do arquivo a ser baixado
                                                mime="image/png"  # Tipo MIME do arquiv
                                            )

                                        data_grouped2 = data.groupby(categorica)[continua].describe().reset_index()
                                        st.dataframe(data_grouped2)

                                        boxgraph = st.radio('Você desejaria ver o gráfico em boxplot?', ['Sim', 'Não '])
                                        if boxgraph == 'Sim':

                                            # cores:
                                            options = ["Blues", "BuGn", "Set1", "Set2", "Set3",
                                                       "viridis",
                                                       "magma", "Pastel1", "Pastel2", "colorblind",
                                                       "Accent",
                                                       "tab10", "tab20", "tab20b", 'tab20c',
                                                       "Paired"]

                                            cor_padrão = "Set2"
                                            cores = st.selectbox('Escolha a cor de interesse:',
                                                                 ['Cores'] + options, index=1)
                                            st.success(f"Você escolheu: {cores}.")
                                            if cores == 'Cores':
                                                cores = cor_padrão

                                            # Criar um slider somente para valores máximos:
                                            max_valor = data[continua].max()
                                            valor_inicial = max_valor  # arredonda para o próximo inteiro

                                            ymax2 = st.number_input(
                                                label="Valor máximo do eixo Y",
                                                min_value=0.00000,
                                                max_value=1000000.0000,
                                                value=float(valor_inicial),
                                                step=0.0000000001,  # 10 casas decimais
                                                format="%.10f",  # mostra 10 casas decimais
                                                 key='not_123'
                                            )


                                            nome_eixo_y = st.text_input(
                                                "Digite o nome que você quer para o eixo Y:",
                                                value=Eixo_y, key='123a')
                                            nome_eixo_x = st.text_input(
                                                "Digite o nome que você quer para o eixo X:",
                                                value=Axis_x, key='125b')
                                            font_opcao = ["serif", "sans-serif", "monospace",
                                                          "Arial",
                                                          "Helvetica", "Verdana", "Tahoma",
                                                          "Calibri",
                                                          "DejaVu Sans", "Geneva", "Roboto",
                                                          "Times New Roman",
                                                          "Georgia", "Garamond", "Cambria",
                                                          "DejaVu Serif",
                                                          "Computer Modern"]

                                            font2 = st.selectbox(
                                                'Escolha a fonte dos eixos e rótulos',
                                                font_opcao, key='103')



                                            pre1 = ['Sim', 'Não ']
                                            prencher = st.selectbox('Você quer tirar  o preenchimento',
                                                                    ['Selecione'] + pre1, key = 'b_104')
                                            if prencher == 'Sim':
                                                val_pre = False
                                            elif prencher == 'Não':
                                                val_pre = True
                                            else:
                                                val_pre = True

                                            gap = st.slider('Escolha o gap entre os boxplots',
                                                            min_value=0.0, max_value=1.0, value=0.1,
                                                            step=0.01, key = 'gap1')
                                            width = st.slider('Espessura das caixas (width)', 0.2, 0.8,
                                                              value=0.5, step=0.05, key = 'keygap2')

                                            tamanho_texto_eixo = st.slider(
                                                "Tamanho dos textos ",
                                                min_value=1,
                                                max_value=32,
                                                value=16,
                                                step=1, key = '99porta'
                                            )

                                            st.header('Gráfico boxplot')

                                            fig23, ax = plt.subplots(figsize=(10, 6))

                                            sns.boxplot(
                                                x=Axis_x, y=Eixo_y,
                                                order=ordem_desejada,
                                                palette=cores,
                                                data=data,
                                                ax=ax,
                                                width=width,
                                                gap=gap,
                                                fill=val_pre,
                                                showfliers=False
                                            )

                                            # Eixos
                                            ax.set_ylabel(nome_eixo_y, fontsize=tamanho_texto_eixo, weight='bold',
                                                          family=font1)
                                            ax.set_xlabel(nome_eixo_x, fontsize=tamanho_texto_eixo, weight='bold',
                                                          family=font1)

                                            # Estilo
                                            ax.set_ylim(0, ymax2)
                                            sns.despine(offset=10, trim=True)
                                            ax.tick_params(axis='y', labelsize=tamanho_texto_eixo, colors=cor)
                                            ax.set_xticklabels(ax.get_xticklabels(), fontsize=tamanho_texto_eixo,
                                                               fontweight='bold', fontfamily=font1)

                                            st.pyplot(fig23)

                                            # Salvar a figura em um arquivo PNG
                                            fig23.savefig(f"Gráfico de interação {categorica} e {continua}.png", dpi=300,
                                                          bbox_inches='tight')  # Salva a figura como .png

                                            # Cria um botão para download
                                            with open(f"Gráfico de interação {categorica} e {continua}.png", "rb") as f:
                                                st.download_button(
                                                    label="Baixar o gráfico",  # Nome do botão
                                                    data=f,  # Dados do arquivo
                                                    file_name=f"Gráfico de interação {categorica} e {continua}.png",
                                                    # Nome do arquivo a ser baixado
                                                    mime="image/png"  # Tipo MIME do arquiv
                                                )
                                            st.success('Prossiga para a teceira página')
                with tab3:

                    st.header(f"Pressupostos da ANOVA ")
                    st.success(f'Modelo completo: {continua}~{categorica}')
                    st.success(f"Parâmetro: {continua}")
                    st.subheader('Teste de normalidade de Shapiro Wilk')
                    st.write('H0: Os resíduos seguem uma distribuição normal ')
                    st.write('Se P < 0.05, então rejeita H0 : Os resíduos não segue uma distribuição normal ')


                    formula = f"{continua} ~ {categorica}"


                    model = smf.ols(formula, data=data).fit()
                    df_resid = data.copy()
                    df_resid['Residuos2'] = model.resid
                    stat, p_valor = shapiro(df_resid['Residuos2'])
                    if p_valor > 0.05:
                        reject1 = 'Não rejeita a H0'
                        decisao = 'Os resíduos  seguem uma distribuição  normal '
                        st.success(f' P-valor =  {p_valor}')
                        st.success(f'Decisão {reject1}')
                        st.success(decisao)
                    else:
                        reject = 'Rejeita H0 '
                        resultando = 'Os resíduos não  seguem uma distribuição  normal '
                        st.success(f' P-valor =  {p_valor}')
                        st.success(f'Decisão {reject}')
                        st.success(resultando)
                    st.subheader('Curva de distribuição KDE')
                    # plotar a curva de KDE
                    fig5, ax = plt.subplots()
                    sns.kdeplot(data=df_resid, x='Residuos2', fill=True, alpha=0.3)
                    ax.set_title(f"Curva de KDE para visualização de normalidade do modelo {categorica}-{continua}")
                    plt.axvline(0, color='red', linestyle='dashed', linewidth=1)  # Linha central em 0
                    # sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
                    st.pyplot(fig5)

                    # Anderson Darling test
                    # Teste de normalidade de Anderson darling

                    st.header("Teste de Normalidade dos resíduos ")
                    st.subheader('Anderson Darling ')
                    st.write(f'H0: Os resíduos do modelo: {categorica}-{continua} seguem distribuição normal ')
                    st.write('H0: Se valor crítico > valor estatístico, então não rejeita H0')
                    test = anderson(df_resid['Residuos2'], dist='norm')
                    critical_value = test.critical_values[2]  # O valor crítico para o nível de 5%

                    if test.statistic > critical_value:
                        reject2 = 'Rejeita H0'
                        resultado = "Os resíduos não seguem uma distribuição normal "
                    else:
                        reject2 = 'Não rejeita H0'
                        resultado = 'Os resíduos seguem uma distribuição normal '

                    # Exibindo os resultados

                    st.success(f' Valor crítico: {critical_value} ')
                    st.success(f'Estatística do teste:  {test.statistic}')
                    st.success(reject2)
                    st.success(resultado)

                    # Homogneidade da variância:
                    st.header('Homogeneidade de variância')
                    st.subheader("Teste de levene")
                    st.write('H0: A variãncia dos grupos comparados são iguais a um nível de significância de 5%')
                    st.write('Se p-valor <0.05, então rejeita H0 e os resíduos não seguem distribuição normal')
                    agrupamento = df_resid.groupby(categorica)
                    grupo = []
                    for nome, dados_grupo in agrupamento:
                        # print(dados_grupo['Residuos'].values)
                        grupo.append(dados_grupo['Residuos2'].values)
                        # print(x)
                    stat, p_value = stats.levene(*grupo)
                    if p_value < 0.05:
                        reject = 'Rejeita a H0'
                        homoge_neo = 'não são '
                        resposta = 'Os resíduos não seguem uma distribuição normal'
                    else:
                        reject = 'Não rejeita H0'
                        homoge_neo = 'são '
                        resposta = 'Os resíduos seguem uma distribuição normal '
                    st.success(
                        f' P-valor :  {p_value}')
                    st.success(f"A variância dos níveis comparados {homoge_neo} homogêneos")
                    st.success(f'Decisão:  {reject} ')
                    st.success(resposta)

                    # teste de barlett
                    st.subheader('Teste de barlett para homogeneidade de variância')
                    st.write('H0: A variãncia dos grupos comparados são iguais a um nível de significância de 5%')
                    st.write('Se p-valor <0.05, então rejeita H0 e os resíduos não seguem distribuição normal')
                    stat, p = stats.bartlett(*grupo)
                    if p_value < 0.05:
                        reject = 'Rejeita a H0'
                        homoge_neo = 'não são '
                        decisao = 'Os resíduos não são homogênos(iguais)'
                    else:
                        reject = 'Não rejeita H0'
                        homoge_neo = 'são '
                        decisao = ' As variâncias dos resíduos são homogêneos '

                    st.success(f'P-valor :  {p_value}')
                    st.success(f'a variância dos níveis comparados {homoge_neo} homogêneos')
                    st.success(reject)
                    st.success(decisao)

                    st.subheader('Independência dos resíduos:')
                    st.write('H0: Os resíduos não são independentes (Não há autocorrelação)')
                    st.write('HA: Os resíduos são dependentes(Há correlação)')
                    st.write('Alfa = 0.05')
                    # Teste de Ljung-Box
                    lb_test = acorr_ljungbox(model.resid, lags=[1],
                                             return_df=True)  # lags=[1] testa apenas para defasagem 1

                    st.dataframe(lb_test)
                    p_valor = lb_test['lb_pvalue'].values[0]

                    if p_valor >=0.05:
                        st.success('Os resíduos não são independentes (Não há autocorrelação')
                    else:
                        st.warning('Os resíduos são dependentes (Há alta correlação)')




                    st.header('ANOVA')
                    model = smf.ols(formula, data=data).fit()
                    anova_table = anova_lm(model)
                    st.dataframe(anova_table)
                    st.write(f"R squared adjusted: {model.rsquared_adj}")
                    data_grouped2 = data.groupby(categorica)[continua].mean().reset_index()
                    st.dataframe(data_grouped2)

                    # pegar coluna de p-valor automaticamente
                    col_p = [c for c in anova_table.columns if 'pr(>f)' in c.lower()]

                    if not col_p:
                        st.error("Erro: coluna de p-valor não encontrada.")
                        st.write("Colunas disponíveis:", anova_table.columns)
                        st.stop()
                    
                    p_value = anova_table[col_p[0]].iloc[0]



                    if p_value < 0.05:
                        st.subheader(f'Análise de tukey para  X: {categorica} e y: {continua}')

                        categorico1 = pd.Categorical(data.iloc[:, 0]
                                                     )  # transformando a primeira coluna em categórica

                        mc = MultiComparison(data.iloc[:, 1], categorico1)
                        tukey_test1 = mc.tukeyhsd(alpha=0.05)
                        st.dataframe(tukey_test1.summary())
                        st.pyplot(fig3)
                    else:
                        st.warning(f' Seu p-valor {p_value} não foi significativo')
                        st.warning('Então não é feito o teste de tukey ')

                    # ------------------------------
                    # T-STUDENT ANALYSIS
                    # ------------------------------

                    st.header('Análise tratamentos pelo Teste t-Student')
                    st.write('H0: As médias dos tratamentos são iguais')
                    st.write('Se p-valor < 0.05 → rejeita-se H0 (diferenças significativas)')

                    # Pega os grupos da primeira coluna
                    grupos = data.iloc[:, 0].unique()

                    # Se houver apenas 2 grupos → t-test direto
                    if len(grupos) == 2:
                        g1 = data[data.iloc[:, 0] == grupos[0]].iloc[:, 1]
                        g2 = data[data.iloc[:, 0] == grupos[1]].iloc[:, 1]

                        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)

                        tabela_t = pd.DataFrame({
                            'Comparação': [f"{grupos[0]} vs {grupos[1]}"],
                            't-Stat': [t_stat],
                            'p-valor': [p_val]
                        })

                        st.dataframe(tabela_t)

                    # Se houver 3 ou mais grupos → comparar todos vs todos
                    else:
                        combinacoes = list(itertools.combinations(grupos, 2))
                        registros = []

                        for a, b in combinacoes:
                            g1 = data[data.iloc[:, 0] == a].iloc[:, 1]
                            g2 = data[data.iloc[:, 0] == b].iloc[:, 1]

                            t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
                            registros.append([f"{a} vs {b}", t_stat, p_val])

                        tabela_t = pd.DataFrame(registros, columns=["Comparação", "t-Stat", "p-valor"])
                        st.dataframe(tabela_t)

                    # ---------------------------------------
                    # PLOT: BOXPLOT + BRACKETS DE SIGNIFICÂNCIA
                    # ---------------------------------------

                    st.header('Gráfico com barras de significância (brackets)')

                    # Slider para largura
                    largura_final = st.slider(
                        'Escolha a largura do gráfico (figura)',
                        min_value=0.5,
                        max_value=15.0,
                        value=10.0,
                        step=0.05,
                        help="Largura total do gráfico em polegadas"
                    )

                    # Slider para altura
                    altura_final = st.slider(
                        'Escolha a altura do gráfico (figura)',
                        min_value=0.5,
                        max_value=15.0,
                        value=6.0,
                        step=0.05,
                        help="Altura total do gráfico em polegadas"
                    )

                    # Criar figura com tamanho ajustado pelo usuário
                    fig26, ax2 = plt.subplots(figsize=(largura_final, altura_final))

                    pb_opcao = st.checkbox(
                    "Destacar os dois primeiros tratamentos em branco e preto",
                    value=False,
                    key="pb_ttest_box"
                    )

                    n_box = len(ordem_desejada)

                    # paleta base escolhida pelo usuário
                    paleta_base = sns.color_palette(cores, n_colors=n_box)
                    
                    if pb_opcao and n_box >= 2:
                        paleta_final = ['white', 'black'] + paleta_base[2:]
                    else:
                        paleta_final = paleta_base

                    ativar_linha = st.checkbox(
                    "Adicionar linha separadora entre controle e tratamento?",
                    value=False
                    )

                     # ---- LINHA SEPARADORA (opcional) ----
                    if ativar_linha:
                        posicao_linha = st.number_input(
                            "Posição da linha (ex: 1.5 separa o 2º do 3º box)",
                            value=1.5,
                            step=0.1
                        )
                    
                        ax2.axvline(
                            x=posicao_linha,
                            color='black',
                            linestyle='--',
                            linewidth=1.2
                        )
                                               

                    ativar_pontos = st.checkbox(
                    "Adicionar pontos individuais (dados brutos)?",
                    value=False
                        )

                    if ativar_pontos:
                        jitter_pontos = st.slider(
                            "Dispersão horizontal dos pontos (jitter)",
                            min_value=0.0,
                            max_value=0.5,
                            value=0.15,
                            step=0.01
                        )
                    
                        tamanho_pontos = st.slider(
                            "Tamanho dos pontos",
                            min_value=1.0,
                            max_value=10.0,
                            value=4.0,
                            step=0.01
                        )

                        alpha_pontos = st.slider(
                        "Transparência dos pontos",
                        0.1, 1.0, 0.6, 0.05
                        )
                
                        cor_pontos = st.selectbox(
                            "Cor dos pontos",
                            options=[
                                "Preto",
                                "Cinza escuro",
                                "Cinza claro"
                                ],
                                index=1
                            )
                
                        mapa_cores = {
                                "Preto": "black",
                                "Cinza escuro": "#4D4D4D",
                                "Cinza claro": "#B0B0B0"
                            }

                    st.subheader("Posição do título do eixo X")

                    ativar_deslocamento_x = st.checkbox(
                        "Ajustar posição do título do eixo X?",
                        value=False
                     )

                    sns.boxplot(
                    x=Axis_x,
                    y=Eixo_y,
                    order=ordem_desejada,
                    palette=paleta_final,
                    data=data,
                    ax=ax2,
                    width=width,
                    gap=gap,
                    fill=val_pre,
                    showfliers=False,
                    
                        )


                    if pb_opcao:
                      # índice do box preto (segundo box)
                        idx = 1
                    
                        # dados do grupo preto
                        grupo_preto = ordem_desejada[idx]
                        dados = data[data[Axis_x] == grupo_preto][Eixo_y]
                    
                        # valor da mediana real
                        y_med = dados.median()
                    
                        # largura do box (mesma do seaborn)
                        meia_largura = width / 2
                    
                        # desenhar mediana branca POR CIMA
                        ax2.hlines(
                            y=y_med,
                            xmin=idx - meia_largura,
                            xmax=idx + meia_largura,
                            colors="white",
                            linewidth=0.9,
                            zorder=10  # MUITO IMPORTANTE
                        )
                    
                                        # pontos (APÓS o boxplot)
                    if ativar_pontos:
                        sns.stripplot(
                            x=Axis_x,
                            y=Eixo_y,
                            order=ordem_desejada,
                            data=data,
                            ax=ax2,
                            jitter=jitter_pontos,      # slider que você criou
                            size=tamanho_pontos, # slider que você criou
                            color=mapa_cores[cor_pontos],            # ou cinza escuro
                            alpha=alpha_pontos,              # transparência
                            dodge=False
                        )

                  
                                        
                                            

                    


                    if ativar_deslocamento_x:
                        deslocamento_x = st.slider(
                            "Deslocamento horizontal do título (0 = centro)",
                            min_value=-1.0,
                            max_value=1.0,
                            value=0.0,
                            step=0.01,
                            help="Valores positivos movem para a direita, negativos para a esquerda"
                        )
                    
                        deslocamento_y = st.slider(
                            "Altura do título do eixo X",
                            min_value=-0.5,
                            max_value=0.5,
                            value=-0.15,
                            step=0.01
                        )
                    else:
                        deslocamento_x = 0.5   # centro padrão do matplotlib
                        deslocamento_y = -0.15



                    st.subheader("Faixa separadora abaixo do eixo X")

                    ativar_faixa = st.checkbox(
                        "Adicionar faixa preta entre o título e os rótulos do eixo X?",
                        value=False
                     )

                    if ativar_faixa:
                        altura_faixa = st.slider(
                            "Altura da faixa",
                            min_value=0.005,
                            max_value=0.08,
                            value=0.02,
                            step=0.001
                        )
                    
                        posicao_faixa = st.slider(
                            "Posição vertical da faixa (negativo = abaixo do eixo)",
                            min_value=-0.3,
                            max_value=0.0,
                            value=-0.12,
                            step=0.01
                        )
                    
                        cor_faixa = st.selectbox(
                            "Cor da faixa",
                            ["Preto", "Cinza escuro"],
                            index=0
                        )
                    
                        mapa_cor_faixa = {
                            "Preto": "black",
                            "Cinza escuro": "#4D4D4D"
                        }

                        deslocamento_faixa_x = st.slider(
                        "Deslocamento horizontal da faixa",
                        min_value=-0.5,
                        max_value=0.5,
                        value=0.0,
                        step=0.01,
                        help="Valores positivos movem a faixa para a direita"
                        )
                    
                        largura_faixa_x = st.slider(
                            "Largura horizontal da faixa",
                            min_value=0.1,
                            max_value=1.0,
                            value=1.0,
                            step=0.01,
                            help="1.0 ocupa todo o eixo X"
                        )

                    if ativar_faixa:
                        x_faixa = 0.5 - largura_faixa_x / 2 + deslocamento_faixa_x

                        faixa = Rectangle(
                            (x_faixa, posicao_faixa),
                            largura_faixa_x,
                            altura_faixa,
                            transform=ax2.transAxes,
                            color=mapa_cor_faixa[cor_faixa],
                            clip_on=False
                            )
                        ax2.add_patch(faixa)

                                        

                    

                    # Ajustes de texto e eixos
                    ax2.set_ylabel(nome_eixo_y, fontsize=tamanho_texto_eixo, weight='bold', family=font1)
                    ax2.set_xlabel(
                        nome_eixo_x,
                        fontsize=tamanho_texto_eixo,
                        weight='bold',
                        family=font1
                    )

                    ax2.xaxis.set_label_coords(deslocamento_x, deslocamento_y)
                    ax2.set_ylim(0, ymax2)

                    # Criar caixa completa (sem despine!)
                    for spine in ax2.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)

                    ax2.tick_params(axis='y', labelsize=tamanho_texto_eixo, colors=cor)

                    ax2.set_xticklabels(
                        ax2.get_xticklabels(),
                        fontsize=tamanho_texto_eixo,
                        fontweight='bold',
                        fontfamily=font1
                    )

                    ax2.tick_params(axis='y', labelsize=tamanho_texto_eixo, colors=cor)

                    # --- aplicar estilo diretamente nos objetos Text dos rótulos do eixo X ---
                    #xticks = ax2.get_xticklabels()  # <<< CORRETO!
                    labels_x = [lbl.get_text() for lbl in ax2.get_xticklabels()]

                    # 2. Pegar os rótulos do eixo X
                    labels_x = [lbl.get_text() for lbl in ax2.get_xticklabels()]

                    # 3. Checkbox para aplicar itálico
                    italic_labels = st.checkbox("Aplicar itálico aos rótulos do eixo X?", value=False)
                    if italic_labels:
                        modo_italico = st.radio(
                            "Como aplicar o itálico?",
                            ["Todos os rótulos", "Selecionar rótulos específicos"]
                        )

                        if modo_italico == "Selecionar rótulos específicos":
                            labels_italico = st.multiselect(
                                "Quais rótulos devem ficar em itálico?",
                                options=labels_x,
                                default=[l for l in labels_x if l.lower() != "wt"]  # sugestão padrão
                            )

                        for lbl in ax2.get_xticklabels():
                            lbl.set_fontfamily(font1)
                            lbl.set_fontweight('bold')
                            lbl.set_fontsize(tamanho_texto_eixo)

                            if italic_labels:
                                if modo_italico == "Todos os rótulos":
                                    lbl.set_fontstyle('italic')
                                else:
                                    if lbl.get_text() in labels_italico:
                                        lbl.set_fontstyle('italic')
                                    else:
                                        lbl.set_fontstyle('normal')
                            else:
                                lbl.set_fontstyle('normal')

                    

                   

                    st.subheader("Adicionar símbolos por tratamento")

                    ativar_simbolo = st.checkbox(
                        "Adicionar símbolos de significância?",
                        value=False,
                        key="simbolo_manual"
                    )
                    
                    if ativar_simbolo:
                    
                        st.markdown("Defina o símbolo e a altura para cada grupo (deixe símbolo vazio para não aplicar):")
                    
                        configuracao_simbolos = {}
                    
                        for g in labels_x:
                            configuracao_simbolos[g] = {
                                "simbolo": st.text_input(
                                    f"Símbolo para {g}",
                                    value="",
                                    key=f"simbolo_{g}"
                                ),
                                "altura": st.number_input(
                                    f"Altura do símbolo para {g}",
                                    min_value=0.00000,
                                    value=float(data[Eixo_y].max() * 1.1),
                                    step=0.00001,
                                    format="%.6f",
                                    key=f"altura_{g}"
                                )
                            }
                    
                        tamanho_simbolo = st.slider(
                            "Tamanho do símbolo",
                            min_value=0.05,
                            max_value=30.00,
                            value=10.00
                        )
                    
                        # desenhar símbolos (um por grupo)
                        for grupo, cfg in configuracao_simbolos.items():
                            simbolo = cfg["simbolo"]
                            altura = cfg["altura"]
                    
                            if simbolo.strip() != "" and grupo in ordem_desejada:
                    
                                x = ordem_desejada.index(grupo)
                    
                                ax2.text(
                                    x,
                                    altura,
                                    simbolo,
                                    ha='center',
                                    va='bottom',
                                    fontsize=tamanho_simbolo,
                                    fontweight='bold'
                                )

                    st.subheader("Adicionar símbolos diferentes  por tratamento")

                    ativar_diferentes = st.checkbox(
                        "Adicionar símbolos de significância?",
                        value=False,
                        key="simbolo_manual2"
                    )
                    
                    if ativar_diferentes:
                    
                        st.markdown("Defina o símbolo e a alturas para cada grupo (deixe símbolo vazio para não aplicar):")
                    
                        configuracao_simbolos = {}
                    
                        for g in labels_x:
                            configuracao_simbolos[g] = {
                                "simbolo": st.text_input(
                                    f"Símbolo para {g}",
                                    value="",
                                    key=f"simbolodiferente_{g}"
                                ),
                                "altura": st.number_input(
                                    f"Altura do símbolo para {g}",
                                    min_value=0.00000,
                                    value=float(data[Eixo_y].max() * 1.1),
                                    step=0.00001,
                                    format="%.6f",
                                    key=f"alturadiferente_{g}"
                                )
                            }
                    
                        tamanho_simbolo = st.slider(
                            "Tamanho do símbolo",
                            min_value=0.05,
                            max_value=30.00,
                            value=10.00,
                            key="tamanho_simbolo"
                        )
                    
                        # desenhar símbolos (um por grupo)
                        for grupo, cfg in configuracao_simbolos.items():
                            simbolo = cfg["simbolo"]
                            altura = cfg["altura"]
                    
                            if simbolo.strip() != "" and grupo in ordem_desejada:
                    
                                x = ordem_desejada.index(grupo)
                    
                                ax2.text(
                                    x,
                                    altura,
                                    simbolo,
                                    ha='center',
                                    va='bottom',
                                    fontsize=tamanho_simbolo,
                                    fontweight='bold'
                                )




                    
                    

                    #fig26.tight_layout()

                    fig26.tight_layout(rect=[0, 0.08, 1, 1])
                    st.pyplot(fig26)



                    # Salvar a figura em um arquivo PNG
                    fig26.savefig(f"Gráfico de interação {categorica} e {continua}.png", dpi=300,
                                  bbox_inches='tight')  # Salva a figura como .png

                    # Cria um botão para download
                    with open(f"Gráfico de interação {categorica} e {continua}.png", "rb") as f:
                        st.download_button(
                            label="Baixar o gráfico",  # Nome do botão
                            data=f,  # Dados do arquivo
                            file_name=f"Gráfico de interação {categorica} e {continua}.png",
                            # Nome do arquivo a ser baixado
                            mime="image/png",  # Tipo MIME do arquiv
                        key = 'josetison'
                        )




        escolhas = []
        if variavel == 2:
            categorica= st.selectbox('Escolha a primeira variável  categórica',['Selecione'] + chaves1, key = '3')
            if categorica != 'Selecione':
                st.success(f"Você escolheu a variável categórica: {categorica}")

            categorica_2 = st.selectbox('Escolha a segunda variável  categórica',['Selecione'] +chaves1, key = '4')
            if categorica_2 != 'Selecione':
                st.success(f"Você escolheu a variável categórica: {categorica_2}")

            continua= st.selectbox('Escolha a variável contínua',['Selecione'] +chaves1, key = '5')
            if continua != 'Selecione':
                st.success(f"Você escolheu a variável contínua: {continua}")

            st.write('Você quer fazer modelo misto?')
            escolhas_misto= st.radio("Você deseja ?", ["Não", "Sim"])
            if escolhas_misto=="Sim":
               repeticao= st.selectbox('Escolha a variável repeticao',['Selecione'] +chaves1, key = '78_l')
               if repeticao!='Selecione':
                   
                   st.warning("Por favor, selecione a variável de repetição para prosseguir com o Modelo Misto.")
                   escolhas.append(repeticao)
                   escolhas.append(categorica)
                   escolhas.append(categorica_2)
                   escolhas.append(continua)
            else:
                escolhas.append(categorica)
                escolhas.append(categorica_2)
                escolhas.append(continua)
           
            if len(escolhas) >= 3:
                data = data[escolhas]  # escolhi e armazenei as variáveis que quero trabalhar
                st.write(data)
                data_na = data.isna().sum()
                # fazer uma função aqui! def retirarana(data na , data) e chamo novamente em outro luga
                if data_na.sum() == 0:
                    st.write(f'Você Não tem **NA** nas  variáveis de seus dados  ')
                    st.dataframe(data_na)
                else:
                    st.write(f'Você  tem **NA** nas  variáveis de seus dados  ')
                    st.dataframe(data_na)
                    st.write('Você gostaria de retirar  as **NAs** ou substituir por valores médios?')
                    escolha_2 = st.radio("Você deseja ?", ["Substituir por Valores médios", "Retirar Na"])
                    if escolha_2 == "Substituir por Valores médios":
                        data = data.fillna(data.median(numeric_only=True))
                        st.write('Dados com valores médios substituidos no lugar de NA')
                        st.dataframe(data)
                    else:
                        data = data.dropna(axis=1)
                        st.dataframe(data)  # manter o mes

            if categorica !='Selecione' and categorica_2 != 'Selecione' and continua!= 'Selecione':
                cv = data.loc[:, continua].values  # transforma em array numpy  e pega os valores, para o cálculo
                # cálculo do cv
                st.write('Análise descritiva dos seus dados ')
                data_grouped = data.groupby(data.columns[0:variavel].tolist()).describe()
                st.dataframe(data_grouped)
                cv2 = np.std(cv) / np.mean(cv) * 100
                st.write(f"CV% = {cv2}")

                st.subheader('Z-score ')
                zscore = (data[continua] - np.mean(data[continua])) / np.std(data[continua])
                data2 = data.copy()
                data2['zscore'] = zscore
                st.write(data2)
                # print(zscore)

                # plotar a curva de KDE
                fig2, ax = plt.subplots()
                sns.kdeplot(data=data2, x='zscore', fill=True, alpha=0.3)
                ax.set_title("Curva de KDE para visualização de normalidade ")
                plt.axvline(0, color='red', linestyle='dashed', linewidth=1)  # Linha central em 0
                # sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
                st.pyplot(fig2)

                # Plotar o boxplot dos z-scores
                fig, ax = plt.subplots()
                sns.boxplot(x=zscore, ax=ax)
                sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
                ax.set_title("Boxplot dos Z-Scores")
                st.pyplot(fig)

                # cálculo de outliers:
                st.subheader('Outlier ')
                st.write(
                    'O cálculo de outlier consiste em identificar os dados que estão acima ou abaixo  de 3 desvios padrão do Z-score e utiliza-se o método do IQR')

                Q1 = data.loc[:, continua].quantile(0.25)
                Q3 = data.loc[:, continua].quantile(0.75)
                # print(Q1)
                # print(Q3)

                IQR = Q3 - Q1
                LS = Q3 + 1.5 * IQR
                LI = Q1 - 1.5 * IQR
                print()
                linha = 70 * '='
                print(linha)

                print(linha)

                # Outliers acima e abaixo:
                st.write('Limite superior = ', LS)
                acima = data[(data.loc[:, continua] > LS)]
                if acima.empty:  # Usa-se empty, porque estamos tratando de um dataframe
                    st.write('Você não tem outliers acima do limite superior  ')
                    st.write(acima)
                else:
                    st.write('Você tem alguns outliers acima do limite superior')
                    st.write(acima)

                st.write("limite inferior = ", LI)
                abaixo = data[(data.loc[:, continua] < LI)]
                if abaixo.empty:  # Usa-se empty, porque estamos tratando de um dataframe
                    st.write('Você não tem outliers abaixo do limite inferior  ')
                    st.write(abaixo)
                else:
                    st.write('Você tem alguns outliers abaixo  do limite inferior')
                    st.write(abaixo)
                #decisão outlier
                escolha_3 = st.radio("Você deseja retirar os outliers ?", ["SIM", "Não "], horizontal=True)
                if escolha_3 == 'SIM':
                    data = data[(data[continua] < LS) & (data[continua] > LI)]
                    st.success('os outliers foram tirados com sucesso ')
                    st.write('Seus dados sem outliers')
                    st.dataframe(data)
                else:
                    st.warning('Não foram retirados outliers')
                        
                escolha_5 = st.radio('Você deseja ver os gráficos boxplot e KDE', ['Sim', 'Não'], horizontal=True)
                if escolha_5 == 'Sim':
                    st.subheader('Z-score ')
                    zscore = (data[continua] - np.mean(data[continua])) / np.std(data[continua])
                    data['zscore'] = zscore

                    # plotar a curva de KDE
                    fig2, ax = plt.subplots()
                    sns.kdeplot(data= data, x='zscore', fill=True, alpha=0.3)
                    ax.set_title("Curva de KDE para visualização de normalidade ")
                    plt.axvline(0, color='red', linestyle='dashed', linewidth=1)  # Linha central em 0
                    # sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
                    st.pyplot(fig2)

                    st.dataframe(data)
                    # Plotar o boxplot dos z-scores
                    fig, ax = plt.subplots()
                    sns.boxplot(x=zscore, ax=ax)
                    sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
                    ax.set_title("Boxplot dos Z-Scores")
                    st.pyplot(fig)

                    st.write('Análise descritiva dos seus dados ')
                    data_grouped = data.groupby(data.columns[0:variavel].tolist()).describe()
                    st.dataframe(data_grouped)
                    cv = data.loc[:, continua].values  # transforma em array numpy  e pega os valores, para o cálculo
                    # st.write(cv)
                    # cálculo do cv
                    cv2 = np.std(cv) / np.mean(cv) * 100
                    st.write(f"CV% = {cv2}")

                    with tab2:
                        st.header('Análise exploratória')
                        st.subheader('Gráfico boxplot')


                        Eixo_y = continua     # A variável numérica (SPAD, NPQT...)
                        Axis_x = categorica   # O Tratamento (Doses)
                        dentro_1 = categorica_2 # O fator Tempo (Dia)
                        #colocando gráfico um ao lado do outro
                        col1, col2 = st.columns(2)

                        with col1:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.boxplot(x=Axis_x, y=Eixo_y, hue=dentro_1, palette="Set2", data=data, ax=ax)
                            sns.despine(offset=10, trim=True)
                            st.pyplot(fig)

                        with col2:

                            st.subheader('Gráfico de barras')
                            fig3, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(x=Axis_x, y=Eixo_y, hue=dentro_1, palette="Set2",errorbar = 'sd', width = 0.5, data=data, ax=ax)
                            plt.ylim(0)
                            #sns.despine(offset=10, trim=True)
                            st.pyplot(fig3)



                        escolha_6 = st.radio('Você gostaria de alterar o gráfico ?', ['Sim', 'Não '])
                        if escolha_6 ==  'Não ':
                            st.warning('Escolha sim para prosseguir com  a análise dos fatores ')

                        else:

                            escolha_7 = st.radio( f"Você gostaria de alterar o nível da variável categórica: {categorica}?",['Sim', 'Não'])
                            if escolha_7 == 'Sim':
                                data_grouped = data[categorica].unique()
                                lista = list(data_grouped)
                                tamanho = len(lista)
                                ordem_desejada = []
                                for k in range(tamanho):
                                    selecionado = st.selectbox(f'Escolha a ordem do nível {1 + k} ',
                                                               ['Selecione'] + lista,
                                                               key=f'ordem1_{20 + k}')
                                    ordem_desejada.append(selecionado)

                                    # Verifica se todos os níveis foram selecionados corretamente
                                    if 'Selecione' not in ordem_desejada and len(set(ordem_desejada)) == len(lista):








                                        escolha_8 = st.radio(f"Você gostaria de alterar os níveis da variável categórica:{categorica_2}", ['Sim', 'Não'])
                                        if escolha_8 ==  'Sim':
                                            data_grouped = data[categorica_2].unique()

                                            lista = list(data_grouped)
                                            tamanho = len(lista)

                                            ordem_desejada2 = []
                                            for k in range(tamanho):
                                                selecionado = st.selectbox(f'Escolha a ordem do nível {1 + k} ',
                                                                           ['Selecione'] + lista,
                                                                           key=f'ordem2_{30 + k}')
                                                ordem_desejada2.append(selecionado)

                                                # Verifica se todos os níveis foram selecionados corretamente

                                            if 'Selecione' not in ordem_desejada and len(set(ordem_desejada2)) == len(lista):

                                                #cores:
                                                options = ["Blues", "BuGn", "Set1", "Set2", "Set3", "viridis",
                                                           "magma", "Pastel1", "Pastel2", "colorblind", "Accent",
                                                           "tab10", "tab20", "tab20b", 'tab20c', "Paired"]

                                                cor_padrão = "Set2"
                                                cores = st.selectbox('Escolha a cor de interesse:',
                                                                     ['Cores'] + options, index=0)
                                                st.success(f"Você escolheu: {cores}.")
                                                if cores == 'Cores':
                                                    cores = cor_padrão

                                                #Criar um slider somente para valores máximos:
                                                max_valor = data[continua].max()
                                                valor_inicial = max_valor  # arredonda para o próximo inteiro


                                                ymax = st.number_input(
                                                    label="Valor máximo do eixo Y",
                                                    min_value=0.00000,
                                                    max_value=1000000.00,
                                                    value=float(valor_inicial),
                                                    step=0.01
                                                )

                                                nome_eixo_y = st.text_input("Digite o nome que você quer para o eixo Y:",
                                                                            value=Eixo_y)
                                                nome_eixo_x = st.text_input("Digite o nome que você quer para o eixo X:", value = Axis_x)
                                                font_opcao = ["serif", "sans-serif", "monospace", "Arial",
                                                              "Helvetica", "Verdana", "Tahoma", "Calibri",
                                                              "DejaVu Sans", "Geneva", "Roboto", "Times New Roman",
                                                              "Georgia", "Garamond", "Cambria", "DejaVu Serif",
                                                              "Computer Modern"]

                                                font2 = st.selectbox('Escolha a fonte dos eixos e rótulos',
                                                                     font_opcao, key='88')

                                                with st.spinner("Por favor, aguarde..."):
                                                    st.subheader(f"Gráfico de interação  {categorica} e {categorica_2}")


                                                    #grráfico de barras e download
                                                    st.subheader(f'Gráfico de barras interação {categorica} e {categorica_2}')
                                                    fig2, ax = plt.subplots(figsize=(14, 8))
                                                    sns.barplot(x=Axis_x, y=Eixo_y, hue=dentro_1, order=ordem_desejada,
                                                                hue_order=ordem_desejada2, palette= cores,linewidth = 1, edgecolor = 'black', data=data,width = 0.5,  ax=ax,
                                                                errorbar='sd')

                                                    ax.set_ylim(0, ymax)#ax.spines['left'].set_linewidth(3)
                                                    cor = 'black'
                                                    tom = 'bold'
                                                    ax.spines['left'].set_linewidth(1)
                                                    ax.spines['left'].set_color(cor)
                                                    ax.tick_params(axis = 'y', labelsize = 17, colors = cor )
                                                    #ax.tick_params(axis = 'y', colors = cor )# cor do eixo y


                                                    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, fontweight='bold', fontfamily = font2)
                                                    ax.set_ylabel(nome_eixo_y, fontsize=18, weight='bold', family = font2)
                                                    ax.set_xlabel(nome_eixo_x, fontsize=18, weight='bold', family = font2)
                                                    plt.legend(title = categorica_2, frameon=False, prop={'weight': 'bold','size': 15,'family': font2},title_fontproperties={'weight': 'bold','size': 16,'family': font2})
                                                    plt.ylim(0)
                                                    st.pyplot(fig2)

                                                    # Salvar a figura em um arquivo PNG
                                                    fig2.savefig(f"Gráfico de interação barras {categorica} e {categorica_2}.png", dpi=300,
                                                                bbox_inches='tight')  # Salva a figura como .png

                                                    # Cria um botão para download
                                                    with open(f"Gráfico de interação barras {categorica} e {categorica_2}.png", "rb") as f:
                                                        st.download_button(
                                                            label="Baixar o gráfico",  # Nome do botão
                                                            data=f,  # Dados do arquivo
                                                            file_name=f"Gráfico de interação barras {categorica} e {categorica_2}.png",
                                                            # Nome do arquivo a ser baixado
                                                            mime="image/png"  # Tipo MIME do arquivo

                                                        )
                                                    data_grouped = data.groupby([categorica, categorica_2])[
                                                        continua].describe().reset_index()
                                                    st.subheader(
                                                        f'Análise das médias para a interação dos fatores  {categorica} e {categorica_2}')
                                                    st.dataframe(data_grouped)


                                                escolha_10 = st.radio('Você gostaria de ver os gráfico sem interação?',['Sim', 'Não'])
                                                if escolha_10 == 'Sim':
                                                    st.subheader(f'Gráfico {categorica_2} ')


                                                    #gráfico de barras:
                                                    st.subheader(f'Gráfico {categorica_2} ')
                                                    fig8, ax = plt.subplots(figsize=(14, 8))
                                                    sns.barplot (y=Eixo_y, hue=dentro_1,
                                                                hue_order=ordem_desejada2, palette=cores, linewidth = 1, edgecolor = 'black',width = 0.4, data=data, ax=ax)
                                                    ax.set_ylabel(nome_eixo_y, fontsize=14, weight='bold')
                                                    ax.set_ylim(0, ymax)  # ax.spines['left'].set_linewidth(3)
                                                    cor = 'black'
                                                    tom = 'bold'
                                                    ax.spines['left'].set_linewidth(1)
                                                    ax.spines['left'].set_color(cor)
                                                    ax.tick_params(axis='y', labelsize=17, colors=cor)
                                                    # ax.tick_params(axis = 'y', colors = cor )# cor do eixo y

                                                    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18,
                                                                       fontweight='bold', fontfamily=font2)
                                                    ax.set_ylabel(nome_eixo_y, fontsize=18, weight='bold',
                                                                  family=font2)
                                                    ax.set_xlabel(nome_eixo_x, fontsize=18, weight='bold',
                                                                  family=font2)

                                                    plt.ylim(0)
                                                    st.pyplot(fig8)

                                                    fig8.savefig(f"Gráfico de barras {categorica_2}.png", dpi=300,
                                                                bbox_inches='tight')  # Sem espaço antes de .png

                                                    with open(f"Gráfico de barras {categorica_2}.png", "rb") as f:
                                                        st.download_button(
                                                            label="Baixar o gráfico",
                                                            data=f,
                                                            file_name=f"Gráfico de barras {categorica_2}.png",
                                                            mime="image/png"

                                                        )

                                                        data_grouped2 = data.groupby(categorica_2)[continua].describe().reset_index()

                                                        st.subheader(f'Análise das médias para o fator {categorica_2}')
                                                        st.dataframe(data_grouped2)






                                                    st.subheader(f"Gráfico {categorica}")



                                                    #gráfico de barras:

                                                    st.subheader(f"Gráfico de barras {categorica}")

                                                    fig11, ax = plt.subplots(figsize=(14, 8))
                                                    sns.barplot(x=Axis_x, y=Eixo_y, order=ordem_desejada,
                                                                palette=cores,linewidth = 1, edgecolor = 'black',width = 0.4, data=data, ax=ax)
                                                    ax.set_ylabel(nome_eixo_y, fontsize=14, weight='bold')
                                                    ax.set_xlabel(nome_eixo_x, fontsize=14, weight='bold')
                                                    ax.set_ylim(0, ymax)  # ax.spines['left'].set_linewidth(3)
                                                    cor = 'black'
                                                    tom = 'bold'
                                                    ax.spines['left'].set_linewidth(1)
                                                    ax.spines['left'].set_color(cor)
                                                    ax.tick_params(axis='y', labelsize=17, colors=cor)
                                                    # ax.tick_params(axis = 'y', colors = cor )# cor do eixo y

                                                    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18,
                                                                       fontweight='bold', fontfamily=font2)
                                                    ax.set_ylabel(nome_eixo_y, fontsize=18, weight='bold',
                                                                  family=font2)
                                                    ax.set_xlabel(nome_eixo_x, fontsize=18, weight='bold',
                                                                  family=font2)

                                                    plt.ylim(0)
                                                    st.pyplot(fig11)


                                                    # Salvar a figura com nome seguro
                                                    fig11.savefig(f"Gráfico barras2 {categorica}.png", dpi=300, bbox_inches='tight')

                                                    # Botão de download
                                                    with open(f"Gráfico barras2 {categorica}.png", "rb") as f:
                                                        st.download_button(
                                                            label="Baixar o gráfico",
                                                            data=f,
                                                            file_name=f"Gráfico barras2 {categorica}.png",
                                                            mime="image/png"
                                                        )

                                                    data_grouped1 = data.groupby(categorica)[continua].describe().reset_index()

                                                    st.subheader(f'Análise das médias para o fator {categorica}')
                                                    st.dataframe(data_grouped1)
                                                    anova_data = st.radio('Você quer prosseguir o gráfico boxplot?',
                                                                          ['Sim', 'Não'], horizontal = True)

                                                    if anova_data == 'Sim':


                                                        # cores:
                                                        options = ["Blues", "BuGn", "Set1", "Set2", "Set3",
                                                                   "viridis",
                                                                   "magma", "Pastel1", "Pastel2", "colorblind",
                                                                   "Accent",
                                                                   "tab10", "tab20", "tab20b", 'tab20c',
                                                                   "Paired"]

                                                        cor_padrão = "Set2"
                                                        cores = st.selectbox('Escolha a cor de interesse:',
                                                                             ['Cores'] + options, index=1)
                                                        st.success(f"Você escolheu: {cores}.")
                                                        if cores == 'Cores':
                                                            cores = cor_padrão

                                                        # Criar um slider somente para valores máximos:
                                                        max_valor = data[continua].max()
                                                        valor_inicial = max_valor  # arredonda para o próximo inteiro

                                                        ymax = st.number_input(
                                                            label="Valor máximo do eixo Y",
                                                            min_value=0.000000,
                                                            max_value=1000000.00,
                                                            value=float(valor_inicial),
                                                            step=0.01, key = '122'
                                                        )

                                                        nome_eixo_y = st.text_input(
                                                            "Digite o nome que você quer para o eixo Y:",
                                                            value=Eixo_y, key = '123')
                                                        nome_eixo_x = st.text_input(
                                                            "Digite o nome que você quer para o eixo X:",
                                                            value=Axis_x,key = '124')
                                                        font_opcao = ["serif", "sans-serif", "monospace",
                                                                      "Arial",
                                                                      "Helvetica", "Verdana", "Tahoma",
                                                                      "Calibri",
                                                                      "DejaVu Sans", "Geneva", "Roboto",
                                                                      "Times New Roman",
                                                                      "Georgia", "Garamond", "Cambria",
                                                                      "DejaVu Serif",
                                                                      "Computer Modern"]

                                                        font2 = st.selectbox(
                                                            'Escolha a fonte dos eixos e rótulos',
                                                            font_opcao, key='102')

                                                        pre1 = ['Sim', 'Não ']
                                                        prencher = st.selectbox('Você quer tirar  o preenchimento', ['Selecione']+ pre1)
                                                        if prencher == 'Sim':
                                                            val_pre = False
                                                        elif prencher  == 'Não':
                                                            val_pre = True
                                                        else:
                                                            val_pre = True

                                                        gap = st.slider('Escolha o gap entre os boxplots',
                                                                        min_value=0.0, max_value=1.0, value=0.1,
                                                                        step=0.01)
                                                        width = st.slider('Espessura das caixas (width)', 0.2, 0.8,
                                                                          value=0.5, step=0.05)



                                                        fig20, ax = plt.subplots(figsize=(14, 8))
                                                        sns.set_theme(style="white")
                                                        sns.boxplot(x= Axis_x, y= Eixo_y, hue = dentro_1, order = ordem_desejada,
                                                                    hue_order = ordem_desejada2, palette = cores,
                                                                    fill = val_pre ,gap= gap,width = width, data = data )

                                                        ax.set_ylim(0, ymax)  # ax.spines['left'].set_linewidth(3)
                                                        cor = 'black'
                                                        tom = 'bold'
                                                        ax.spines['left'].set_linewidth(1)
                                                        ax.spines['left'].set_color(cor)
                                                        ax.tick_params(axis='y', labelsize=17, colors=cor)
                                                        # ax.tick_params(axis = 'y', colors = cor )# cor do eixo y

                                                        ax.set_xticklabels(ax.get_xticklabels(), fontsize=18,
                                                                           fontweight='bold', fontfamily=font2)
                                                        ax.set_ylabel(nome_eixo_y, fontsize=18, weight='bold',
                                                                      family=font2)
                                                        ax.set_xlabel(nome_eixo_x, fontsize=18, weight='bold',
                                                                      family=font2)
                                                        plt.legend(title=categorica_2, frameon=False,
                                                                   prop={'weight': 'bold', 'size': 12,
                                                                         'family': font2},
                                                                   title_fontproperties={'weight': 'bold',
                                                                                         'size': 11,
                                                                                         'family': font2})
                                                        plt.ylim(0)
                                                        st.pyplot(fig20)

                                                        # Salvar a figura com nome seguro
                                                        fig20.savefig(f"Gráfico boxplot {categorica}x{categorica_2}.png", dpi=300,
                                                                      bbox_inches='tight')

                                                        # Botão de download
                                                        with open(f"Gráfico boxplot {categorica}x{categorica_2}.png", "rb") as f:
                                                            st.download_button(
                                                                label="Baixar o gráfico",
                                                                data=f,
                                                                file_name=f"Gráfico boxplot {categorica}x{categorica_2}.png",
                                                                mime="image/png"
                                                            )




                                                        #gráfico das variáveis isoladas:
                                                        st.subheader(f' Boxplot   {categorica}')

                                                        fig25, ax = plt.subplots(figsize=(14, 8))
                                                        sns.set_theme(style="white")
                                                        sns.boxplot(x= Axis_x, y= Eixo_y,  order = ordem_desejada,
                                                                    palette = cores,
                                                                    fill = val_pre ,gap= gap,width = width, data = data )

                                                        ax.set_ylim(0, ymax)  # ax.spines['left'].set_linewidth(3)
                                                        cor = 'black'
                                                        tom = 'bold'
                                                        ax.spines['left'].set_linewidth(1)
                                                        ax.spines['left'].set_color(cor)
                                                        ax.tick_params(axis='y', labelsize=17, colors=cor)
                                                        # ax.tick_params(axis = 'y', colors = cor )# cor do eixo y

                                                        ax.set_xticklabels(ax.get_xticklabels(), fontsize=18,
                                                                           fontweight='bold', fontfamily=font2)
                                                        ax.set_ylabel(nome_eixo_y, fontsize=18, weight='bold',
                                                                      family=font2)
                                                        ax.set_xlabel(nome_eixo_x, fontsize=18, weight='bold',
                                                                      family=font2)

                                                        plt.ylim(0)
                                                        st.pyplot(fig25)

                                                        fig25.savefig(
                                                            f"Gráfico boxplot {categorica}.png",
                                                            dpi=300,
                                                            bbox_inches='tight')

                                                        # Botão de download
                                                        with open(
                                                                f"Gráfico boxplot {categorica}.png",
                                                                "rb") as f:
                                                            st.download_button(
                                                                label="Baixar o gráfico",
                                                                data=f,
                                                                file_name=f"Gráfico boxplot {categorica}.png",
                                                                mime="image/png"
                                                            )


                                                        #Gráfico das variáveis isoladas:

                                                        # gráfico das variáveis isoladas:
                                                        st.subheader(f' Boxplot  {categorica_2}')

                                                        fig260, ax = plt.subplots(figsize=(14, 8))
                                                        sns.set_theme(style="white")
                                                        sns.boxplot(x=dentro_1, y=Eixo_y,
                                                                    order= ordem_desejada2,
                                                                     palette=cores,
                                                                    fill=val_pre, gap=gap, width=width, data=data)

                                                        ax.set_ylim(0, ymax)  # ax.spines['left'].set_linewidth(3)
                                                        cor = 'black'
                                                        tom = 'bold'
                                                        ax.spines['left'].set_linewidth(1)
                                                        ax.spines['left'].set_color(cor)
                                                        ax.tick_params(axis='y', labelsize=17, colors=cor)
                                                        # ax.tick_params(axis = 'y', colors = cor )# cor do eixo y

                                                        ax.set_xticklabels(ax.get_xticklabels(), fontsize=18,
                                                                           fontweight='bold', fontfamily=font2)
                                                        ax.set_ylabel(nome_eixo_y, fontsize=18, weight='bold',
                                                                      family=font2)
                                                        ax.set_xlabel(nome_eixo_x, fontsize=18, weight='bold',
                                                                      family=font2)

                                                        plt.ylim(0)
                                                        st.pyplot(fig260)
                                                        fig260.savefig(
                                                            f"Gráfico boxplot {categorica_2}.png",
                                                            dpi=300,
                                                            bbox_inches='tight')

                                                        # Botão de download
                                                        with open(
                                                                f"Gráfico boxplot {categorica_2}.png",
                                                                "rb") as f:
                                                            st.download_button(
                                                                label="Baixar o gráfico",
                                                                data=f,
                                                                file_name=f"Gráfico boxplot {categorica_2}.png",
                                                                mime="image/png"
                                                            )

                                                        pontos = st.radio('Você deseja ver  o violinoplot??', ['Sim', 'Não'])
                                                        if pontos == 'Sim':
                                                            pre1 = ['Sim', 'Não ']
                                                            prencher = st.selectbox(
                                                                'Você quer tirar  o preenchimento',
                                                                ['Selecione'] + pre1, key = 'p_99')
                                                            if prencher == 'Sim':
                                                                val_pre = False
                                                            elif prencher == 'Não':
                                                                val_pre = True
                                                            else:
                                                                val_pre = True

                                                            fig21, ax = plt.subplots(figsize=(14, 8))

                                                            sns.violinplot(x=Axis_x, y= Eixo_y, hue = dentro_1, width=0.3, data = data, fill = val_pre)


                                                            ax.set_ylim(0,
                                                                        ymax)  # ax.spines['left'].set_linewidth(3)
                                                            cor = 'black'
                                                            tom = 'bold'
                                                            ax.spines['left'].set_linewidth(1)
                                                            ax.spines['left'].set_color(cor)
                                                            ax.tick_params(axis='y', labelsize=17, colors=cor)
                                                            # ax.tick_params(axis = 'y', colors = cor )# cor do eixo y

                                                            ax.set_xticklabels(ax.get_xticklabels(), fontsize=18,
                                                                               fontweight='bold', fontfamily=font2)
                                                            ax.set_ylabel(nome_eixo_y, fontsize=18, weight='bold',
                                                                          family=font2)
                                                            ax.set_xlabel(nome_eixo_x, fontsize=18, weight='bold',
                                                                          family=font2)
                                                            plt.legend(title=categorica_2, frameon=False,
                                                                       prop={'weight': 'bold', 'size': 12,
                                                                             'family': font2},
                                                                       title_fontproperties={'weight': 'bold',
                                                                                             'size': 11,
                                                                                             'family': font2})
                                                            plt.ylim(0)
                                                            st.pyplot(fig21)





                                            with tab3:
                                                st.header(f"Pressupostos da ANOVA ")
                                                st.success(f'Modelo completo: {categorica}:{categorica_2}')
                                                st.success(f"Parâmetro: {continua}")
                                                st.subheader('Teste de normalidade de Shapiro Wilk')
                                                st.write('H0: Os resíduos seguem uma distribuição normal ')
                                                st.write('Se P < 0.05, então rejeita H0 : O resíduos não segue uma distribuição normal ')



                                                formula = f'{continua}~{categorica_2}*{categorica}'
                                                # print(formula)
                                                # modelo
                                                model = smf.ols(formula, data= data).fit()
                                                df_resid = data.copy()
                                                df_resid['Residuos2'] = model.resid
                                                stat, p_valor = shapiro(df_resid['Residuos2'])
                                                if p_valor > 0.05:
                                                    reject = 'Não rejeita a H0'
                                                    decisao= 'Os resíduos  seguem uma distribuição  normal '
                                                    st.success(f' P-valor =  {p_valor}')
                                                    st.success(f'Decisão {reject}')
                                                    st.success(decisao )
                                                else:
                                                    reject = 'Rejeita H0 '
                                                    decisao = 'Os resíduos não  seguem uma distribuição  normal '
                                                    st.success(f' P-valor =  {p_valor}')
                                                    st.success(f'Decisão {reject}')
                                                    st.success(decisao)
                                                st.subheader('Curva de distribuição KDE')
                                                # plotar a curva de KDE
                                                fig5, ax = plt.subplots()
                                                sns.kdeplot(data=df_resid, x= 'Residuos2', fill=True, alpha=0.3)
                                                ax.set_title(f"Curva de KDE para visualização de normalidade do modelo {categorica}-{categorica_2}")
                                                plt.axvline(0, color='red', linestyle='dashed', linewidth=1)  # Linha central em 0
                                                # sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
                                                st.pyplot(fig5)

                                                # Anderson Darling test
                                                # Teste de normalidade de Anderson darling


                                                st.header("Teste de Normalidade dos resíduos ")
                                                st.subheader('Anderson Darling ')
                                                st.write(f'H0: Os resíduos do modelo: {categorica}-{categorica_2} seguem distribuição normal ')
                                                st.write('H0: Se valor crítico > valor estatístico, então não rejeita H0')
                                                test = anderson(df_resid['Residuos2'], dist='norm')
                                                critical_value = test.critical_values[2]  # O valor crítico para o nível de 5%

                                                if test.statistic > critical_value:
                                                    reject2 = 'Rejeita H0'
                                                    resultado = "Os resíduos não seguem uma distribuição normal "
                                                else:
                                                    reject2 = 'Não rejeita H0'
                                                    resultado = 'Os resíduos seguem uma distribuição normal '

                                                # Exibindo os resultados
                                                print(linha)
                                                st.success(f' Valor crítico: {critical_value} ')
                                                st.success(f'Estatística do teste:  {test.statistic}')
                                                st.success(reject2)
                                                st.success(resultado)

                                                #Homogneidade da variância:
                                                st.header('Homogeneidade de variância')
                                                st.subheader("Teste de levene")
                                                st.write('H0: A variãncia dos grupos comparados são iguais a um nível de significância de 5%')
                                                st.write('Se p-valor <0.05, então rejeita H0 e os resíduos não seguem distribuição normal')
                                                agrupamento = df_resid.groupby(categorica)
                                                grupo = []
                                                for nome, dados_grupo in agrupamento:
                                                    # print(dados_grupo['Residuos'].values)
                                                    grupo.append(dados_grupo['Residuos2'].values)
                                                    # print(x)
                                                stat, p_value = stats.levene(*grupo)
                                                if p_value < 0.05:
                                                    reject = 'Rejeita a H0'
                                                    homoge_neo = 'não são '
                                                    resposta = 'Os resíduos não seguem uma distribuição normal'
                                                else:
                                                    reject = 'Não rejeita H0'
                                                    homoge_neo = 'são '
                                                    resposta = 'Os resíduos seguem uma distribuição normal '
                                                st.success(
                                                    f' P-valor :  {p_value}' )
                                                st.success(f"A variância dos níveis comparados {homoge_neo} homogêneos")
                                                st.success(f'Decisão:  {reject} ')
                                                st.success(resposta)




                                                #teste de barlett
                                                st.subheader('Teste de barlett para homogeneidade de variância')
                                                st.write('H0: A variãncia dos grupos comparados são iguais a um nível de significância de 5%')
                                                st.write('Se p-valor <0.05, então rejeita H0 e os resíduos não seguem distribuição normal')
                                                stat, p = stats.bartlett(*grupo)
                                                if p_value < 0.05:
                                                    reject = 'Rejeita a H0'
                                                    homoge_neo = 'não são '
                                                    decisao = 'Os resíduos não são homogênos(iguais)'
                                                else:
                                                    reject = 'Não rejeita H0'
                                                    homoge_neo = 'são '
                                                    decisao = ' As variâncias dos resíduos são homogêneos '

                                                st.success(f'P-valor :  {p_value}')
                                                st.success(f'a variância dos níveis comparados {homoge_neo} homogêneos')
                                                st.success(reject)
                                                st.success(decisao)
                                                st.subheader('Independência dos resíduos:')
                                                st.write('H0: Os resíduos não são independentes(Não há correlação )')
                                                st.write('HA: Os resíduos são dependentes(Há correlação)')
                                                st.write('Se p<0.05, então rejeita H0: os resíduos são autocorrelacionados')

                                                # Teste de Ljung-Box
                                                lb_test = acorr_ljungbox(model.resid, lags=[1],
                                                                         return_df=True)  # lags=[1] testa apenas para defasagem 1

                                                st.dataframe(lb_test)
                                                p_valor = lb_test['lb_pvalue'].values[0]

                                                if p_valor >= 0.05:
                                                    st.success('Os resíduos não são  dependentes (Não há autocorrelação)')
                                                    st.success(p_valor)
                                                else:
                                                    st.warning('Os resíduos são dependentes (Há alta correlação)')
                                                    st.warning(f'p-valor = {p_valor}')

                                                st.header('ANOVA')
                                                model1 = smf.ols(formula, data=data).fit()
                                                anova_table = anova_lm(model1, typ=2)
                                                st.dataframe(anova_table)
                                                data_grouped = data.groupby([categorica, categorica_2])[continua].mean().reset_index()
                                                st.subheader(f'Análise das médias para a interação dos fatores  {categorica} e {categorica_2}')
                                                st.dataframe(data_grouped)
                                                st.write(f"R squared adjusted: {model.rsquared_adj}")
                                                p_value = anova_table['PR(>F)'][2]

                                                


                                                # O segredo está aqui: groups=data['Repeticao']
                                                # Isso diz que a medida é repetida no mesmo indivíduo
                                                
                                                
                                                # --- CASO MODELO MISTO ---
                                                if escolhas_misto == "Sim":
                                                    if repeticao != 'Selecione':
                                                        st.subheader(f"📊 Resultado: Modelo Misto (Efeito Aleatório: {repeticao})")
                                                        
                                                        # O segredo: groups=data[repeticao] 
                                                        # Ele isola a singularidade da planta que a gente conversou!
                                                        model_misto = smf.mixedlm(formula, data=data, groups=data[repeticao]).fit()
                                                        
                                                        # Exibe aquela tabela técnica detalhada
                                                        st.text(model_misto.summary())
                                                        st.write("#### P-valores simplificados")
                                                        st.dataframe(model_misto.pvalues)
                                                    else:
                                                        st.warning("Selecione a variável de repetição para calcular o modelo misto.")

                                                if p_value < 0.05:
                                                    print(f'Análise de tukey para o moddelo {categorica}: {categorica_2}')
                                                    df_clean2 = data.copy()
                                                    df_clean2['Combinação'] = df_clean2[categorica].astype(str) + ':' + df_clean2[
                                                        categorica_2].astype(str)
                                                    # Garantindo que a coluna Combinação seja categórica
                                                    df_clean2['Combinação'] = pd.Categorical(df_clean2['Combinação'])

                                                    mc = MultiComparison(df_clean2.iloc[:, 2], df_clean2['Combinação'])
                                                    tukey_test = mc.tukeyhsd(alpha=0.05)
                                                    st.dataframe(tukey_test.summary())
                                                    #gráfico
                                                    st.pyplot(fig2)

                                                else:
                                                    st.warning('O testde tukey não pode ser mostrado, pois não houve um p-valor significativo na interação')
                                                    st.warning(f'O p-valor foi de {p_value}')
                                                    st.warning('Que está acima de 0.05')
                                                    anova2 = st.radio('Você deseja fazer a análise dos fatores isolados?', ['Sim','Não'])

                                                    if anova2 == 'Sim':
                                                        st.header('Análise dos fatores isolados')
                                                        st.subheader(f'Modelo: {categorica} +{categorica_2}')
                                                        formula = f'{continua}~{categorica}+{categorica_2}'
                                                        model = smf.ols(formula, data=data).fit()
                                                        anova_table1 = anova_lm(model)
                                                        st.dataframe(anova_table1)
                                                        st.write(f"R squared adjusted: {model.rsquared_adj}")
                                                        p_value1 = anova_table['PR(>F)'][1]
                                                        p_value2 = anova_table['PR(>F)'][0]
                                                        data_grouped1 = data.groupby(categorica)[continua].mean().reset_index()

                                                        st.subheader(f'Análise das médias para o fator {categorica}')
                                                        st.dataframe(data_grouped1)
                                                        data_grouped2 = data.groupby(categorica_2)[continua].mean().reset_index()

                                                        st.subheader(f'Análise das médias para o fator {categorica_2}')
                                                        st.dataframe(data_grouped2)
                                                        if p_value1 < 0.05:
                                                            st.subheader(f'Análise de tukey para  o fator   {categorica}')

                                                            categorico1 = pd.Categorical(data.iloc[:,0]
                                                               )  # transformando a primeira coluna em categórica

                                                            mc = MultiComparison(data.iloc[:, 2], categorico1)
                                                            tukey_test1 = mc.tukeyhsd(alpha=0.05)
                                                            st.dataframe(tukey_test1.summary())
                                                            col2, col3 = st.columns(2)
                                                            with col2:
                                                                st.pyplot(fig11)
                                                            with col3:


                                                                data_grouped = data.groupby([categorica, categorica_2])[continua].mean().reset_index()


                                                        else:
                                                            st.warning(f'O valor de p para o fator {categorica} não foi significativo')
                                                            st.warning(p_value1)
                                                            st.warning('Não prossegue a análise de contraste')

                                                        if p_value2< 0.05:

                                                            st.subheader(f'Análise de tukey para  o fator  {categorica_2}')
                                                            categorico2 = pd.Categorical(data.iloc[:,1]
                                                               )  # transforma a segunda coluna em categórica

                                                            mc2= MultiComparison(data.iloc[:, 2], categorico2)
                                                            tukey_test2 = mc2.tukeyhsd(alpha=0.05)
                                                            st.dataframe(tukey_test2.summary())
                                                            cols = st.columns(2)  # Cria 3 colunas
                                                              # Pega a primeira coluna
                                                            col2 = cols[0]
                                                            col3 = cols[1]

                                                                


                                                        else:
                                                            st.warning(f'O valor de p para o fator {categorica_2} não foi significativo')
                                                            st.warning(p_value2)
                                                            st.warning('Não prossegue a análise de contraste')

        escolhas = []
        if variavel == 3:
            categorica= st.selectbox('Escolha as variável categórica',['Selecione'] + chaves1,key = '6')
            if categorica != 'Selecione':
                st.success(f"Você escolheu a variável categórica: {categorica}")
            categorica_2 = st.selectbox('Escolha as variável categórica', ['Selecione'] + chaves1, key = '7')
            if categorica_2 != 'Selecione':
                st.success(f"Você escolheu a variável categórica: {categorica_2}")
            categorica_3 = st.selectbox('Escolha as variável categórica', ['Selecione'] +chaves1, key='8')
            if categorica_3 != 'Selecione':
                st.success(f"Você escolheu a variável categórica: {categorica_3}")
            continua = st.selectbox('Escolha a variável contínua', ['Selecione'] +chaves1, key = '9')
            if continua  != 'Selecione':
                st.success(f"Você escolheu a variável contínua: {continua}")

            if categorica != 'Selecione' and  categorica_2 != 'Selecione' and categorica_3  and continua  != 'Selecione':

                escolhas.append(categorica)
                escolhas.append(categorica_2)
                escolhas.append(categorica_3)
                escolhas.append(continua)

                data = data[escolhas]  # escolhi e armazenei as variáveis que quero trabalhar
                st.write(data)
                data_na = data.isna().sum()
                # fazer uma função aqui! def retirarana(data na , data) e chamo novamente em outro lugar
                if data_na.sum() == 0:
                    st.write(f'Você Não tem **NA** nas  variáveis de seus dados  ')
                    st.dataframe(data_na)
                else:
                    st.write(f'Você  tem **NA** nas  variáveis de seus dados  ')
                    st.dataframe(data_na)
                    st.write('Você gostaria de retira  as **NAs** ou substituir por valores médios?')
                    escolha_2 = st.radio("Você deseja ?", ["Substituir por Valores médios", "Retirar Na"])
                    if escolha_2 == "Substituir por Valores médios":
                        data = data.fillna(data.median(numeric_only=True))
                        st.write('Dados com valores médios substituidos no lugar de NA')
                        st.dataframe(data)
                    else:
                        data = data.dropna(axis=1)
                        st.dataframe(data)  # manter o mes



                cv = data.loc[:, continua].values  # transforma em array numpy  e pega os valores, para o cálculo
                # cálculo do cv
                cv2 = np.std(cv) / np.mean(cv) * 100
                st.write(f"CV% = {cv2}")

                st.subheader('Z-score ')
                zscore = (data[continua] - np.mean(data[continua])) / np.std(data[continua])
                data2 = data.copy()
                data2['zscore'] = zscore
                st.write(data2)
                # print(zscore)

                # plotar a curva de KDE
                fig2, ax = plt.subplots()
                sns.kdeplot(data=data2, x='zscore', fill=True, alpha=0.3)
                ax.set_title("Curva de KDE para visualização de normalidade ")
                plt.axvline(0, color='red', linestyle='dashed', linewidth=1)  # Linha central em 0
                # sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
                st.pyplot(fig2)

                # Plotar o boxplot dos z-scores
                fig, ax = plt.subplots()
                sns.boxplot(x=zscore, ax=ax)
                sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
                ax.set_title("Boxplot dos Z-Scores")
                st.pyplot(fig)

                # cálculo de outliers:
                st.subheader('Outlier ')
                st.write(
                    'O cálculo de outlier consiste em identificar os dados que estão acima ou abaixo  de 3 desvios padrão do Z-score e utiliza-se o método do IQR')

                Q1 = data.loc[:, continua].quantile(0.25)
                Q3 = data.loc[:, continua].quantile(0.75)
                # print(Q1)
                # print(Q3)

                IQR = Q3 - Q1
                LS = Q3 + 1.5 * IQR
                LI = Q1 - 1.5 * IQR
                print()
                linha = 70 * '='
                print(linha)

                print(linha)

                # Outliers acima e abaixo:
                st.write('Limite superior = ', LS)
                acima = data[(data.loc[:, continua] > LS)]
                if acima.empty:  # Usa-se empty, porque estamos tratando de um dataframe
                    st.write('Você não tem outliers acima do limite superior  ')
                    st.write(acima)
                else:
                    st.write('Você tem alguns outliers acima do limite superior')
                    st.write(acima)

                st.write("limite inferior = ", LI)
                abaixo = data[(data.loc[:, continua] < LI)]
                if abaixo.empty:  # Usa-se empty, porque estamos tratando de um dataframe
                    st.write('Você não tem outliers abaixo do limite inferior  ')
                    st.write(abaixo)
                else:
                    st.write('Você tem alguns outliers abaixo  do limite inferior')
                    st.write(abaixo)

                escolha_3 = st.radio("Você deseja retirar os outliers ?", ["SIM", "Não "], horizontal=True)
                if escolha_3 == 'SIM':
                    data = data[(data[continua] < LS) & (data[continua] > LI)]
                    st.success('os outliers foram tirados com sucesso ')
                    escolha_4 = st.radio("Você gostaria de ver os dados sem outliers?", ['Sim', 'Não'])
                    if escolha_4 == 'Sim':
                        st.write('Seus dados sem outliers')
                        st.dataframe(data)
                    escolha_5 = st.radio('Você deseja ver os gráficos boxplot e KDE', ['Sim', 'Não'], horizontal=True)
                    if escolha_5 == 'Sim':
                        st.subheader('Z-score ')
                        zscore = (data[continua] - np.mean(data[continua])) / np.std(data[continua])
                        data['zscore'] = zscore

                        # plotar a curva de KDE
                        fig2, ax = plt.subplots()
                        sns.kdeplot(data=data, x='zscore', fill=True, alpha=0.3)
                        ax.set_title("Curva de KDE para visualização de normalidade ")
                        plt.axvline(0, color='red', linestyle='dashed', linewidth=1)  # Linha central em 0
                        # sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
                        st.pyplot(fig2)

                        st.dataframe(data)
                        # Plotar o boxplot dos z-scores
                        fig, ax = plt.subplots()
                        sns.boxplot(x=zscore, ax=ax)
                        sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
                        ax.set_title("Boxplot dos Z-Scores")
                        st.pyplot(fig)

                        st.write('Análise descritiva dos seus dados ')
                        data_grouped = data.groupby(data.columns[0:variavel].tolist()).describe()
                        st.dataframe(data_grouped)



























