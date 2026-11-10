import math
import re


corpus = [
    "gato gosta de leite",
    "cachorro gosta de passear",
    "gato e cachorro brincam",
    "estudante aprende Python na escola",
    "estudante estuda Python para aprender",
    "professora ensina Python na escola",
    "sol aquece a praia",
    "praia tem areia e sol",
    "chuva molha a rua",
    "chuva e vento chegaram",
]


def tokenizar(frase):
    # Retorna uma lista de palavras (tokens) extraídas da frase, convertidas para minúsculas.
    return re.findall(r"[a-záàâãéêíóôõúüç]+", frase.lower())

# Cria uma lista de listas de tokens para cada frase no corpus.
tokens_por_texto = [tokenizar(frase) for frase in corpus]
# Cria um vocabulário único de palavras presentes em todo o corpus, ordenado alfabeticamente.
vocabulario = sorted(
    {palavra for tokens in tokens_por_texto for palavra in tokens}
)

# BoW: conta quantas vezes cada palavra aparece em cada texto.
bow = []
for tokens in tokens_por_texto:
    # Cria um dicionário para cada texto, onde as chaves são as palavras do vocabulário e
    # os valores são as contagens dessas palavras no texto.
    bow.append({palavra: tokens.count(palavra) for palavra in vocabulario})

# DF: número de textos que contêm cada palavra.
df = {
    palavra: sum(palavra in tokens for tokens in tokens_por_texto)
    for palavra in vocabulario
}

# TF-IDF: TF relativo multiplicado pelo IDF suavizado.
tf_idf = []
numero_de_textos = len(corpus) #total de textos no corpus
for tokens, contagens in zip(tokens_por_texto, bow):
    pesos_documento = {} #
    for palavra in vocabulario:
        tf = contagens[palavra] / len(tokens) # TF relativo: frequência da palavra no texto dividido pelo total de palavras no texto
        idf = math.log((numero_de_textos + 1) / (df[palavra] + 1)) + 1 # log(N / (df + 1)) + 1, onde N é o número total de textos e df é o número de textos que contêm a palavra
        pesos_documento[palavra] = tf * idf  # coloca o peso TF-IDF da palavra no dicionário do documento
    tf_idf.append(pesos_documento) # cada documento recebe um dicionário com os pesos TF-IDF de cada palavra do vocabulário


# Exibe os documentos nas linhas e as palavras do vocabulário nas colunas.
def mostrar_matriz(titulo, matriz):
    print(f"\n{titulo}")
    tamanho_grupo = 7
    for inicio in range(0, len(vocabulario), tamanho_grupo):
        palavras = vocabulario[inicio:inicio + tamanho_grupo]
        cabecalho = ["Documento"] + palavras
        linhas = [
            [f"D{numero}"] + [
                str(round(vetor[palavra], 3)) for palavra in palavras
            ]
            for numero, vetor in enumerate(matriz, start=1)
        ]
        larguras = [
            max(len(cabecalho[coluna]), *(len(linha[coluna]) for linha in linhas))
            for coluna in range(len(cabecalho))
        ]
        borda = "+-" + "-+-".join("-" * largura for largura in larguras) + "-+"

        print(borda)
        print("| " + " | ".join(
            cabecalho[coluna].ljust(larguras[coluna])
            for coluna in range(len(cabecalho))
        ) + " |")
        print(borda)
        for linha in linhas:
            print("| " + " | ".join(
                linha[coluna].ljust(larguras[coluna])
                for coluna in range(len(linha))
            ) + " |")
        print(borda)


print("Corpus:")
for num_texto, documento in enumerate(corpus, start=1):
    print(f"Texto {num_texto}: {documento}")

print("\nVocabulário:", vocabulario)
print("\nFórmulas utilizadas:")
print("BoW(t, d) = quantidade de vezes que o termo t aparece no texto d")
print("TF(t, d) = quantidade de t em d / total de palavras em d")
print("IDF(t) = ln((N + 1) / (DF(t) + 1)) + 1")
print("TF-IDF(t, d) = TF(t, d) * IDF(t)")
print("N = número de textos no corpus; DF(t) = número de textos que contêm t")
mostrar_matriz("BoW (contagem de palavras):", bow)
mostrar_matriz("TF-IDF:", tf_idf)
