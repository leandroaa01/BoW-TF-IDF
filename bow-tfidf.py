import math
import re
import csv
import os

CONTRACOES = [
    (r"won't", "will not"), (r"can't", "can not"), (r"shan't", "shall not"),
    (r"n't",    " not"),    (r"'re",  " are"),     (r"'ve",  " have"),
    (r"'ll",   " will"),    (r"'m",   " am"),      (r"'s",   ""),
]


def tokenizar(frase):
    frase = frase.lower()
    frase = re.sub(r"&\w+;", " ", frase)           # &amp; &quot; ...
    for padrao, sub in CONTRACOES:                  # don't -> do not
        frase = re.sub(padrao, sub, frase)
    frase = re.sub(r"[^a-z]", " ", frase)          # só letras
    return [t for t in frase.split() if len(t) >= 2]   # descarta 1 letra


def carregar_corpus(caminho):
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {os.path.abspath(caminho)}")

    corpus, rotulos, ignoradas = [], [], 0
    for enc in ("utf-8", "latin-1"):
        try:
            with open(caminho, encoding=enc) as f:
                linhas = f.readlines()
            break
        except UnicodeDecodeError:
            continue

    for linha in linhas:
        linha = linha.rstrip("\r\n")
        if "\t" not in linha:
            ignoradas += 1
            continue
        texto, rotulo = linha.rsplit("\t", 1)      # <-- separa no ÚLTIMO tab
        texto, rotulo = texto.strip(), rotulo.strip()
        if rotulo not in {"0", "1"} or not texto:
            ignoradas += 1
            continue
        corpus.append(texto)
        rotulos.append(int(rotulo))                # 0 = negativo, 1 = positivo

    print(f"[carregar] frases={len(corpus)}  ignoradas={ignoradas}")
    print(f"[classes]  positivos={sum(rotulos)}  negativos={len(rotulos)-sum(rotulos)}")
    return corpus, rotulos


def construir_matrizes(tokens_por_texto, min_df=2):
    freq_global = {}
    for tokens in tokens_por_texto:
        for palavra in tokens:
            freq_global[palavra] = freq_global.get(palavra, 0) + 1
    vocabulario = sorted(p for p, c in freq_global.items() if c >= min_df)

    # BoW
    bow = [
        {palavra: tokens.count(palavra) for palavra in vocabulario}
        for tokens in tokens_por_texto
    ]

    # DF e IDF suavizado
    N = len(tokens_por_texto)
    df = {p: sum(p in tokens for tokens in tokens_por_texto) for p in vocabulario}
    idf = {p: math.log((N + 1) / (df[p] + 1)) + 1 for p in vocabulario}

    # TF-IDF
    tf_idf = []
    for tokens, contagens in zip(tokens_por_texto, bow):
        total = len(tokens) if tokens else 1
        tf_idf.append({p: (contagens[p] / total) * idf[p] for p in vocabulario})

    print(f"[matrizes] N={N}  vocabulário={len(vocabulario)}")
    return vocabulario, bow, tf_idf


# =====================================================================
# Salvar CSV  ->  features + última coluna 'label' (0/1)
# =====================================================================
def salvar_csv(nome, matriz, vocabulario, rotulos):
    caminho = os.path.abspath(nome)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(vocabulario + ["label"])                # <-- cabeçalho
        for vetor, y in zip(matriz, rotulos):
            writer.writerow(
                [round(vetor[p], 6) for p in vocabulario] + [y]  # <-- rótulo no FIM
            )
    print(f"[salvar]   {caminho}  ({len(matriz)} linhas, {len(vocabulario)+1} colunas)")


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    corpus, rotulos = carregar_corpus("imdb_labelled.txt")
    tokens_por_texto = [tokenizar(frase) for frase in corpus]

    vocabulario, bow, tf_idf = construir_matrizes(tokens_por_texto, min_df=1)

    salvar_csv("Imdb_BOW.csv",    bow,    vocabulario, rotulos)
    salvar_csv("Imdb_TF-IDF.csv", tf_idf, vocabulario, rotulos)