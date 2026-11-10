# Representação de Texto Estruturada: Bag of Words (BoW) e TF-IDF

## 🎯 Objetivos da aula

* Compreender a importância e a necessidade de transformar texto não estruturado em representações vetoriais estruturadas para algoritmos de IA [1].
* Dominar as etapas essenciais de pré-processamento de texto (tokenização, normalização, remoção de stopwords, stemming e lematização) que antecedem a vetorização [2, 3].
* Entender a mecânica do modelo **Bag of Words (BoW)** e suas variações (binário, contagem/frequência absoluta e frequência relativa/normalizada) [4, 5].
* Construir e interpretar uma **Matriz Documento-Termo** [4].
* Identificar as capacidades e limitações severas da abordagem BoW (perda de ordem, sintaxe, contexto e negação) [4, 6].
* Dominar a formulação matemática do **TF-IDF (Term Frequency – Inverse Document Frequency)** e compreender como ele penaliza termos muito comuns e valoriza termos discriminativos [7, 8, 9].
* Conhecer os principais modelos de Machine Learning (Naive Bayes, Regressão Logística, SVM, KNN, K-Means) que utilizam essas representações para classificação, busca e agrupamento [11].
* Resolver exercícios práticos de construção de vocabulário, matrizes BoW e cálculos de TF-IDF [12, 13].

---

## 📚 Conteúdo principal

### 1. Necessidade da Representação Vetorial de Textos

* **Conceito:** A representação de texto é o processo de converter dados textuais não estruturados (como frases, notícias ou e-mails) em vetores numéricos estruturados que podem ser processados por algoritmos de Aprendizado de Máquina [1].
* **Explicação:** Modelos estatísticos e de IA não conseguem ler ou processar diretamente palavras em linguagem natural. Para que um classificador ou algoritmo de agrupamento opere sobre o texto, cada documento deve ser traduzido em um conjunto fixo de características (características numéricas). A representação vetorial define quais aspectos do texto ficam visíveis para o modelo — como a presença, a frequência ou a relevância relativa de cada termo [1, 2].
* **Exemplo:** A frase *"O filme foi excelente!"* é transformada em um vetor numérico como `[0, 1, 2, 0, ...]` com base num vocabulário pré-definido, onde cada número representa a presença ou a contagem de uma palavra específica [1].
* **Importante:** Cada documento do corpus passa a ser representado por um vetor numérico que possui exatamente o mesmo número de dimensões (igual ao tamanho do vocabulário do corpus) [2].

---

### 2. O Pipeline de Processamento de Linguagem Natural (PLN)

* **Conceito:** É a sequência organizada de etapas por onde o texto bruto passa até ser convertido em um resultado final útil para o modelo [1].
* **Explicação:** O pipeline padrão é estruturado nas seguintes fases consecutivas:
  1. **Texto Bruto:** Entrada original sem tratamento (ex: frases, e-mails, posts) [1].
  2. **Pré-processamento:** Limpeza e padronização dos tokens [1, 2].
  3. **Vetorização (BoW / TF-IDF):** Conversão dos tokens limpos em vetores numéricos [1].
  4. **Modelo de IA:** Algoritmo que processa os vetores (ex: Classificador Naive Bayes, SVM) [1, 11].
  5. **Resultado:** Saída desejada (ex: rótulo de sentimento positivo/negativo, classe de spam) [1, 11].
* **Exemplo:** Entrada `Texto bruto` → Limpeza/Tokenização → Matriz Numérica → Modelo Naive Bayes → Previsão de `Spam / Não Spam` [1, 11].
* **Importante:** As escolhas feitas na etapa de pré-processamento impactam diretamente a qualidade da matriz vetorial gerada e o desempenho do modelo de Machine Learning [11].

---

### 3. Etapas de Pré-processamento de Texto

* **Conceito:** Conjunto de técnicas de limpeza e tratamento aplicadas ao texto antes da extração de características para reduzir o ruído e a dimensão do vocabulário [2, 3].
* **Explicação:** 
  * **Tokenização:** Processo de dividir uma cadeia de texto contínua em unidades menores chamadas *tokens* (geralmente palavras, mas podem ser subpalavras, caracteres ou n-gramas) [2, 3].
  * **Normalização:** Padronização do texto, como conversão de todas as letras para minúsculas (caixa baixa) e remoção opcional de acentuação para evitar que `"Gato"`, `"GATO"` e `"gato"` sejam tratados como termos distintos [2].
  * **Remoção de Pontuação e Stopwords:** Eliminação de pontuações e de palavras extremamente frequentes que possuem pouca carga semântica própria (como artigos e preposições: *"de"*, *"a"*, *"o"*, *"e"*) [3].
  * **Stemming e Lematização:** Redução das palavras às suas formas radicais ou canônicas (ex: `"brincavam"`, `"brincando"` → `"brincar"`) [3].
  * **Preservação de Informações Críticas:** Manutenção de marcadores essenciais para a tarefa, como palavras de negação (*"não"*, *"nunca"*), que modificam drasticamente o sentido [3, 6].
* **Exemplo:** O texto bruto `"Gatos, GATO e gatos!"` após tokenização e normalização de caixa resulta na lista de tokens `[gato, gato, gato]` [2, 3].
* **Importante:** Remover indiscriminadamente stopwords como a palavra *"não"* em tarefas de análise de sentimento pode destruir o significado do texto (ex: transformar *"não gostei"* em *"gostei"*) [6].

---

### 4. Modelo Bag of Words (BoW) — Sacola de Palavras

* **Conceito:** É uma técnica de representação em que o documento é descrito pela presença ou frequência das palavras que nele ocorrem, ignorando completamente a ordem e a estrutura sintática [4].
* **Explicação:** No modelo BoW, o texto é tratado como se todas as suas palavras fossem despejadas dentro de uma "sacola" [4]. O que importa para o modelo é **quais termos aparecem** e **quantas vezes aparecem** [4]. 
* **Exemplo:**
  * Frase A: *"cão morde homem"* → Vetor BoW: `[cão: 1, morde: 1, homem: 1]` [4].
  * Frase B: *"homem morde cão"* → Vetor BoW: `[cão: 1, morde: 1, homem: 1]` [4].
  * Embora o significado das duas frases seja diferente na vida real, no modelo BoW elas possuem **exatamente o mesmo vetor** [4].
* **Importante:** A principal limitação do BoW é que ele **descarta a ordem das palavras e as relações sintáticas** [4, 6].

---

### 5. Tipos de Representação no BoW

* **Conceito:** O modelo BoW pode codificar o peso de cada palavra no vetor de três formas distintas: Binário, Contagem (Frequência Absoluta) e Frequência Relativa (Normalizada) [5].
* **Explicação:**
  1. **BoW Binário:** Registra apenas a **presença (1)** ou **ausência (0)** do termo no documento, ignorando quantas vezes ele se repetiu [5].
  2. **BoW por Contagem (Frequência Absoluta):** Registra o número exato de vezes que o termo ocorre dentro daquele documento específico [4, 5].
  3. **BoW por Frequência Relativa (Normalizada):** Divide a contagem do termo pelo número total de tokens do documento, permitindo comparar documentos de tamanhos diferentes [5].
* **Exemplo (em um documento com 5 tokens onde a palavra "gato" aparece 2 vezes):**
  * Binário: `1` [5].
  * Contagem: `2` [5].
  * Frequência Relativa: `2 / 5 = 0,40` (40% dos tokens) [5].
* **Importante:** A frequência absoluta registra a repetição, mas não considera o tamanho do documento (uma palavra que aparece 5 vezes em um tweet de 10 palavras tem peso muito diferente do que em um livro de 100.000 palavras) [5].

---

### 6. Matriz Documento-Termo (Document-Term Matrix)

* **Conceito:** É a estrutura matricial em que cada linha representa um documento do corpus e cada coluna representa um termo único do vocabulário [2, 4].
* **Explicação:** O vocabulário é o conjunto ordenado de todos os termos distintos presentes em todo o corpus [2, 3]. Cada coluna recebe uma posição fixa no vetor [3]. O valor na interseção da linha $i$ com a coluna $j$ indica a contagem, presença ou peso do termo $j$ no documento $i$ [4].

#### Exemplo de Matriz Documento-Termo (Frequência Absoluta):
Considerando o corpus:
* **D1:** *"gato gosta de leite"*
* **D2:** *"cachorro gosta de carne"*
* **D3:** *"gato e cachorro brincam"*

| Documento | gato | gosta | de | leite | cachorro | carne | e | brincam |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D1** | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| **D2** | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 0 |
| **D3** | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 1 |

* **Importante:** Essa matriz é geralmente muito **esparsa** (contém uma grande quantidade de zeros), pois um único documento utiliza apenas uma pequena fração de todo o vocabulário do corpus [11].

---

### 7. O que o BoW Captura vs. O que ele Perde

* **Conceito:** O balanço entre os aspectos informativos preservados e os perdidos na simplificação Bag of Words [6].

* **O que o BoW Captura:**
  * Presença de termos [6].
  * Frequência local dos termos [6].
  * Padrões lexicais [6].
  * Pistas de assunto/tópico [6].

* **O que o BoW NÃO Captura (Limitações):**
  * Ordem das palavras [6, 11].
  * Estrutura sintática e relações gramaticais [6, 11].
  * Sentido contextual e negação (ex: *"não gostei"* vira o mesmo que *"gostei"* se a negação for desconsiderada) [6, 11].
  * Sinônimos e paráfrases (ex: *"carro"* e *"automóvel"* são tratados como dimensões totalmente diferentes) [11].
  * Polissemia (a mesma palavra com sentidos diferentes em contextos distintos é mapeada no mesmo elemento do vetor) [11].

---

### 8. TF-IDF (Term Frequency – Inverse Document Frequency)

* **Conceito:** É uma técnica de ponderação numérica que combina a frequência local de um termo em um documento com a sua raridade em todo o corpus, penalizando termos comuns e destacando termos discriminativos [7, 10].
* **Explicação:** No modelo BoW simples, palavras muito frequentes como preposições e artigos (*"de"*, *"a"*, *"e"*) recebem contagens altas, embora ajudem muito pouco a diferenciar os assuntos dos documentos [6]. O TF-IDF resolve isso combinando duas perguntas [7]:
  1. **TF (Term Frequency):** Com que frequência o termo aparece neste documento específico? (Mede a relevância local) [7].
  2. **IDF (Inverse Document Frequency):** Em quantos documentos do corpus este termo aparece? (Mede a raridade global) [7, 8].
* **Exemplo:** A palavra *"de"* aparece em quase todos os documentos de uma biblioteca (IDF muito baixo), portanto seu peso TF-IDF será próximo de zero. Já a palavra *"fotossíntese"* aparece em poucos documentos de biologia (IDF alto), recebendo um peso elevado nesses documentos [6, 9].
* **Importante:** Se um termo é frequente dentro de um documento específico (TF alto) E é raro no corpus geral (IDF alto), ele terá um peso **TF-IDF elevado**, tornando-se altamente discriminativo [9, 10, 13].

---

### 9. Fórmulas Matemáticas do TF-IDF e Legenda dos Termos

#### 1. Frequência do Termo — $TF(t, d)$
Existem duas formas principais de calcular o $TF$ [7]:
* **TF Absoluto:**
  $$\text{TF}(t, d) = f_{t, d}$$
* **TF Relativo (Normalizado):**
  $$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$

**Legenda:**
* $t$: O termo (palavra) sendo avaliado [7].
* $d$: O documento específico sob análise [7].
* $f_{t, d}$: Número de ocorrências (contagem absoluta) do termo $t$ dentro do documento $d$ [7].
* $\sum_{t' \in d} f_{t', d}$: Número total de tokens presentes no documento $d$ [7].

---

#### 2. Frequência Inversa do Documento — $IDF(t)$
Fórmula comum suavizada (usando logaritmo natural $\ln$) [7, 8]:

$$\text{IDF}(t) = \ln\left( \frac{N + 1}{\text{DF}(t) + 1} \right) + 1$$

**Legenda:**
* $\text{IDF}(t)$: Frequência Inversa do Documento para o termo $t$ [7].
* $N$: Número total de documentos existentes no corpus [7, 8].
* $\text{DF}(t)$: *Document Frequency* — número de documentos no corpus que contêm o termo $t$ pelo menos uma vez [7, 8].
* $+ 1$: Fator de suavização (*smoothing*) adicionado ao numerador e ao denominador para evitar divisão por zero quando $\text{DF}(t) = 0$ e para garantir que o resultado dentro do logaritmo seja sempre $\ge 1$ [7, 8].
* $+ 1$ (externo): Garante que termos presentes em todos os documentos não recebam peso zero absoluto [7, 8].

---

#### 3. Peso Final — $\text{TF-IDF}(t, d)$

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

**Legenda:**
* $\text{TF-IDF}(t, d)$: Peso final atribuído ao termo $t$ no documento $d$ [7, 9].
* Multiplicação da relevância local ($\text{TF}$) pelo poder discriminativo global ($\text{IDF}$) [7, 9, 13].

---

### 10. Aplicações Práticas e Modelos de Machine Learning

* **Conceito:** Áreas da computação e algoritmos de IA que se beneficiam de representações BoW e TF-IDF [10, 11].
* **Explicação:**
  * **Classificação de Textos:** Análise de sentimento, detecção de spam, categorização temática de notícias [11].
  * **Busca Textual e Recuperação de Informação:** Ranking de documentos com base na relevância de consultas de busca (onde o TF-IDF é amplamente utilizado) [10, 11].
  * **Agrupamento (Clustering):** Organização automática de documentos em grupos com base na similaridade do vocabulário [11].
* **Modelos Comuns de IA Utilizados:**
  * Naive Bayes [11].
  * Regressão Logística [11].
  * Perceptron Multicamadas (MLP) [11].
  * Máquinas de Vetores de Suporte (SVM) [11].
  * k-Nearest Neighbors (KNN) e K-Means (com cuidados na escolha da métrica de distância devido à alta dimensão) [11].

---

## ✏️ Exercícios Resolvidos e Desafio (Slides 12 e 13)

### Questão Prática

Considere o seguinte corpus contendo $N = 3$ documentos [12]:
* **D1:** *"gosto de inteligência artificial"*
* **D2:** *"inteligência artificial transforma a educação"*
* **D3:** *"não gosto de educação tradicional"*

---

#### 1. Defina o vocabulário e construa a matriz BoW por contagem [12].

**Resolução:**

* **Passo A — Extração dos Tokens por Documento:**
  * D1: `[gosto, de, inteligência, artificial]` (4 tokens) [12]
  * D2: `[inteligência, artificial, transforma, a, educação]` (5 tokens) [12]
  * D3: `[não, gosto, de, educação, tradicional]` (5 tokens) [12]

* **Passo B — Vocabulário Ordenado (10 termos únicos):**
  `[a, artificial, de, educação, gosto, inteligência, não, tradicional, transforma]` (Atenção: 9 termos no total).

* **Matriz Documento-Termo BoW por Contagem (Frequência Absoluta):**

| Documento | a | artificial | de | educação | gosto | inteligência | não | tradicional | transforma | Total Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D1** | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | **4** |
| **D2** | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | **5** |
| **D3** | 0 | 0 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | **5** |

---

#### 2. Calcule DF e IDF; em D1, calcule o TF-IDF de dois termos [12].

**Resolução:**

* **Cálculo da Frequência do Documento $\text{DF}(t)$ para cada termo ($N = 3$):**
  * `a`: aparece em D2 $\implies \text{DF} = 1$ [12]
  * `artificial`: aparece em D1, D2 $\implies \text{DF} = 2$ [12]
  * `de`: aparece em D1, D3 $\implies \text{DF} = 2$ [12]
  * `educação`: aparece em D2, D3 $\implies \text{DF} = 2$ [12]
  * `gosto`: aparece em D1, D3 $\implies \text{DF} = 2$ [12]
  * `inteligência`: aparece em D1, D2 $\implies \text{DF} = 2$ [12]
  * `não`: aparece em D3 $\implies \text{DF} = 1$ [12]
  * `tradicional`: aparece em D3 $\implies \text{DF} = 1$ [12]
  * `transforma`: aparece em D2 $\implies \text{DF} = 1$ [12]

* **Cálculo do $\text{IDF}(t)$ (usando a fórmula suavizada com $\ln$ dos slides [8]):**
  $$\text{IDF}(t) = \ln\left( \frac{3 + 1}{\text{DF}(t) + 1} \right) + 1$$

  * Para termos com **$\text{DF}(t) = 1$** (`a`, `não`, `tradicional`, `transforma`):
    $$\text{IDF} = \ln\left( \frac{4}{2} \right) + 1 = \ln(2) + 1 \approx 0{,}6931 + 1 = \mathbf{1{,}693}$$ [8]

  * Para termos com **$\text{DF}(t) = 2$** (`artificial`, `de`, `educação`, `gosto`, `inteligência`):
    $$\text{IDF} = \ln\left( \frac{4}{3} \right) + 1 = \ln(1{,}3333) + 1 \approx 0{,}2877 + 1 = \mathbf{1{,}288}$$ [8]

* **Cálculo do $\text{TF-IDF}$ em D1 para dois termos (`gosto` e `inteligência`):**
  * Em D1, o total de tokens é $4$.
  * **Termo 1: `gosto` em D1**
    $$\text{TF}(\text{gosto}, D1) = \frac{1}{4} = 0{,}25$$
    $$\text{IDF}(\text{gosto}) \approx 1{,}288$$
    $$\text{TF-IDF}(\text{gosto}, D1) = 0{,}25 \times 1{,}288 = \mathbf{0{,}322}$$ [9, 10]
  * **Termo 2: `inteligência` em D1**
    $$\text{TF}(\text{inteligência}, D1) = \frac{1}{4} = 0{,}25$$
    $$\text{IDF}(\text{inteligência}) \approx 1{,}288$$
    $$\text{TF-IDF}(\text{inteligência}, D1) = 0{,}25 \times 1{,}288 = \mathbf{0{,}322}$$ [9, 10]

---

#### 3. Compare os pesos e explique o que foi mais discriminativo [12].

**Resolução:**
No documento D1, todos os 4 termos presentes (`gosto`, `de`, `inteligência`, `artificial`) possuem exatamente a mesma frequência local ($\text{TF} = 0{,}25$) e aparecem em exatamente 2 documentos no corpus ($\text{DF} = 2$), obtendo todos o peso $\text{TF-IDF} = 0{,}322$. 

Contudo, ao analisar o corpus como um todo, os termos que possuem $\text{DF} = 1$ (como `transforma` no D2 ou `tradicional` no D3) possuem um $\text{IDF} = 1{,}693$ significativamente maior do que os termos com $\text{DF} = 2$ ($\text{IDF} = 1{,}288$). Portanto, os termos que aparecem em um único documento são os **mais discriminativos do corpus**, pois possuem maior capacidade de identificar o documento exclusivo em que ocorrem [8, 9, 10].

---

### Desafio de Pensamento Crítico [12]

1. **O que acontece se removermos a palavra "não" como stopword?**
   * **Resposta:** Se a palavra *"não"* for removida como stopword, o documento D3 (*"não gosto de educação tradicional"*) será transformado em `[gosto, educação, tradicional]`. A polaridade do sentimento será completamente invertida de **negativa** para **positiva** (ficando praticamente idêntica a uma frase de aprovação) [6]. Isso exemplifica que em tarefas como análise de sentimento, palavras de negação nunca devem ser removidas como stopwords [3, 6].

2. **Qual termo você espera que tenha maior IDF: "de" ou "inteligência"?**
   * **Resposta:** Em um corpus real e abrangente da língua portuguesa, o termo **"inteligência"** terá um **IDF muito maior** do que "de". A palavra *"de"* é uma preposição extremamente genérica que aparece em quase 100% dos documentos ($\text{DF}$ muito alto $\implies \text{IDF}$ baixo, próximo de 1) [6, 8]. O termo *"inteligência"* é específico e restrito a certos tópicos ($\text{DF}$ menor $\implies \text{IDF}$ elevado) [6, 8, 9].

3. **Como a representação mudaria se usássemos bigramas?**
   * **Resposta:** Em vez de palavras isoladas (unigramas), o vocabulário passaria a ser composto por pares de palavras consecutivas (ex: `[gosto de]`, `[de inteligência]`, `[inteligência artificial]`, `[não gosto]`, `[educação tradicional]`). Isso permitiria preservar a ordem local, capturar expressões compostas (*"inteligência artificial"*) e preservar a negação (*"não gosto"*), porém aumentaria significativamente o tamanho do vocabulário e a esparsidade da matriz [3, 11, 12].

4. **Em que situação BoW pode ser uma escolha adequada, mesmo sendo simples?**
   * **Resposta:** O modelo BoW é adequado como uma linha de base (*baseline*) rápida e eficiente para **classificação de documentos longos** por tema (ex: separar notícias de Esportes de notícias de Economia). Em textos longos, a mera presença e frequência de palavras-chave temáticas fortes (ex: *"gol"*, *"futebol"* vs. *"inflação"*, *"PIB"*) costuma ser suficiente para determinar o assunto com alta precisão, sem a necessidade do custo computacional de modelos sequenciais complexos [10, 11, 12].

---

## 🔑 Conceitos-chave

* **Pipeline de PLN:** Fluxo sequencial de conversão: Texto Bruto → Pré-processamento → Vetorização (BoW/TF-IDF) → Modelo → Resultado [1].
* **Tokenização:** Divisão do texto em unidades discretas (tokens/palavras) [2, 3].
* **Normalização:** Padronização de caixa (minúsculas) e acentos para uniformizar termos [2].
* **Stopwords:** Palavras muito frequentes e com pouca carga semântica (artigos, preposições) que podem ser removidas [3, 6].
* **Bag of Words (BoW):** Representação vetorial baseada unicamente na presença ou contagem de termos, descartando a ordem e a sintaxe [4, 6].
* **Matriz Documento-Termo:** Matriz esparsa onde linhas são documentos e colunas são termos do vocabulário [2, 4].
* **TF (Term Frequency):** Frequência com que um termo ocorre dentro de um documento específico [7].
* **IDF (Inverse Document Frequency):** Medida de raridade de um termo no corpus geral; penaliza termos comuns e valoriza termos raros [7, 8].
* **TF-IDF:** Produto de $\text{TF} \times \text{IDF}$ que atribui o peso final discriminativo a cada termo em cada documento [7, 9, 13].
* **Esparsidade:** Característica das matrizes documento-termo em possuir a maioria de seus elementos preenchidos por zero [11].

---

## 🧠 Relações entre os conceitos

```
┌─────────────────────────────────────────────────────────────────┐
│                           Texto Bruto                           │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Pré-Processamento                        │
│   (Tokenização ──► Normalização ──► Remoção de Stopwords)       │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Vocabulário do Corpus                      │
│            (Conjunto único e ordenado de todos os termos)       │
└─────────────────────────────────────────────────────────────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
┌─────────────────────────────────┐             ┌─────────────────────────────────┐
│      Bag of Words (BoW)         │             │             TF-IDF              │
│  • Ponderação por Frequência    │             │  • Relevância Local (TF)        │
│    Absoluta, Binária ou Relativa│             │             ×                   │
│  • Não penaliza termos comuns   │             │  • Raridade no Corpus (IDF)     │
└─────────────────────────────────┘             └─────────────────────────────────┘
         │                                               │
         └───────────────────────┬───────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Matriz Documento-Termo                      │
│                 (Entrada para Modelos de IA)                    │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Modelos de Aplicação                       │
│    (Naive Bayes, Regressão Logística, SVM, KNN, K-Means)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚖️ Comparações

### 1. Comparativo entre Tipos de Representação BoW

| Abordagem | Tipo de Valor | Vantagem Principal | Desvantagem / Limitação |
| :--- | :--- | :--- | :--- |
| **BoW Binário** | 0 ou 1 (Presença/Ausência) | Muito simples e leve computacionalmente; ótimo para textos curtos [5]. | Descarta a intensidade de repetição de palavras importantes [5]. |
| **BoW por Contagem** | Inteiros $\ge 0$ (Frequência Absoluta) | Registra a quantidade exata de vezes que o termo se repete [4, 5]. | Distorcido por documentos muito longos (não normaliza tamanho) [5]. |
| **BoW por Frequência Relativa** | Decimais $[0, 1]$ (Normalizado) | Permite comparar documentos de tamanhos totalmente diferentes [5]. | Ainda dá peso alto a palavras extremamente comuns (ex: *"de"*, *"que"*) [6]. |

---

### 2. Comparativo entre BoW vs. TF-IDF

| Aspecto | Bag of Words (BoW) | TF-IDF |
| :--- | :--- | :--- |
| **Valor no Vetor** | Contagem absoluta, presença ou fração simples [5, 10]. | Peso ponderado $\text{TF} \times \text{IDF}$ [7, 10]. |
| **Tratamento de Termos Comuns** | Atribui valores altos a termos muito frequentes [6, 10]. | **Penaliza** termos comuns reduzindo seu peso através do IDF [7, 8, 10]. |
| **Tratamento de Termos Raros** | Trata da mesma forma que os demais termos [10]. | **Valoriza** termos raros atribuindo pesos maiores [8, 9, 10]. |
| **Ordem das Palavras** | Não preserva a ordem ou estrutura sintática [4, 6, 10]. | Não preserva a ordem ou estrutura sintática [10]. |
| **Uso Típico Recomendado** | Baselines simples, classificação de documentos longos [10, 11]. | Busca textual, ranking de relevância, agrupamento e classificação refinada [10, 11]. |

---

## 📝 Resumo para revisão

* **Entrada de IA exige números:** Modelos de Machine Learning não processam texto bruto; ele precisa ser convertido em vetores numéricos estruturados via pré-processamento e vetorização [1, 2].
* **Pré-processamento prepara os dados:** Tokenização corta o texto em termos; normalização padroniza caixas; remoção de stopwords limpa ruídos; lematização/stemming reduz radicais [2, 3].
* **Bag of Words (BoW):** Representa documentos através da contagem de termos em relação a um vocabulário fixo [4]. Descarta a ordem das palavras, sintaxe e contexto [4, 6].
* **Fórmula do TF-IDF:** $\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$ [7, 9].
  * **TF (Term Frequency):** Quantas vezes o termo aparece no documento especificamente (frequência local) [7].
  * **IDF (Inverse Document Frequency):** $\ln\left(\frac{N+1}{\text{DF}+1}\right) + 1$. Mede a raridade do termo em todo o corpus [7, 8].
* **Por que usar TF-IDF sobre BoW?** O BoW dá peso alto a palavras irrelevantes que aparecem muito (como *"de"* e *"e"*). O TF-IDF reduz o peso de palavras comuns a todo o corpus e eleva o peso de palavras específicas e discriminativas [6, 7, 8, 10].
* **Cuidado com a Negação:** Remover stopwords como a palavra *"não"* pode alterar completamente o sentimento de uma frase (ex: *"não gostei"* vira *"gostei"*) [6, 12].
