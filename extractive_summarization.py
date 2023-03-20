import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')

# stopwords = nltk.corpus.stopwords.words('portuguese')
new_stopwords = ['de', 'em', 'que', 'com', '.', ',', '(', ')', ':', '-', 'ter', 'tem']
# stopwords.append('new_stopwords')

def text_summarizer(texto: str, num_sentencas: int, freq: int) -> str:
    """
    Gera um resumo do texto com base nas sentenças que contém as palavras mais frequentes no texto completo.

    Args:
        texto: Uma string contendo o texto a ser resumido.
        num_sentencas: O número de sentenças que o resumo deve conter.
        freq: A frequência mínima das palavras para serem consideradas importantes.

    Returns:
        Uma string contendo o resumo gerado.

    """
    # tokenização do texto em sentenças
    sentences = sent_tokenize(texto)

    # lista de stopwords em português
    stop_words = set(stopwords.words('portuguese'))

    # lista de todas as palavras no texto
    words = []
    for sentence in sentences:
        # tokenização da sentença em palavras
        sentence_words = word_tokenize(sentence)

        # remoção de stopwords
        sentence_words = [word for word in sentence_words if not word.lower() in stop_words]
        sentence_words = [word for word in sentence_words if not word.lower() in new_stopwords]

        # adiciona as palavras da sentença na lista de palavras do texto
        words.extend(sentence_words)

    # contagem das palavras e seleção das mais frequentes
    freq_table = nltk.FreqDist(words)
    most_freq = freq_table.most_common(freq)

    # lista de sentenças selecionadas para o resumo
    summary_sentences = []

    # processamento de cada sentença
    for sentence in sentences:
        
        # verificação de quais sentenças contém as palavras mais frequentes
        if any(word[0] in sentence for word in most_freq) and sentence not in summary_sentences:
            
            # Seleciona a sentença com o maior número de palavras dentre as mais frequentes
            selected_sentence = max([sentence for sentence in sentences if any(word[0] in sentence for word in most_freq)], key=lambda x: len(word_tokenize(x)))
            summary_sentences.append(selected_sentence)

        # parada quando atingir o número máximo de sentenças selecionadas
        if len(summary_sentences) == num_sentencas:
            break

    # junção das sentenças selecionadas em um texto resumido
    resumo = ' '.join(summary_sentences)
    return resumo, most_freq


# Este código utiliza a biblioteca NLTK para tokenizar o texto em sentenças e palavras, remover as stopwords e contar as palavras mais frequentes em cada sentença. Em seguida, é feita a seleção das sentenças mais importantes com base nas palavras mais frequentes encontradas em cada uma delas. As sentenças selecionadas são agrupadas em um único texto resumido. O número de sentenças selecionadas pode ser controlado pelo parâmetro num_sentencas.

# Agora, ao invés de verificar se a sentença contém alguma das palavras mais frequentes e adicioná-la ao resumo, o código utiliza a função max do Python para selecionar a sentença com o maior número de palavras dentre as sentenças que contêm as palavras mais frequentes. Isso garante que o resumo contenha as sentenças mais informativas do texto.
