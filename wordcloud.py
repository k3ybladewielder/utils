from wordcloud import WordCloud
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

# plotando wordclouds
def plot_wordcloud(text, title):
    # criando o objeto WordCloud
    nuvem_palavras = WordCloud(width = 1200, height = 800,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 8).generate(palavras)

    # plotando a wordcloud
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(nuvem_palavras)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.title(f'{title}', fontsize=20)

    plt.show()
    
# plotando multiplos wordlouds lado a lado    
def plot_paired_wordcloud(textos, titulos):
	wordclouds = []
	for i, texto in enumerate(textos):
	    wordcloud = WordCloud(width=1200, height=800, max_words=50,
	    			  background_color="white", stopwords = stopwords,
	    			  min_font_size = 8).generate(texto)
	    wordclouds.append(wordcloud)

	# Plotando as WordClouds lado a lado com seus títulos
	fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

	for i, ax in enumerate(axes.flatten()):
	    ax.imshow(wordclouds[i], interpolation='bilinear')
	    ax.set_title(titulos[i])
	    ax.axis("off")

	plt.tight_layout()
	plt.show()    
