import string
import pymorphy2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# загрузка данных
with open('queries.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()

# убираю лишние слова
stop_words = ['и', 'в', 'на', 'к', 'для']
morph = pymorphy2.MorphAnalyzer() # класс для нормализации слов


def preprocess(text):
    punc = string.punctuation + '«»—'  # лишние символы
    text = ''.join(ch for ch in text if ch not in punc)  # удаление символов
    words = text.split()
    words = [morph.parse(word)[0].normal_form for word in words if
             word not in stop_words]  # удаление лишних слов и нормализация слов
    return ' '.join(words)


clean_queries = []
for query in data:
    clean_query = preprocess(query)
    clean_queries.append(clean_query)

# векторизация слов, по количеству вхождений
num_clusters = 5
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(clean_queries)

# обучение кластеризации
km = KMeans(n_clusters=num_clusters)
km.fit(X)

# дендрограмма
Z = linkage(X.toarray(), 'ward')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)

# вывод наиболее приближенных слов в кластерах
print("Топ запросов по теме:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for i in range(num_clusters):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
    print()

# количество объектов у каждого кластера
labels = km.labels_
result = {}
for i in range(num_clusters):
    result[i] = labels.tolist().count(i)
print(result)

plt.show()