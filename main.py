import string
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


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

# векторизация слов, по важности слов
num_clusters = 20
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(clean_queries)

# обучение кластеризации
km = KMeans(n_clusters=num_clusters)
km.fit(X)

# дендрограмма
Z = linkage(X.toarray(), 'ward')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)

# вывод наиболее приближенных слов в кластерах

order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()


# вывод TF-IDF коэффициентов для каждого слова
for i in range(num_clusters):
    print("Кластер %d:" % i)
    print("Топ слов и коэффициент tf-idf:")
    for ind in order_centroids[i, :10]:
        print(' %s (%f)' % (terms[ind], km.cluster_centers_[i, ind]))
    print()

parameters = {'n_clusters': [5, 10, 15, 20], 'init': ['k-means++', 'random'],
              'n_init': [10, 20, 30], 'algorithm': ['full', 'elkan']}
grid_search = GridSearchCV(KMeans(), parameters, cv=5, n_jobs=-1)
grid_search.fit(X)
print(grid_search.best_params_)
# количество объектов у каждого кластера
labels = km.labels_
result = {}
for i in range(num_clusters):
    result[i] = labels.tolist().count(i)
print(result)

plt.show()

# метрики качества
silhouette_avg = silhouette_score(X, km.labels_)
print("Silhouette score:", silhouette_avg)


ch_score = calinski_harabasz_score(X.toarray(), km.labels_)
print("Calinski-Harabasz Index:", ch_score)


db_score = davies_bouldin_score(X.toarray(), km.labels_)
print("Davies-Bouldin Index:", db_score)
