import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Arial'] # 用来正常显示中文标签
plt.rcParams['figure.dpi'] = 300
from transformers import MPNetTokenizer, MPNetModel
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import umap
from sklearn.cluster import KMeans
import re
from nltk.corpus import stopwords
import nltk
from keybert import KeyBERT
import warnings
warnings.filterwarnings('ignore')

class Text_cluster:
    def __init__(self, input_file, output_file,  model_path, model_name ='distilbert', device='cuda', target='摘要', num_clusters = 6, top_n=10, cluster_output = False, umap_n = 3):
        self.target = target
        self.df = pd.read_excel(input_file)
        self.output_file = output_file
        self.model_path = model_path
        self.device = torch.device(device)
        self.df1 = self.df.drop_duplicates(subset=[self.target], keep='first')[[self.target]]
        self.df1[self.target] = self.df1[self.target].apply(lambda x: x[:512])
        if model_name == 'distilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-nli-mean-tokens')
            self.model = DistilBertModel.from_pretrained('distilbert-base-nli-mean-tokens')
        if model_name == 'mpnet':
            self.tokenizer = MPNetTokenizer.from_pretrained('all-mpnet-base-v2')
            self.model = MPNetModel.from_pretrained('all-mpnet-base-v2')
        if model_path:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.num_clusters = num_clusters
        self.top_n = top_n
        self.cluster_output = cluster_output
        self.umap_n = umap_n
        self.model_name = model_name


    def get_text_embedding(self, text, model):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :] # 仅保留(batch_size, hidden_size)维度
        return embedding

    def create_embeddings(self):
        text = self.df1[self.target].values.tolist()
        embeddings = self.get_text_embedding(text, model= self.model)
        return embeddings

    def umap_reduction(self, embeddings, n ):
        umap_model = umap.UMAP(n_components=n, random_state=2022)
        umap_embeddings = umap_model.fit_transform(embeddings)
        return umap_embeddings

    def plot_3d_scatter(self, df, title='UMAP Visualization', labels=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(df['x'], df['y'], df['z'], c=df['x'], cmap='YlGnBu', s=df['y']*1)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('X')
        if labels is not None:
            for cluster_label, marker in zip(range(6), ['o', 's', '^', 'x', '+', 'd']):
                cluster_points = df[df['cluster'] == cluster_label]
                ax.scatter(cluster_points['x'], cluster_points['y'], cluster_points['z'],
                           label=f'Cluster {cluster_label}', marker=marker, s=cluster_points['y']*1)
            ax.legend(fontsize='small')
        plt.show()

    def kmeans_clustering(self, umap_embeddings, n_clusters):
        cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(umap_embeddings)
        labels = cluster.labels_
        return labels

    def filter_text(self, text):
        text = re.sub(r'\([^)]*\)', '', text)
        pattern = re.compile(r'[^a-zA-Z\s]')
        text = pattern.sub(' ', text)
        words = text.split()
        filtered_words = [word for word in words if word not in stopwords.words('english')]
        return ' '.join(filtered_words)

    def process_data(self):
        self.df1[self.target] = self.df1[self.target].astype(str)
        self.df1[self.target] = [self.filter_text(item) for item in self.df1[self.target]]
        self.df1[self.target] = self.df1[self.target].apply(self.seg_depart)
        return self.df1

    def seg_depart(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        stop_words = set(stopwords.words('english'))
        outstr = ''
        for word in tokens:
            if word.lower() not in stop_words:
                outstr += word
                outstr += " "
        return outstr

    def get_top_n_words(self, text, top_n=10):
        if self.model_name == 'distilbert':
            model = KeyBERT('distilbert-base-nli-mean-tokens')
        if self.model_name == 'mpnet':
            model = KeyBERT('all-mpnet-base-v2')
        keywords = model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=top_n)
        return keywords

    def get_cluster_keywords(self, df, num_clusters, top_n=10):
        df4 = []
        for i in range(num_clusters):
            cluster = df[df['cluster']==i]
            cluster_text = ' '.join(cluster[self.target].tolist())
            common_words = self.get_top_n_words(cluster_text, top_n)
            df4.append(common_words)
        df5 = pd.DataFrame(df4).T
        df5.columns = [f'cluster{i}' for i in range(num_clusters)]
        return df5

    def run(self):
        embeddings = self.create_embeddings()
        embeddings = embeddings.detach().cpu().numpy()

        umap_embeddings = self.umap_reduction(embeddings, n = self.umap_n)
        df = pd.DataFrame(umap_embeddings, columns=['x', 'y', 'z'])
        self.plot_3d_scatter(df)
        labels = self.kmeans_clustering(umap_embeddings, n_clusters=self.num_clusters)
        df['cluster'] = labels
        self.plot_3d_scatter(df, title='KMeans Clustering', labels=labels)
        self.df1['cluster'] = labels
        df1_processed = self.process_data()
        df_keywords = self.get_cluster_keywords(df1_processed, self.num_clusters, self.top_n)
        df_keywords.to_excel(self.output_file,index=False)
        if self.cluster_output ==True:
            self.df1.to_excel('cluster_result.xlsx',index=False)
        print('结束')
        return df_keywords
