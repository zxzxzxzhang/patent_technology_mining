import pandas as pd
import networkx as nx

# 读取xlsx文件
data = pd.read_excel(r'C:\Users\RIC_ZX\Desktop\papers\ZLW\(申请号去重后）美国和欧盟 已筛选TAC_(poly butylene adipate-co-terephthalate) OR TAC_(PBAT).xlsx')

# 构建边列表
edges = []
for row in data.iterrows():
    _, patent = row
    cited_patents = patent['引用专利'].split(' | ')
    for cited_patent in cited_patents:
        if cited_patent != '-':
            edges.append((patent['公开(公告)号'], cited_patent))

from collections import Counter

# 统计第二列中每个字符串的数量
string_counts = Counter(edge[1] for edge in edges)

# 仅保留字符串数量大于等于2的行
filtered_edges = [edge for edge in edges if string_counts[edge[1]] >= 8]

# 创建有向图
G = nx.DiGraph()
G.add_edges_from(filtered_edges)

# 生成带权邻接矩阵
adj_matrix = nx.to_pandas_adjacency(G, weight='weight')

# 保存邻接矩阵到CSV文件
adj_matrix.to_csv("专利引用网络带权邻接矩阵大于等于8.csv")