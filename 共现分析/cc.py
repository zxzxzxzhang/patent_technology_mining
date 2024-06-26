import pandas as pd
import networkx as nx
import seaborn as sb
import itertools

import mat.pyplot as plt
df = pd.read_excel(r'C:\Users\RIC_ZX\Desktop\papers\ZLW\美国和欧盟 已筛选TAC_(poly butylene adipate-co-terephthalate) OR TAC_(PBAT).XLSX')

# 提取发明人信息
inventors = []
for i, row in df.iterrows():
    names = row["发明人"].split(" | ")
    inventors.append(names)

pairs = []
for i in inventors:
    for j in itertools.combinations(i, 2):
        pairs.append(j)

# 创建图形对象
G = nx.Graph()

# 添加节点
for i, row in df.iterrows():
    names = row["发明人"].split(" | ")
    for name in names:
        G.add_node(name)

# 添加边
for pair in pairs:
    G.add_edge(pair[0], pair[1])

# 设置节点大小和颜色
node_size = [len(G[n]) * 10 for n in G]
node_color = [G.degree(n) for n in G]

# 绘制图形
pos = nx.spring_layout(G, k=0.5)
nx.draw_networkx(G, pos, with_labels=True, node_size=node_size, node_color=node_color, font_size=8, cmap=plt.cm.Blues)

# 保存图形
plt.savefig("inventor_network.png")
plt.show()