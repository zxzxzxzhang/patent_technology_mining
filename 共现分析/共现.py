import pandas as pd
import networkx as nx
import mat.pyplot as plt
import itertools

# 读取数据
df = pd.read_excel(r'C:\Users\RIC_ZX\Desktop\papers\ZLW\美国和欧盟 已筛选TAC_(poly butylene adipate-co-terephthalate) OR TAC_(PBAT).XLSX')

# 提取发明人信息
inventors = []
for i, row in df.iterrows():
    names = row["[标]当前申请(专利权)人"].split(" | ")
    inventors.append(names)

pairs = []
for i in inventors:
    for j in itertools.combinations(i, 2):
        pairs.append(j)

# 创建图形对象
G = nx.Graph()

# 添加节点和边
for i, row in df.iterrows():
    names = row["[标]当前申请(专利权)人"].split(" | ")
    for name in names:
        G.add_node(name)
    for pair in pairs:
        G.add_edge(pair[0], pair[1])

# 绘制图形
pos = nx.shell_layout(G)

# 绘制节点
plt.figure(figsize=(10, 10))
nx.draw(G, pos, node_color='b', node_size=3000, with_labels=True, font_size=8)

# 显示图形
plt.show()
