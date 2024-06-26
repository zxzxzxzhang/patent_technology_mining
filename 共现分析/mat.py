import pandas as pd
import networkx as nx
import mat.pyplot as plt
import itertools

# 读取数据
df = pd.read_excel(r'C:\Users\RIC_ZX\Desktop\papers\ZLW\美国和欧盟 已筛选TAC_(poly butylene adipate-co-terephthalate) OR TAC_(PBAT).XLSX')

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['figure.dpi'] = 300

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

# 添加节点
for i, row in df.iterrows():
    names = row["[标]当前申请(专利权)人"].split(" | ")
    for name in names:
        G.add_node(name)

# 添加边
for pair in pairs:
    G.add_edge(pair[0], pair[1])

# 绘制图形
pos = nx.spring_layout(G, k=0.5)
nx.draw_networkx_nodes(G, pos, node_size=10, node_color='blue')
nx.draw_networkx_edges(G, pos, edge_color='gray')

# 标签显示
labels = {node: node for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=8)

# 设置布局
plt.axis('off')

# 显示图形
plt.show()
