import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import itertools

# 读取数据，注意修改文件路径和列名
df = pd.read_excel(r'C:\Users\RIC_ZX\Desktop\papers\ZLW\美国和欧盟 已筛选TAC_(poly butylene adipate-co-terephthalate) OR TAC_(PBAT).XLSX')
inventor_col = '[标]当前申请(专利权)人'

# 提取发明人信息
inventors = []
for i, row in df.iterrows():
    names = row[inventor_col].split(" | ")
    inventors.append(names)

pairs = []
for i in inventors:
    for j in itertools.combinations(i, 2):
        pairs.append(j)

# 创建图形对象
G = nx.Graph()

# 添加节点
for i, row in df.iterrows():
    names = row[inventor_col].split(" | ")
    for name in names:
        G.add_node(name)

# 添加边
for pair in pairs:
    G.add_edge(pair[0], pair[1])

# 绘制图形
pos = nx.spring_layout(G, k=0.5)

# 创建节点坐标列表
node_x = []
node_y = []
for node, coordinates in pos.items():
    x, y = coordinates
    node_x.append(x)
    node_y.append(y)

# 创建节点的标签列表
node_labels = list(G.nodes())

# 创建边的端点坐标列表
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

# 创建绘图对象
fig = go.Figure()

# 添加节点和边
fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='gray'), hoverinfo='none'))
fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=10, color='blue'), text=node_labels,
                         hovertemplate='%{text}', hoverlabel=dict(font_size=8)))

# 设置布局
fig.update_layout(showlegend=False, hovermode='closest')

# 显示图形
fig.show()

# 将图形对象转换为NetworkX图形对象
nx_G = nx.Graph(G)

# 计算节点度数
degrees = nx.degree(nx_G)

# 计算节点聚类系数
clustering_coefficients = nx.clustering(nx_G)

# 输出结果
print("节点度数：")
print(degrees)
print("\n节点聚类系数：")
print(clustering_coefficients)


