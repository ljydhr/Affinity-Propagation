import pandas as pd
from sklearn.cluster import AffinityPropagation
import numpy as np
import matplotlib.pyplot as plt

# 从Excel文件读取轨迹数据
df = pd.read_excel('trajectories.xlsx', header=None)

# 将轨迹数据转换为适合聚类分析的形式
trajectories = df[0].str.split(';').tolist()
trajectories = [[tuple(map(float, point.split(','))) for point in trajectory] for trajectory in trajectories]
X = np.vstack(trajectories)

# 初始化 AffinityPropagation 类
af = AffinityPropagation(preference=-50, damping=0.5, random_state=0).fit(X)

# 获得聚类标签和聚类中心
labels = af.labels_
cluster_centers = af.cluster_centers_

n_clusters = len(cluster_centers)
print('Estimated number of clusters: %d' % n_clusters)

print('Cluster labels:')
print(labels)

print('Cluster centers:')
print(cluster_centers)

# 展示输入的轨迹数据和聚类后获得的轨迹数据
plt.figure(figsize=(10, 5))

# 展示输入的轨迹数据
plt.subplot(1, 2, 1)
for trajectory in trajectories:
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o')
plt.title('Input Trajectories')
plt.xlabel('X')
plt.ylabel('Y')

# 展示聚类后获得的轨迹数据
plt.subplot(1, 2, 2)
for i in range(n_clusters):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label='Cluster {}'.format(i))
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', label='Cluster Centers')
plt.title('Clustered Trajectories')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.tight_layout()
plt.show()