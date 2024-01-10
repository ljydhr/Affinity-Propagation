from sklearn.cluster import AffinityPropagation
import numpy as np

np.random.seed(0)

def generate_random_trajectory():
    start_options = [[0, 0], [0, 15], [15, 0]]
    start_index = np.random.choice(len(start_options))
    start = start_options[start_index]
    end = [50, 50]
    trajectory = [start]
    current_point = start
    while current_point != end:
        if current_point[0] >= 11 and current_point[1] >= 11:
            next_point = end
        else:
            next_point = [current_point[0] + np.random.randint(0, 2), current_point[1] + np.random.randint(0, 2)]
        trajectory.append(next_point)
        current_point = next_point
    return trajectory

trajectories = [generate_random_trajectory() for _ in range(10)]

# 展示生成的随机轨迹数据集
for i, trajectory in enumerate(trajectories):
    print(f"Trajectory {i+1}: {trajectory}")

# 将轨迹数据转换为适合聚类分析的形式
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