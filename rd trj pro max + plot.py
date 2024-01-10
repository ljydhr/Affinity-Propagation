import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation

np.random.seed(0)

def generate_random_trajectory():
    start_options = [[0, 0], [0, 1], [1, 0]]
    start_index = np.random.choice(len(start_options))
    start = start_options[start_index]
    end = [12, 12]
    trajectory = [start]
    current_point = start
    while current_point != end:
        if current_point[0] >= 11 and current_point[1] >= 11:
            next_point = end
        else:
            deviation = np.random.uniform(-0.1, 0.1, size=2)
            next_point = [current_point[0] + np.random.randint(0, 2) + deviation[0],
                          current_point[1] + np.random.randint(0, 2) + deviation[1]]
        trajectory.append(next_point)
        current_point = next_point
    return trajectory

trajectories = [generate_random_trajectory() for _ in range(20)]

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

# 绘制输入的轨迹数据
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
for trajectory in trajectories:
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o')
plt.title('Input Trajectories')
plt.xlabel('X')
plt.ylabel('Y')

# 绘制聚类中心按序号连接形成的轨迹
plt.subplot(1, 2, 2)
sorted_centers = sorted(cluster_centers, key=lambda x: (x[0], x[1]))
sorted_indices = np.argsort(cluster_centers[:, 0])
for i, center in enumerate(sorted_centers):
    plt.plot(center[0], center[1], marker='o', label=str(i + 1))
    if i > 0:
        prev_center = sorted_centers[i - 1]
        plt.plot([prev_center[0], center[0]], [prev_center[1], center[1]], linestyle='-', color='blue')
plt.title('Cluster Centers')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.tight_layout()
plt.show()