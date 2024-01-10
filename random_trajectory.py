import numpy as np

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
            next_point = [current_point[0] + np.random.randint(0, 2), current_point[1] + np.random.randint(0, 2)]
        trajectory.append(next_point)
        current_point = next_point
    return trajectory

trajectories = [generate_random_trajectory() for _ in range(5)]

# 展示生成的随机轨迹数据集
for i, trajectory in enumerate(trajectories):
    print(f"Trajectory {i+1}: {trajectory}")