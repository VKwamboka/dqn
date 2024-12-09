import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

log_file = '/home/kwamboka/dqn_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/simulation_results.log'

episodes = []
scores = []
epsilons = []

with open(log_file, 'r') as file:
    for line in file:
        if 'Episode:' in line:
            try:
                parts = line.split('INFO:root:')[1].split(', ')  

                episode = int(parts[0].split(':')[1].strip())  # Episode
                score = float(parts[1].split(':')[1].strip())  # Score
                epsilon = float(parts[2].split(':')[1].strip())  # Epsilon

                episodes.append(episode)
                scores.append(score)
                epsilons.append(epsilon)
            except Exception as e:
                # Handle any line that doesn't match the expected format
                print(f"Skipping invalid line: {line}")
                continue

data = pd.DataFrame({
    'Episode': episodes,
    'Score': scores,
    'Epsilon': epsilons
})

mean_score = np.mean(scores)
std_score = np.std(scores)

print(f"Mean Score: {mean_score}")
print(f"Standard Deviation of Score: {std_score}")

# Visualize the scores using a box plot
plt.figure(figsize=(10, 6))
plt.boxplot(scores, vert=False, patch_artist=True, notch=True)

plt.axvline(mean_score, color='r', linestyle='--', label=f'Mean: {mean_score:.2f}')
plt.axvline(mean_score + std_score, color='g', linestyle=':', label=f'Std Dev: {std_score:.2f}')
plt.axvline(mean_score - std_score, color='g', linestyle=':')

plt.title('Box Plot of Scores with Mean and Standard Deviation')
plt.xlabel('Score')
plt.legend()


plt.show()

# Visualize the epsilon decay (optional plot)
# plt.figure(figsize=(10, 6))
# plt.plot(episodes, epsilons, label='Epsilon Decay', color='b')
# plt.title('Epsilon Decay Over Episodes')
# plt.xlabel('Episode')
# plt.ylabel('Epsilon')
# plt.legend()
# plt.show()
