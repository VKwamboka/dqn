# Multi-Agent Reinforcement Learning for TurtleBot3 Using ROS2 Humble and Gazebo

## Overview
This project demonstrates multi-agent reinforcement learning (MARL) on TurtleBot3 robots in a ROS2 Humble and Gazebo environment. The workflow includes training a single TurtleBot agent using reinforcement learning (DQN) and then extending the model to multiple agents.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
    - [Ubuntu Installation](#ubuntu-installation)
    - [ROS2 Installation](#ros2-installation)
    - [Dependent ROS2 Packages](#dependent-ros2-packages)
    - [TurtleBot3 Packages](#turtlebot3-packages)
3. [Environment Configuration](#environment-configuration)
4. [Setting Up Machine Learning](#setting-up-machine-learning)
5. [Running the Project](#running-the-project)
6. [Spawning Multiple agents](#spawning-multiple-agents)
7. [Project Workflow](#project-workflow)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites
- A PC with Ubuntu 22.04 LTS Desktop installed.
- Basic knowledge of ROS2, Gazebo, and Python.
- Familiarity with reinforcement learning concepts.

---

## Installation

### Ubuntu Installation
1. Download Ubuntu 22.04 LTS Desktop image from the official [Ubuntu website](https://ubuntu.com/download/desktop).
2. Follow the [Ubuntu installation guide](https://ubuntu.com/tutorials/install-ubuntu-desktop) to set up Ubuntu on your PC.

### ROS2 Installation
1. Install ROS2 Humble by following the [official ROS2 documentation](https://docs.ros.org/en/humble/Installation.html). The Debian package installation method is recommended for Linux users.

### Dependent ROS2 Packages
Install the required ROS2 packages for Gazebo, Cartographer, and Navigation2:

```bash
# Install Gazebo
sudo apt install ros-humble-gazebo-*

# Install Cartographer
sudo apt install ros-humble-cartographer
sudo apt install ros-humble-cartographer-ros

# Install Navigation2
sudo apt install ros-humble-navigation2
sudo apt install ros-humble-nav2-bringup
```

## TurtleBot3 Packages

### Create a Workspace and Clone the TurtleBot3 Repositories
To set up the TurtleBot3 packages, follow the steps below:

```bash
mkdir -p ~/turtlebot3_ws/src
cd ~/turtlebot3_ws/src/
git clone -b humble https://github.com/ROBOTIS-GIT/DynamixelSDK.git
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3.git
sudo apt install python3-colcon-common-extensions
cd ~/turtlebot3_ws
colcon build --symlink-install
echo 'source ~/turtlebot3_ws/install/setup.bash' >> ~/.bashrc
source ~/.bashrc
```
## Environment Configuration
Set up the ROS environment:
```bash
echo 'export ROS_DOMAIN_ID=30 #TURTLEBOT3' >> ~/.bashrc
echo 'source /usr/share/gazebo/setup.sh' >> ~/.bashrc
source ~/.bashrc
```
## Setting Up Machine Learning
### 1. Clone the turtlebot3_machine_learning repository:
```bash
git clone https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning.git
```
### 2. Install Python Libraries
Install necessary Python libraries such as TensorFlow, Keras, and other dependencies. You can use either Anaconda or pip for installation.

### 3. Configure Reinforcement Learning Parameters
Set the reinforcement learning parameters in the training script. By default:

- The agent receives a positive reward for moving closer to the goal.
- The agent receives a negative reward for moving away from the goal or colliding with obstacles.
- A large positive reward is given when the agent reaches the goal.
- A large negative reward is given upon collision with an obstacle.


## Running the Project
### Training a Single Agent

The training process is divided into four stages, each introducing increasing levels of complexity to the environment:
1. **Stage 1**: Plain environment without obstacles.
2. **Stage 2**: Environment with static obstacles.
3. **Stage 3**: Environment with moving obstacles.
4. **Stage 4**: Environment with both static and moving obstacles.

## Training in Stage 3 (Example)
To train the agent in Stage 3, follow the steps below:

### 1. Launch the Gazebo Environment
Start the Gazebo simulation for Stage 3:
```bash
ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage3.launch.py
```
### 2. Spawning Goals in Gazebo
To spawn goals in the Gazebo environment for the TurtleBot3 training, use the following command:

```bash
ros2 run turtlebot3_dqn dqn_gazebo 3
```
The number 3 specifies the stage in which the goals are being spawned. Replace 3 with:
1 for Stage 1 (plain environment).
2 for Stage 2 (static obstacles).
3 for Stage 3 (moving obstacles).
4 for Stage 4 (combined moving and static obstacles).

### 3. Launch the DQN Environment
Start the DQN environment:
```bash
ros2 run turtlebot3_dqn dqn_environment
```
### 4. Start the DQN Agent
Begin training the agent in Stage 3:
```bash
ros2 run turtlebot3_dqn dqn_agent 3
```
### Training in Other Stages
To train in other stages, replace 3 with the desired stage number (1, 2, or 4) in the commands above. For example, to train in Stage 1:
```bash
ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage1.launch.py
ros2 run turtlebot3_dqn dqn_gazebo 1
ros2 run turtlebot3_dqn dqn_environment
ros2 run turtlebot3_dqn dqn_agent 1
```

## Spawning Multiple agents
To spawn multiple TurtleBots in the Gazebo environment, the simulation file in the `turtlebot3_simulations` package was modified. The updated code allows multiple robots to be spawned in a grid layout. Below is the modified code:
```python
def generate_launch_description():
    ld = LaunchDescription()

    # Configuration for the TurtleBot3 model
    TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'burger')
    model_folder = 'turtlebot3_' + TURTLEBOT3_MODEL
    urdf_path = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'models',
        model_folder,
        'model.sdf'
    )

    # Variables for multiple robots
    num_robots = 2  # Adjust the number of robots as needed
    num_columns = 2  # Define how many robots in each row (columns)
    x_start, y_start = 0, 0  # Starting positions for spawning
    x_spacing, y_spacing = 2.0, 2.0  # Spacing between robots

    last_action = None

    for i in range(num_robots):
        # Compute row and column for grid layout
        row = i // num_columns
        col = i % num_columns

        # Unique name and namespace for each robot
        name = f'turtlebot{i}'
        namespace = f'/robot{i}'

        # Compute the x and y positions for grid layout
        x_position = x_start + col * x_spacing
        y_position = y_start + row * y_spacing

        # Spawn each TurtleBot3 robot at a different position
        spawn_robot = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-file', urdf_path,
                '-entity', name,
                '-robot_namespace', namespace,
                '-x', str(x_position),
                '-y', str(y_position),
                '-z', '0.01'
            ],
            output='screen'
        )

        # Add the state publisher for each robot
        state_publisher = Node(
            package='robot_state_publisher',
            namespace=namespace,
            executable='robot_state_publisher',
            output='screen',
            arguments=[urdf_path]
        )

        # Handle sequential spawning to avoid conflicts
        if last_action is None:
            ld.add_action(spawn_robot)
            ld.add_action(state_publisher)
        else:
            spawn_event = RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=last_action,
                    on_exit=[spawn_robot, state_publisher]
                )
            )
            ld.add_action(spawn_event)

        # Set last action for sequential spawning
        last_action = spawn_robot

    return ld
```
## Loading the Trained Model to New Agents

In multi-agent reinforcement learning, once an agent is trained in the environment, the learned model can be reused for additional agents being added to the environment. This process helps in transferring knowledge, speeding up the training process, and ensuring that new agents start with a baseline understanding of the task.

### Concept Overview

1. **Pre-Trained Model**:
   - The trained model from a single agent is a neural network that has learned to map the agent's state to optimal actions for achieving the goal.
   - This model includes parameters (weights and biases) that represent the agent's policy for navigating the environment.

2. **New Agents**:
   - When new agents are introduced to the environment, they can start with the pre-trained model instead of learning from scratch.
   - This ensures that the new agents have a functional policy that allows them to perform basic navigation and task completion.

3. **Benefits**:
   - **Faster Training**: New agents already have a foundation to build upon, reducing the time needed for them to learn complex behaviors.
   - **Consistency**: Ensures all agents operate under a similar policy, reducing variability in behavior.
   - **Scalability**: Facilitates the addition of more agents without significant computational overhead for training.

### Steps for Loading the Trained Model

1. **Save the Trained Model**:
   - After training the first agent, save the model's weights to a file (e.g., `trained_model.h5`).

2. **Load the Model for New Agents**:
   - In the environment setup for new agents, load the saved model weights into the neural network.
   - This can be achieved using frameworks like TensorFlow or PyTorch. For example:
     ```python
     from tensorflow.keras.models import load_model

     # Load the trained model
     model = load_model('trained_model.h5')
     ```

3. **Assign the Model to New Agents**:
   - Initialize the new agents with the loaded model, enabling them to start using the pre-trained policy.

4. **Fine-Tuning (Optional)**:
   - Allow new agents to continue learning in the environment to adapt to any changes or additional complexities introduced with more agents.

### Application in the TurtleBot3 Multi-Agent Environment

- **Single-Agent Training**:
  - Train a single TurtleBot3 in the environment using stages (e.g., plain, static obstacles, moving obstacles, combined obstacles).
  - Save the trained model after successful training.

- **Adding New Agents**:
  - When introducing additional TurtleBots, initialize their neural networks with the saved model.
  - Launch the multi-agent environment and assign the loaded model to the new agents.

