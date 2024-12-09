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
6. [Project Workflow](#project-workflow)
7. [Troubleshooting](#troubleshooting)

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

