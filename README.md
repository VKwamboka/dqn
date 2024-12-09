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
