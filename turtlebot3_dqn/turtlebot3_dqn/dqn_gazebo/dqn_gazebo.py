#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Ryan Shim, Gilbert

import os
import random
import sys

from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty


class DQNGazebo(Node):
    def __init__(self, stage):
        super().__init__('dqn_gazebo')

        """************************************************************
        ** Initialise variables
        ************************************************************"""
        # Stage
        self.stage = int(stage)        

        def get_package_share_directory(package_name):
            from ament_index_python.packages import get_package_share_directory as get_pkg_share_dir
            return get_pkg_share_dir(package_name)


        # Entity 'goal'
        # self.entity_dir_path = os.path.dirname(os.path.realpath(__file__))
        # self.entity_dir_path = self.entity_dir_path.replace(
        #     'turtlebot3_machine_learning/turtlebot3_dqn/turtlebot3_dqn/dqn_gazebo',
        #     'turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_dqn_world/goal_box')
        # self.entity_path = os.path.join(self.entity_dir_path, 'model.sdf')
        # self.entity = open(self.entity_path, 'r').read()
        self.entity_dir_path = os.path.join(
            get_package_share_directory('turtlebot3_gazebo'),  # Update package name if needed
            'models/turtlebot3_dqn_world/goal_box'
        )
        self.entity_path = os.path.join(self.entity_dir_path, 'model.sdf')
        self.entity = open(self.entity_path, 'r').read()
        self.entity_name = 'goal'

        self.goal_pose_x = 0.5
        self.goal_pose_y = 0.0

        self.init_state = False

        """************************************************************
        ** Initialise ROS publishers, subscribers and clients
        ************************************************************"""
        qos = QoSProfile(depth=10)

        # Initialise publishers
        self.goal_pose_pub = self.create_publisher(Pose, 'goal_pose', qos)

        # Initialise client
        self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
        self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
        self.reset_simulation_client = self.create_client(Empty, 'reset_simulation')

        # Initialise servers
        self.task_succeed_server = self.create_service(
            Empty,
            'task_succeed',
            self.task_succeed_callback)
        self.task_fail_server = self.create_service(Empty, 'task_fail', self.task_fail_callback)

        # Process
        self.publish_timer = self.create_timer(
            0.010,  # unit: s
            self.publish_callback)

    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""
    def publish_callback(self):
        # Init
        if self.init_state is False:
            self.delete_entity()
            self.reset_simulation()
            self.init_state = True
            print("init!!!")
            print("Goal pose: ", self.goal_pose_x, self.goal_pose_y)

        # Publish goal pose
        goal_pose = Pose()
        goal_pose.position.x = self.goal_pose_x
        goal_pose.position.y = self.goal_pose_y
        self.goal_pose_pub.publish(goal_pose)
        self.spawn_entity()

    def task_succeed_callback(self, request, response):
        self.delete_entity()
        # new
        # self.reset_simulation()
        self.generate_goal_pose()
        print("generate a new goal :)", self.goal_pose_x, self.goal_pose_y)

        return response

    def task_fail_callback(self, request, response):
        self.delete_entity()
        self.reset_simulation()
        self.generate_goal_pose()
        print("reset the gazebo environment :(")

        return response

    def generate_goal_pose(self):
        if self.stage != 4:
            self.goal_pose_x = random.randrange(-15, 16) / 10.0
            self.goal_pose_y = random.randrange(-15, 16) / 10.0
        else:
            goal_pose_list = [[1.0, 0.0], [2.0, -1.5], [0.0, -2.0], [2.0, 2.0], [0.8, 2.0],
                              [-1.9, 1.9], [-1.9, 0.2], [-1.9, -0.5], [-2.0, -2.0], [-0.5, -1.0]]
            index = random.randrange(0, 10)
            self.goal_pose_x = goal_pose_list[index][0]
            self.goal_pose_y = goal_pose_list[index][1]
            print("Goal pose: ", self.goal_pose_x, self.goal_pose_y)

    def reset_simulation(self):
        req = Empty.Request()
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        self.reset_simulation_client.call_async(req)

    def delete_entity(self):
        req = DeleteEntity.Request()
        req.name = self.entity_name
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        self.delete_entity_client.call_async(req)

    def spawn_entity(self):
        goal_pose = Pose()
        goal_pose.position.x = self.goal_pose_x
        goal_pose.position.y = self.goal_pose_y
        req = SpawnEntity.Request()
        req.name = self.entity_name
        req.xml = self.entity
        req.initial_pose = goal_pose
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        self.spawn_entity_client.call_async(req)


def main(args=sys.argv[1:]):
    rclpy.init(args=args)
    stage = args[0] if args else "1"
    dqn_gazebo = DQNGazebo(stage)
    rclpy.spin(dqn_gazebo)

    dqn_gazebo.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


# import math
# import os
# import random
# import sys

# from gazebo_msgs.srv import DeleteEntity, SpawnEntity
# from geometry_msgs.msg import Pose
# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile
# from std_srvs.srv import Empty


# class ActorCriticGazebo(Node):
#     def __init__(self, stage, num_agents=1):
#         super().__init__('actor_critic_gazebo')

#         """************************************************************
#         ** Initialise variables
#         ************************************************************"""
#         self.stage = int(stage[0])
#         self.num_agents = num_agents

        # # Entity 'goal' for each agent
        # self.entity_dir_path = os.path.join(
        #     get_package_share_directory('turtlebot3_gazebo'),  # Update package name if needed
        #     'models/turtlebot3_dqn_world/goal_box'
        # )
        # self.entity_path = os.path.join(self.entity_dir_path, 'model.sdf')
        # self.entity = open(self.entity_path, 'r').read()

#         self.goal_pose_x = [0.5 for _ in range(self.num_agents)]
#         self.goal_pose_y = [0.0 for _ in range(self.num_agents)]

#         self.init_state = False

#         """************************************************************
#         ** Initialise ROS publishers, subscribers and clients
#         ************************************************************"""
#         qos = QoSProfile(depth=10)

#         # Initialise publishers
#         # self.goal_pose_pub = self.create_publisher(Pose, 'goal_pose', qos)
#         self.goal_pose_pubs = [
#             self.create_publisher(Pose, f'goal_pose{i}', qos) for i in range (self.num_agents)
#         ]

#         # Initialise clients
#         self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
#         self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
#         self.reset_simulation_client = self.create_client(Empty, 'reset_simulation')

#         # Initialise servers
#         self.task_succeed_server = self.create_service(
#             Empty,
#             'task_succeed',
#             self.task_succeed_callback
#         )
#         self.task_fail_server = self.create_service(
#             Empty,
#             'task_fail',
#             self.task_fail_callback
#         )

#         # Process
#         self.publish_timer = self.create_timer(
#             0.010,  # unit: s
#             self.publish_callback
#         )

#     """*******************************************************************************
#     ** Callback functions and relevant functions
#     *******************************************************************************"""

#     def publish_callback(self):
#         # Init
#         if not self.init_state:
#             self.delete_all_entities()
#             self.reset_simulation()
#             self.init_state = True
#             print("Initialization complete!")
#             self.generate_goal_poses()

#         # Publish goal poses for all agents
#         for i in range(self.num_agents):
#             goal_pose = Pose()
#             goal_pose.position.x = self.goal_pose_x[i]
#             goal_pose.position.y = self.goal_pose_y[i]
#             # self.get_logger().info(f"Publishing Goal for Agent {i}: (x={goal_pose.position.x}, y={goal_pose.position.y})")
#             self.spawn_entity(i, goal_pose)
#             self.goal_pose_pubs[i].publish(goal_pose) 


#     def task_succeed_callback(self, request, response):
#         for i in range(self.num_agents):
#             self.delete_entity(i)
#             self.generate_goal_pose(i)
#             print(f"Agent {i} generated a new goal :)")
#         return response

#     def task_fail_callback(self, request, response):
#         for i in range(self.num_agents):
#             self.delete_entity(i)
#             self.generate_goal_pose(i)
#             print(f"Agent {i} reset the environment :(")
#         self.reset_simulation()
#         return response

#     def generate_goal_poses(self):
#         for i in range(self.num_agents):
#             self.generate_goal_pose(i)

#     def generate_goal_pose(self, agent_id):
#         if self.stage != 4:
#             self.goal_pose_x[agent_id] = random.uniform(-1.5, 1.5)
#             self.goal_pose_y[agent_id] = random.uniform(-1.5, 1.5)
#         else:
#             goal_pose_list = [
#                 [1.0, 0.0], [2.0, -1.5], [0.0, -2.0], [2.0, 2.0],
#                 [0.8, 2.0], [-1.9, 1.9], [-1.9, 0.2], [-1.9, -0.5],
#                 [-2.0, -2.0], [-0.5, -1.0]
#             ]
#             index = random.randrange(len(goal_pose_list))
#             self.goal_pose_x[agent_id] = goal_pose_list[index][0]
#             self.goal_pose_y[agent_id] = goal_pose_list[index][1]
#             print(f"Agent {agent_id} Goal pose: {self.goal_pose_x[agent_id]}, {self.goal_pose_y[agent_id]}")

#     def reset_simulation(self):
#         req = Empty.Request()
#         while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
#             self.get_logger().info('Service reset_simulation not available, waiting...')
#         self.reset_simulation_client.call_async(req)

#     def delete_entity(self, agent_id):
#         req = DeleteEntity.Request()
#         req.name = f'goal_{agent_id}'  # Unique name for each agent's goal
#         while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
#             self.get_logger().info(f'Service delete_entity not available for Agent {agent_id}, waiting...')
#         self.delete_entity_client.call_async(req)

#     def delete_all_entities(self):
#         for i in range(self.num_agents):
#             self.delete_entity(i)

#     def spawn_entity(self, agent_id, pose):
#         req = SpawnEntity.Request()
#         req.name = f'goal_{agent_id}'  # Unique name for each agent's goal
#         req.xml = self.entity
#         req.initial_pose = pose
#         while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
#             self.get_logger().info(f'Service spawn_entity not available for Agent {agent_id}, waiting...')
#         self.spawn_entity_client.call_async(req)

#     def euler_from_quaternion(self, quat):
#         """
#         Converts quaternion (w in last place) to euler roll, pitch, yaw
#         quat = [x, y, z, w]
#         """
#         x = quat.x
#         y = quat.y
#         z = quat.z
#         w = quat.w

#         sinr_cosp = 2 * (w * x + y * z)
#         cosr_cosp = 1 - 2 * (x * x + y * y)
#         roll = math.atan2(sinr_cosp, cosr_cosp)

#         sinp = 2 * (w * y - z * x)
#         pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)

#         siny_cosp = 2 * (w * z + x * y)
#         cosy_cosp = 1 - 2 * (y * y + z * z)
#         yaw = math.atan2(siny_cosp, cosy_cosp)

#         return roll, pitch, yaw


# def get_package_share_directory(package_name):
#     from ament_index_python.packages import get_package_share_directory as get_pkg_share_dir
#     return get_pkg_share_dir(package_name)


# def main(args=None):
#     rclpy.init(args=args)
#     actor_critic_gazebo = ActorCriticGazebo(stage=sys.argv[1:], num_agents=1)  # Set num_agents as needed
#     rclpy.spin(actor_critic_gazebo)

#     actor_critic_gazebo.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()
