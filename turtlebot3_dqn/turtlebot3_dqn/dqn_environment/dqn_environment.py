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


import math
import numpy

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Dqn


class DQNEnvironment(Node):
    def __init__(self, num_agents = 2):
        super().__init__('dqn_environment')

        """************************************************************
        ** Initialise variables
        ************************************************************"""
        self.goal_pose_x = 0.0
        self.goal_pose_y = 0.0
        self.last_pose_x = 0.0
        self.last_pose_y = 0.0
        self.last_pose_theta = 0.0
        self.num_agents = num_agents

        self.action_size = 5
        self.done = False
        self.fail = False
        self.succeed = False

        self.goal_angle = 0.0
        self.goal_distance = 1.0
        self.init_goal_distance = 1.0
        self.scan_ranges = []
        self.min_obstacle_distance = 10.0
        self.min_obstacle_angle = 10.0

        self.local_step = 0

        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        qos = QoSProfile(depth=10)

        # Initialise publishers
        # self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)
        self.cmd_vel_pub = [
            self.create_publisher(Twist, f'/robot{i}/cmd_vel', qos) for i in range(self.num_agents)
        ]

       
        # Initialise subscribers
        # self.goal_pose_sub = self.create_subscription(
        #     Pose,
        #     'goal_pose',
        #     self.goal_pose_callback,
        #     qos)
        self.goal_pose_subs = [
            self.create_subscription(Pose, f'/goal_pose_{i}', self.goal_pose_callback, qos) for i in range(self.num_agents)
        ]
                
        # self.odom_sub = self.create_subscription(
        #     Odometry,
        #     'odom',
        #     self.odom_callback,
        #     qos)
        # self.odom_sub = self.create_subscription(
        #     Odometry, '/robot0/odom', self.odom_callback, qos)
        
        self.odom_subs = [
            self.create_subscription(Odometry, f'/robot{i}/odom', self.odom_callback, qos) for i in range(self.num_agents)
        ]
        

        self.get_logger().info(f"Subscribed to goal_pose topics for {self.num_agents} agents.")
        # self.scan_sub = self.create_subscription(
        #     LaserScan,
        #     'scan',
        #     self.scan_callback,
        #     qos_profile=qos_profile_sensor_data)
        self.scan_subs = [
            self.create_subscription(LaserScan, f'/robot{i}/scan', self.scan_callback, qos) for i in range(self.num_agents)
        ]

        # Initialise client
        self.task_succeed_client = self.create_client(Empty, 'task_succeed')
        # self.scan_sub = self.create_subscription(
        #     LaserScan,
        #     'scan',
        #     self.scan_callback,
        #     qos_profile=qos_profile_sensor_data)

        # Initialise client
        # self.task_succeed_client = self.create_client(Empty, 'task_succeed')
        self.task_fail_client = self.create_client(Empty, 'task_fail')

        # Initialise servers
        # self.dqn_com_server = self.create_service(Dqn, 'dqn_com', self.dqn_com_callback)

        # for i in range(self.num_agents):
        #     service_name = f'dqn_com_agent_{i}'  # Unique service name per agent
        #     callback = getattr(self, f'dqn_com_callback_agent_{i}')  
        #     setattr(self, f'dqn_com_server_{i}', self.create_service(Dqn, service_name, callback))
        for i in range(self.num_agents):
            self.create_service(Dqn, f'dqn_com_agent_{i}', lambda req, res: self.dqn_com_callback(req, res, i))


        
    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""
    def goal_pose_callback(self, msg):
        self.goal_pose_x = msg.position.x
        self.goal_pose_y = msg.position.y

    def odom_callback(self, msg):
        self.last_pose_x = msg.pose.pose.position.x
        self.last_pose_y = msg.pose.pose.position.y
        _, _, self.last_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        goal_distance = math.sqrt(
            (self.goal_pose_x-self.last_pose_x)**2
            + (self.goal_pose_y-self.last_pose_y)**2)

        path_theta = math.atan2(
            self.goal_pose_y-self.last_pose_y,
            self.goal_pose_x-self.last_pose_x)

        goal_angle = path_theta - self.last_pose_theta
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi

        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = goal_distance
        self.goal_angle = goal_angle

    def scan_callback(self, msg):
        self.scan_ranges = msg.ranges
        self.min_obstacle_distance = min(self.scan_ranges)
        self.min_obstacle_angle = numpy.argmin(self.scan_ranges)

    # def get_state(self):
    #     state = list()
    #     state.append(float(self.goal_distance))
    #     state.append(float(self.goal_angle))
    #     state.append(float(self.min_obstacle_distance))
    #     state.append(float(self.min_obstacle_angle))
    #     self.local_step += 1

    #     # Succeed
    #     if self.goal_distance < 0.20:  # unit: m
    #         print("Goal! :)")
    #         self.succeed = True
    #         self.done = True
    #         self.cmd_vel_pub.publish(Twist())  # robot stop
    #         self.local_step = 0
    #         req = Empty.Request()
    #         while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
    #             self.get_logger().info('service not available, waiting again...')
    #         self.task_succeed_client.call_async(req)

    #     # Fail
    #     if self.min_obstacle_distance < 0.13:  # unit: m
    #         print("Collision! :(")
    #         self.fail = True
    #         self.done = True
    #         self.cmd_vel_pub.publish(Twist())  # robot stop
    #         self.local_step = 0
    #         req = Empty.Request()
    #         while not self.task_fail_client.wait_for_service(timeout_sec=1.0):
    #             self.get_logger().info('service not available, waiting again...')
    #         self.task_fail_client.call_async(req)

    #     if self.local_step == 500:
    #         print("Time out! :(")
    #         self.done = True
    #         self.local_step = 0
    #         req = Empty.Request()
    #         while not self.task_fail_client.wait_for_service(timeout_sec=1.0):
    #             self.get_logger().info('service not available, waiting again...')
    #         self.task_fail_client.call_async(req)

    #     return state

    # def reset(self):
    #     return self.state
    
    def get_state(self, agent_index):
        # Initialize the state for the specific agent
        state = list()
        state.append(float(self.goal_distance[agent_index]))
        state.append(float(self.goal_angle[agent_index]))
        state.append(float(self.min_obstacle_distance[agent_index]))
        state.append(float(self.min_obstacle_angle[agent_index]))
        self.local_step[agent_index] += 1

        # Succeed
        if self.goal_distance[agent_index] < 0.20:  # unit: m
            print(f"Agent {agent_index} reached the goal! :)")
            self.succeed[agent_index] = True
            self.done[agent_index] = True
            self.cmd_vel_pub[agent_index].publish(Twist())  # Stop the robot
            self.local_step[agent_index] = 0
            req = Empty.Request()
            while not self.task_succeed_client[agent_index].wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Service for agent {agent_index} not available, waiting again...')
            self.task_succeed_client[agent_index].call_async(req)

        # Fail
        if self.min_obstacle_distance[agent_index] < 0.13:  # unit: m
            print(f"Agent {agent_index} encountered a collision! :(")
            self.fail[agent_index] = True
            self.done[agent_index] = True
            self.cmd_vel_pub[agent_index].publish(Twist())  # Stop the robot
            self.local_step[agent_index] = 0
            req = Empty.Request()
            while not self.task_fail_client[agent_index].wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Service for agent {agent_index} not available, waiting again...')
            self.task_fail_client[agent_index].call_async(req)

        # Timeout
        if self.local_step[agent_index] == 500:
            print(f"Agent {agent_index} timed out! :(")
            self.done[agent_index] = True
            self.local_step[agent_index] = 0
            req = Empty.Request()
            while not self.task_fail_client[agent_index].wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Service for agent {agent_index} not available, waiting again...')
            self.task_fail_client[agent_index].call_async(req)

        return state

    def reset(self, agent_index):
        return self.get_state(agent_index)
    
    # def dqn_com_callback_agent_0(self, request, response):
    #     action = request.action
    #     twist = Twist()
    #     twist.linear.x = 0.3
    #     twist.angular.z = ((self.action_size - 1) / 2 - action) * 1.5
    #     self.cmd_vel_pub[0].publish(twist)  # Agent 0's command velocity

    #     response.state = self.get_state()
    #     response.reward = self.get_reward(action)
    #     response.done = self.done

    #     if self.done is True:
    #         self.done = False
    #         self.succeed = False
    #         self.fail = False

    #     if request.init is True:
    #         self.init_goal_distance = math.sqrt(
    #             (self.goal_pose_x - self.last_pose_x) ** 2
    #             + (self.goal_pose_y - self.last_pose_y) ** 2
    #         )

    #     return response

    # def dqn_com_callback_agent_1(self, request, response):
    #     action = request.action
    #     twist = Twist()
    #     twist.linear.x = 0.3
    #     twist.angular.z = ((self.action_size - 1) / 2 - action) * 1.5
    #     self.cmd_vel_pub[1].publish(twist)  # Agent 1's command velocity

    #     response.state = self.get_state()
    #     response.reward = self.get_reward(action)
    #     response.done = self.done

    #     if self.done is True:
    #         self.done = False
    #         self.succeed = False
    #         self.fail = False

    #     if request.init is True:
    #         self.init_goal_distance = math.sqrt(
    #             (self.goal_pose_x - self.last_pose_x) ** 2
    #             + (self.goal_pose_y - self.last_pose_y) ** 2
    #         )

    #     return response


    # def dqn_com_callback(self, request, response):
    #     action = request.action
    #     twist = Twist()
    #     twist.linear.x = 0.3
    #     twist.angular.z = ((self.action_size - 1) / 2 - action) * 1.5

    #     # Get the agent_index from the request to determine which agent is calling
    #     agent_index = request.agent_index  # This should be part of the request

    #     # Publish the command velocity to the correct agent's cmd_vel
    #     self.cmd_vel_pub[agent_index].publish(twist)

    #     # Update the state, reward, and done status for the agent
    #     response.state = self.get_state()
    #     response.reward = self.get_reward(action)
    #     response.done = self.done

    #     # Reset conditions if the task is done
    #     if self.done is True:
    #         self.done = False
    #         self.succeed = False
    #         self.fail = False

    #     # Initialize the goal distance if required
    #     if request.init is True:
    #         self.init_goal_distance = math.sqrt(
    #             (self.goal_pose_x - self.last_pose_x) ** 2
    #             + (self.goal_pose_y - self.last_pose_y) ** 2
    #         )

    #     return response
    
    def dqn_com_callback(self, request, response, agent_index):
        action = request.action

        # Publish velocity for the specific agent
        twist = Twist()
        twist.linear.x = 0.3
        twist.angular.z = ((self.action_size - 1) / 2 - action) * 1.5
        self.cmd_vel_pub[agent_index].publish(twist)

        # Update state and reward for the specific agent
        response.state = self.get_state(agent_index)
        response.reward = self.get_reward(action, agent_index)
        response.done = self.done[agent_index]

        # Reset conditions if needed
        if self.done[agent_index]:
            self.done[agent_index] = False
            self.succeed[agent_index] = False
            self.fail[agent_index] = False

        # Initialize goal distance for the agent
        if request.init:
            self.init_goal_distance[agent_index] = math.sqrt(
                (self.goal_pose_x[agent_index] - self.last_pose_x[agent_index]) ** 2 +
                (self.goal_pose_y[agent_index] - self.last_pose_y[agent_index]) ** 2
            )

        return response



    def get_reward(self, action):
        yaw_reward = 1 - 2*math.sqrt(math.fabs(self.goal_angle / math.pi))

        distance_reward = (2 * self.init_goal_distance) / \
            (self.init_goal_distance + self.goal_distance) - 1

        # Reward for avoiding obstacles
        if self.min_obstacle_distance < 0.25:
            obstacle_reward = -2
        else:
            obstacle_reward = 0

        reward = yaw_reward + distance_reward + obstacle_reward

        # + for succeed, - for fail
        if self.succeed:
            reward += 5
        elif self.fail:
            reward -= -10
        print(reward)

        return reward

    """*******************************************************************************
    ** Below should be replaced when porting for ROS 2 Python tf_conversions is done.
    *******************************************************************************"""
    def euler_from_quaternion(self, quat):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quat = [x, y, z, w]
        """
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w*x + y*z)
        cosr_cosp = 1 - 2*(x*x + y*y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w*y - z*x)
        pitch = numpy.arcsin(sinp)

        siny_cosp = 2 * (w*z + x*y)
        cosy_cosp = 1 - 2 * (y*y + z*z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    dqn_environment = DQNEnvironment()
    rclpy.spin(dqn_environment)

    dqn_environment.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
