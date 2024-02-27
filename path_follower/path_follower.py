# BSD 3-Clause License
#
# Copyright (c) 2022 University of Wisconsin - Madison
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.#
###############################################################################
## Author: Harry Zhang
###############################################################################

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped, TwistStamped
from chrono_ros_interfaces.msg import DriverInputs as VehicleInput
from chrono_ros_interfaces.msg import Body
from nav_msgs.msg import Path
from ament_index_python.packages import get_package_share_directory
import numpy as np
import os
import torch
os.environ["KERAS_BACKEND"] = "torch"
import csv 
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from keras_core.models import load_model

import sys
ament_tools_root = os.path.join(os.path.dirname(__file__), '.')
sys.path.insert(0, os.path.abspath(ament_tools_root))
from casadi import *
from casadi.tools import *
from mpc_wpts import mpc_wpts_solver

import time

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        
        # update frequency of this node
        self.freq = 10.0

        # READ IN SHARE DIRECTORY LOCATION
        package_share_directory = get_package_share_directory('path_follower')
        # initialize control inputs
        self.steering = 0.0
        self.throttle = 0.0
        self.braking = 0.0

        # initialize vehicle state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0


        # data that will be used by this class
        self.state = Body()
        self.path = Path()
        self.go = False
        self.vehicle_cmd = VehicleInput()
        self.lidar_data = LaserScan()
        self.model = load_model('/sbel/Desktop/ros_ws/src/nn_models_lib/single_speed_MPC_IL_NN.keras')
        self.file = open("/sbel/Desktop/waypoints_paths/path_iros_2.csv")
        self.ref_traj = np.loadtxt(self.file,delimiter=",")
        self.lookahead = 1.0
        # publishers and subscribers
        qos_profile = QoSProfile(depth=1)
        qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        self.sub_state = self.create_subscription(PoseStamped, '/chrono_ros_node/output/vehicle/state/pose', self.state_callback, qos_profile)
        self.sub_vel = self.create_subscription(TwistStamped, '/chrono_ros_node/output/vehicle/state/twist', self.vel_callback, qos_profile)
        self.pub_vehicle_cmd = self.create_publisher(VehicleInput, '/chrono_ros_node/input/driver_inputs', 10)
        #self.sub_PCdata = self.create_subscription(LaserScan,'/chrono_ros_node/output/lidar_2d/data/laser_scan',self.lidar_callback,qos_profile)
        self.timer = self.create_timer(1/self.freq, self.pub_callback)
    # subscribe manual control inputs
    # function to process data this class subscribes to
    def state_callback(self, msg):
        self.go = True 
        #self.get_logger().info("Received '%s'" % msg)
        self.state = msg
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        #convert quaternion to euler angles
        e0 = msg.pose.orientation.x
        e1 = msg.pose.orientation.y
        e2 = msg.pose.orientation.z
        e3 = msg.pose.orientation.w
        self.theta = np.arctan2(2*(e0*e3+e1*e2),e0**2+e1**2-e2**2-e3**2)
        
        #self.get_logger().info("(x, y, theta, v): (%s,%s,%s,%s)" % (self.x, self.y ,self.theta,self.v))
        
    def vel_callback(self, msg):
        self.v = np.sqrt(msg.twist.linear.x ** 2 + msg.twist.linear.y ** 2)
            
    
    def error_state(self):
        
        x_current = self.x
        y_current = self.y
        theta_current = self.theta
        v_current = self.v
        
        #post process theta
        while theta_current<-np.pi:
            theta_current = theta_current+2*np.pi
        while theta_current>np.pi:
            theta_current = theta_current - 2*np.pi

        dist = np.zeros((1,len(self.ref_traj[:,1])))
        for i in range(len(self.ref_traj[:,1])):
            dist[0][i] = dist[0][i] = (x_current+np.cos(theta_current)*self.lookahead-self.ref_traj[i][0])**2+(y_current+np.sin(theta_current)*self.lookahead-self.ref_traj[i][1])**2
        index = dist.argmin()

        ref_state_current = list(self.ref_traj[index,:])
        err_theta = 0
        ref = ref_state_current[2]
        act = theta_current

        if( (ref>0 and act>0) or (ref<=0 and act <=0)):
            err_theta = ref-act
        elif( ref<=0 and act > 0):
            if(abs(ref-act)<abs(2*np.pi+ref-act)):
                err_theta = -abs(act-ref)
            else:
                err_theta = abs(2*np.pi + ref- act)
        else:
            if(abs(ref-act)<abs(2*np.pi-ref+act)):
                err_theta = abs(act-ref)
            else: 
                err_theta = -abs(2*np.pi-ref+act)


        RotM = np.array([ 
            [np.cos(-theta_current), -np.sin(-theta_current)],
            [np.sin(-theta_current), np.cos(-theta_current)]
        ])

        errM = np.array([[ref_state_current[0]-x_current],[ref_state_current[1]-y_current]])

        errRM = RotM@errM


        error_state = [errRM[0][0],errRM[1][0],err_theta, ref_state_current[3]-v_current]

        return error_state

    # callback to run a loop and publish data this class generates
    def pub_callback(self):
        if(not self.go):
            return
        ## get error state
        e = self.error_state()
        self.get_logger().info('error state: %s' % e)
        ## NN controller 
        start_time = time.time()
        error_input = np.array(e).reshape(1,-1)
        ctrl_nn = self.model.predict(error_input)
        self.throttle = ctrl_nn[0][0]
        steering = ctrl_nn[0][1] 
        self.get_logger().info('solving time: %s' % (time.time() - start_time))
        
        ## MPC controller
        # start_time = time.time()
        # self.throttle, steering = mpc_wpts_solver(e,[self.throttle,self.steering], self.v, 1)
        # steering = steering * 1.6667
        # self.get_logger().info('solving time: %s' % (time.time() - start_time))
        ## PID Controller
        # start_time = time.time()
        # self.throttle = sum([x * y for x, y in zip(e, [0.37013526 ,0.00507144, 0.15476554 ,1.0235402 ])])
        # steering = sum([x * y for x, y in zip(e, [0.02176878 , 0.72672704 , 0.78409284 ,-0.0105355 ])])
        # self.get_logger().info('solving time: %s' % (time.time() - start_time))
        
        # ensure steering can't change too much between timesteps, smooth transition
        delta_steering = steering - self.steering
        if abs(delta_steering) > 0.25:
            self.steering = self.steering + 0.25 * delta_steering / abs(delta_steering)
            self.get_logger().info("steering changed too much, smoothing")
        else:
            self.steering = steering
        
        ### for vehicle one
        msg = VehicleInput()
        msg.steering = np.clip(self.steering, -1.0, 1.0)
        msg.throttle = np.clip(self.throttle, 0, 1)
        msg.braking = np.clip(self.braking, 0, 1)
        ### for vehicle two
        #self.get_logger().info("sending vehicle inputs: %s" % msg)
        self.pub_vehicle_cmd.publish(msg)
        
        with open ('sim_pid.csv','a', encoding='UTF8') as csvfile:
                my_writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE, escapechar=' ')
                #for row in pt:
                my_writer.writerow([self.x, self.y, self.theta, e[0],e[1],e[2],e[3],msg.throttle,msg.steering])
                csvfile.close()



def main(args=None):
    rclpy.init(args=args)
    control = ControlNode()
    rclpy.spin(control)

    control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
