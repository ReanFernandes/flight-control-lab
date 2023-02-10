# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2016 Bitcraze AB
#
#  Crazyflie Nano Quadcopter Client
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
Simple example that connects to one crazyflie (check the address at the top
and update it to your crazyflie address) and send a sequence of setpoints,
one every 5 seconds.
This example is intended to work with the Loco Positioning System in TWR TOA
mode. It aims at documenting how to set the Crazyflie in position control mode
and how to send setpoints.
"""
import time
import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper
from NMPC_solver import solver

##################################################
# define the solver params 
Q = np.diag([120,
            100,
            100,
            1,
            1,
            1,
            1,
            7e-1,
            1.0,
            4.0,
            1e-2,
            1e-2,
            1e-2])
R = np.diag([1, 1, 1, 1])*0.1

solver_bounds = {"upper_pose_limit":[1, 1, 1.5],
                "lower_pose_limit":[0, 0, 0],
                "upper_vel_limit":[10, 10, 10],
                "lower_vel_limit":[-10, -10, -10],
                "upper_att_limit":[1,1,1,1],
                "lower_att_limit":[0,-1,-1,-1],
                "upper_ang_limit":[10, 10, 10],
                "lower_ang_limit":[-10, -10, -10],
                "u_min" : [ 0, 0, 0, 0],
                "u_max" : [ 22, 22, 22, 22]}

nlp_opts = {"ipopt": {"max_iter": 3000, "print_level" :0}, "print_time":0}
cost_type = "slack"      # use slack variables for the terminal cost          
# time for dms closed loop
dms_timing = { "frequency" : 50,         # sampling frequency
            "solution_time" : 0.1,      # real world time to navigate the drone
            "N" : 10    ,   
            "DMS_RK4_step_size" : 2, # step size of RK4 method for DMS
            "degree" : 2,
            "step_multiplier":2}           # degree of the collocation polynomial  
# time for dc closed loop
dc_timing = { "frequency" : 50,         # sampling frequency
            "solution_time" : 0.1, 
            "N" : 15   , # real world time to navigate the drone
            "DMS_RK4_step_size" : 2, # step size of RK4 method for DMS
            "degree" : 3,
            "step_multiplier":10}       

min_deviation = 0.05

# URI to the Crazyflie to connect to
uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

# Change the sequence according to your setup
#             x    y    z  YAW
# sequence = [
#     (0.0, 0.0, 0.4, 0),
#     (0.0, 0.0, 1.2, 0),
#     (0.5, -0.5, 1.2, 0),
#     (0.5, 0.5, 1.2, 0),
#     (-0.5, 0.5, 1.2, 0),
#     (-0.5, -0.5, 1.2, 0),
#     (0.0, 0.0, 1.2, 0),
#     (0.0, 0.0, 0.4, 0),
# ]
def extract_control_input():
    control_input  = np.loadtxt('formatted_code/control_array.txt')
    control_input = control_input.reshape(100,4)
    return control_input.tolist()

sequence = extract_control_input()
# def extract_position():
#     position = np.loadtxt('position.txt')
#     position = position.reshape(130,3)
#     return position.tolist()

# sequence = extract_position()

control_sequence = []

def wait_for_position_estimator(scf):
    print('Waiting for estimator to find position...')

    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data['kalman.varPX'])
            print("data",  data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varPZ'])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            # print("{} {} {}".
            #       format(max_x - min_x, max_y - min_y, max_z - min_z))

            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break


def reset_estimator(scf):
    cf = scf.cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')

    wait_for_position_estimator(cf)


def state_callback(timestamp, data, logconf):
    roll = data['stateEstimate.x']
    pitch = data['stateEstimate.y']
    yaw_rate = data['stateEstimate.z']
    # thrust = data['controller.actuatorThrust']
    control_sequence.append([roll,pitch,yaw_rate])
    # control_sequence.append([roll,pitch])
    print('pos: ({}, {}, {})'.format(roll, pitch, yaw_rate))
    # print('control: ({}, {})'.format(roll, pitch))


def start_position_printing(scf):
    pos_conf = LogConfig(name='Position', period_in_ms=20)
    pos_conf.add_variable('stateEstimate.x', 'float')
    pos_conf.add_variable('stateEstimate.y', 'float')
    pos_conf.add_variable('stateEstimate.z', 'float')

    quat_conf = LogConfig(name='Quaternion', period_in_ms=20)
    quat_conf.add_variable('stateEstimate.qw', 'float')
    quat_conf.add_variable('stateEstimate.qx', 'float')
    quat_conf.add_variable('stateEstimate.qy', 'float')
    quat_conf.add_variable('stateEstimate.qz', 'float')
    
    vel_conf = LogConfig(name='Velocity', period_in_ms=20)
    vel_conf.add_variable('stateEstimate.vx', 'float')
    vel_conf.add_variable('stateEstimate.vy', 'float')
    vel_conf.add_variable('stateEstimate.vz', 'float')

    rate_conf = LogConfig(name='Rate', period_in_ms=20)
    rate_conf.add_variable('stateEstimate.rateRoll', 'int16_t')
    rate_conf.add_variable('stateEstimate.ratePitch', 'int16_t')
    rate_conf.add_variable('stateEstimate.rateYaw', 'int16_t')


    # log_conf.add_variable('controller.actuatorThrust', 'float')
    scf.cf.log.add_config(pos_conf, quat_conf, vel_conf, rate_conf)
    pos_conf.data_received_cb.add_callback(state_callback)
    quat_conf.data_received_cb.add_callback(state_callback)
    vel_conf.data_received_cb.add_callback(state_callback)
    rate_conf.data_received_cb.add_callback(state_callback)
    pos_conf.start()
    quat_conf.start()
    vel_conf.start()
    rate_conf.start()
    


def run_sequence(scf, sequence):
    cf = scf.cf
    cf.commander.send_setpoint(0,0,0,0) # unlock thrust control
    for position in sequence:
        # print('Setting position {}'.format(position))
        for i in range(29):
            cf.commander.send_setpoint(position[0],
                                                position[1],
                                                position[2],
                                                int(position[3]))
            time.sleep(0.75)

    cf.commander.send_stop_setpoint()
    # Make sure that the last packet leaves before the link is closed
    # since the message queue is not flushed before closing
    time.sleep(0.1)


if __name__ == '__main__':
    cflib.crtp.init_drivers()
    # initialise solvers
    DMS = solver( timing = dms_timing,
                  solver_bounds = solver_bounds,
                  nlp_opts = nlp_opts,
                  Q = Q,
                  R = R,
                  cost_type = "slack",
                  simulation_type = None
    )



    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        # reset_estimator(scf)
        # start_position_printing(scf)
        run_sequence(scf, sequence)

    with open('position_sequence.txt', 'w') as f:
        for item in control_sequence:
            f.write("%s " % item[0])
            f.write("%s " % item[1])
            f.write("%s " % item[2])
            f.write("%s " % item[3])