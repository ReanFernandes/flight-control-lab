import logging
import numpy as np
import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
from class_NMPC import NMPC
from cflib.positioning.motion_commander import MotionCommander
import argparse

uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
logging.basicConfig(level=logging.ERROR)

DEFAULT_HEIGHT = 0.5
BOX_LIMIT = 0.5

def simple_log(scf, logconf):
    with SyncLogger(scf, lg_stab) as logger:

        for log_entry in logger:

            timestamp = log_entry[0]
            data = log_entry[1]
            logconf_name = log_entry[2]

            print('[%d][%s]: %s' % (timestamp, logconf_name, data))

def simple_param_async(scf, groupstr, namestr):
    cf = scf.cf
    full_name = groupstr+ "." +namestr
    # cf.param.add_update_callback(full_name,
    #                                 cb=param_stab_est_callback)
    cf.param.add_update_callback(group=groupstr, name=namestr,
                                    cb=param_stab_est_callback)
    time.sleep(1)
    cf.param.set_value(full_name, 2)
    time.sleep(1)
    cf.param.set_value(full_name, 1)
    time.sleep(1) 

def param_stab_est_callback(name, value):
    print('The crazyflie has parameter ' + name + ' set at number: ' + value)

def log_stab_callback(timestamp, data, logconf):
    print('[%d][%s]: %s' % (timestamp, logconf.name, data))

def argparse_init():
    '''
    Initialization for cmd line args
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--connect", action='store_true',
                        help="Connect to the drone then disconnects")
    parser.add_argument("-s", "--logsync", action='store_true',
                    help="Synchronous log of roll, pitch and yaw")
    parser.add_argument("-a", "--logasync", action='store_true',
                    help="Asynchronous log of roll, pitch and yaw")
    parser.add_argument("-p", "--param", action='store_true',
                    help="asynchronous set of estimator parameter")

    return parser

def simple_log_async(scf, logconf):
    cf = scf.cf
    cf.log.add_config(logconf)
    logconf.data_received_cb.add_callback(log_stab_callback)
    logconf.start()
    time.sleep(5)
    logconf.stop()           

def simple_connect():

    print("Connected")
    time.sleep(10)
    print("Disconnecting")

if __name__ == '__main__':
    # initialise low level drivers
    cflib.crtp.init_drivers()
    lg_kalman = LogConfig(name='kalman', period_in_ms=50)
    lg_kalman.add_variable('kalman.stateX', 'float') # x position in global frame
    lg_kalman.add_variable('kalman.stateY', 'float') # y position in global frame
    lg_kalman.add_variable('kalman.stateZ', 'float') # z position in global frame
    lg_kalman.add_variable('kalman.statePX', 'float') # x velocity in body frame
    lg_kalman.add_variable('kalman.statePY', 'float') # y velocity in body frame
    lg_kalman.add_variable('kalman.statePZ', 'float') # z velocity in body frame
    lg_stab = LogConfig(name='Stabilizer', period_in_ms=50)
    lg_stab.add_variable('stabilizer.roll', 'float') # roll angle in body frame
    lg_stab.add_variable('stabilizer.pitch', 'float') # pitch angle in body frame
    lg_stab.add_variable('stabilizer.yaw', 'float') # yaw angle in body frame
    
    lg_gyro = LogConfig(name='Gyro', period_in_ms=50)
    lg_gyro.add_variable('gyro.x', 'float') # roll rate in body frame
    lg_gyro.add_variable('gyro.y', 'float') # pitch rate in body frame
    lg_gyro.add_variable('gyro.z', 'float') # yaw rate in body frame

    # create parameters for NMPC object
    #
    
    Q = np.diag([120,   #x
                120,    #y
                120,    #z
                1e-2,   #phi
                1e-2,   #theta
                1e-2,   #psi
                7e-1,   #vx
                1,      #vy
                4,      #vz
                1e-5,   #phi_dot
                1e-5,   #theta_dot
                10])   #psi_dot

    R = np.diag([1, 1, 1, 1])* 0.6
    N = 10
    T = 0.01
    Tf = 0.1

    nlpopts_dc = {'ipopt': {'print_level': 0, 'max_iter':200}, 'print_time' : 0}
    nlp_opts_dms = {'ipopt': {'print_level': 0, 'max_iter':100}, 'print_time' : 0}

    # create NMPC object
    nmpc = NMPC(Q, R, N, T, Tf, nlpopts_dc, nlp_opts_dms)
    
    # Define solution method and create solver
    method = "DC" ;  degree = 2 ;
    # method = "DMS"
    nmpc.set_solver(method, degree)

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:

        # simple_connect()
        simple_log(scf, [lg_kalman, lg_stab, lg_gyro])

