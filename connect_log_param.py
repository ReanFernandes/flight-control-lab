import logging
import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
from class_NMPC import NMPC


uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
logging.basicConfig(level=logging.ERROR)

def simple_log(scf, logconf):
    with SyncLogger(scf, lg_stab) as logger:

        for log_entry in logger:

            timestamp = log_entry[0]
            data = log_entry[1]
            logconf_name = log_entry[2]

            print('[%d][%s]: %s' % (timestamp, logconf_name, data))

            

def simple_connect():

    print("Connected")
    time.sleep(10)
    print("Disconnecting")

if __name__ == '__main__':
    # initialise low level drivers

    cflib.crtp.init_drivers()
    lg_stab = LogConfig(name='Stabilizer', period_in_ms=50)
    lg_stab.add_variable('stabilizer.roll', 'float')
    lg_stab.add_variable('stabilizer.pitch', 'float')
    lg_stab.add_variable('stabilizer.yaw', 'float')
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:

        # simple_connect()
        simple_log(scf, lg_stab)
