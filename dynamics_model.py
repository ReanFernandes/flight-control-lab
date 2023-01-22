from casadi import * 
import numpy as np

params = {  'm' : 0.0027,   # mass of quadcopter
            'l' : 30e-3,    # dist bw com and motor
            'k_f' : 4.86e-4, #thrust coeff
            'k_m' : 1.5e-7,  #moment coeff
            'I' : np.diag([16.5717e-5, 16.6556e-5, 29.2616e-5]), # Inertia matrix
            'g' : 9.81 } # gravity
# Define variables
nx = 12
nu = 4
x = SX.sym('x', nx, 1) # state vector [x, y, z, phi, theta, psi, xdot, ydot, zdot, phi_dot, theta_dot, psi_dot]
u = SX.sym('u', nu, 1)

def model_force_control():

    # control vector : [thrust, tau_phi, tau_theta, tau_psi]

    # define state equations Pose, attitude, velocity, angular velocity

    Pose_dot = vertcat( cos(x[5]) * x[6] - sin(x[5]) * x[7] ,
                        sin(x[5]) * x[6] + cos(x[5]) * x[7] ,
                        x[8] )

    Attitude_dot = vertcat( x[9] + sin(x[3]) * tan(x[4]) * x[10] + cos(x[3]) * tan(x[4]) * x[11],
                            cos(x[3]) * x[10] - sin(x[3]) * x[11],
                            (x[10] * sin(x[3])) / cos(x[4]) + (x[11] * cos(x[3])) / cos(x[4]) )

    Velocity_dot = vertcat( -sin(x[4]) * u[0] / params['m'],
                            sin(x[3]) * u[0] / params['m'],
                            + params['g'] - (cos(x[3]) * cos(x[4]) * u[0]) / params['m'] )

    Angular_velocity_dot = vertcat ( (u[1] + x[10] * x[11] * (params['I'][1,1] - params['I'][2,2])) / params['I'][0,0],
                                    (u[2] + x[9] * x[11] * (params['I'][2,2] - params['I'][0,0])) / params['I'][1,1],
                                    (u[3] + x[9] * x[10] * (params['I'][0,0] - params['I'][1,1])) / params['I'][2,2] )

    # define state derivative 
    xdot = vertcat(Pose_dot, Attitude_dot, Velocity_dot, Angular_velocity_dot)

    return xdot, x, u, nx, nu

def model_rpm_control():

    # control vector : [rpm1, rpm2, rpm3, rpm4]

    # thrust = params['k_f'] * (u[0]**2 + u[1]**2 + u[2]**2 + u[3]**2)

    # tau_phi = params['k_f'] * (u[0]**2 - u[1]**2 - u[2]**2 + u[3]**2) * params['l'] * -1

    # tau_theta = params['k_f'] * (-u[0]**2 - u[1]**2 + u[2]**2 + u[3]**2) * params['l'] * -1

    # tau_psi = params['k_m'] * (-u[0]**2 + u[1]**2 - u[2]**2 + u[3]**2) * -1


    pose_dot =  vertcat( cos(x[5]) * x[6] - sin(x[5]) * x[7] ,
                         sin(x[5]) * x[6] + cos(x[5]) * x[7] ,
                            x[8] )

    attitude_dot = vertcat( x[9] + sin(x[3]) * tan(x[4]) * x[10] + cos(x[3]) * tan(x[4]) * x[11],
                            cos(x[3]) * x[10] - sin(x[3]) * x[11],
                            (x[10] * sin(x[3])) / cos(x[4]) + (x[11] * cos(x[3])) / cos(x[4]) )

    velocity_dot = vertcat( -sin(x[4]) * params['k_f'] * (u[0]**2 + u[1]**2 + u[2]**2 + u[3]**2) / params['m'],
                            sin(x[3]) * params['k_f'] * (u[0]**2 + u[1]**2 + u[2]**2 + u[3]**2) / params['m'],
                            - params['g'] + (cos(x[3]) * cos(x[4]) * params['k_f'] * (u[0]**2 + u[1]**2 + u[2]**2 + u[3]**2)) / params['m'] )
    
    angular_velocity_dot = vertcat ( (params['k_f'] * (u[0]**2 - u[1]**2 - u[2]**2 + u[3]**2) * params['l'] * -1 + x[10] * x[11] * (params['I'][1,1] - params['I'][2,2])) / params['I'][0,0],
                                     (params['k_f'] * (-u[0]**2 - u[1]**2 + u[2]**2 + u[3]**2) * params['l'] * -1 + x[9] * x[11] * (params['I'][2,2] - params['I'][0,0])) / params['I'][1,1],
                                     (params['k_m'] * (-u[0]**2 + u[1]**2 - u[2]**2 + u[3]**2) * -1 + x[9] * x[10] * (params['I'][0,0] - params['I'][1,1])) / params['I'][2,2] )

    xdot = vertcat(pose_dot, attitude_dot, velocity_dot, angular_velocity_dot)

    return xdot, x, u, nx, nu








