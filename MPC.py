from casadi import *
import numpy as np
from shooting_method import NLP_multiple_shooting, NLP_direct_collocation
import time

nx = 12
nu = 4
def MPC_multiple_shooting( Q, R, function_type, x_init, x_desired, N, T, Tf, nlpopts = None):
    # get the Solver
    
    solver, w0, lbw, ubw, lbg, ubg = NLP_multiple_shooting(Q, R, function_type, N, T, nlpopts)
   

    
    # set the number of steps
    N_sim = int(Tf/T/N)
    print('N_sim for Multiple Shooting: ', N_sim)
    print('Length of w0: ', len(w0))

    # set the initial state
    x0 = np.concatenate((x_init, np.zeros(nx-3)))
    x_ref = np.concatenate((x_desired, np.zeros(nx-3)))

    # set optimisation variables
    w_opt = np.zeros(len(w0))
    # if function_type == "rpm_control":
    #     w_opt[nx:nx+nu] = np.array([22, 22, 22, 22])

    X_mpc = np.zeros((N_sim+1, nx))
    U_mpc = np.zeros((N_sim, nu))

    # set the start state in X_mpc
    X_mpc[0,:] = x0
    timer = 0
    stable_state_counter = 0
    shift = [*np.zeros(nx), *np.zeros(nu)]
    deviation = []
    for i in range(N_sim+1):
        # time the method
        start = time.time()
        print('step: ', i)
        #shift initialisation
        w0 = [*w_opt[nx+nu:], *shift]
        
        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=vertcat(X_mpc[i,:], x_ref) )
        w_opt = sol['x'].full().flatten()
        # print('w_opt: ', w_opt)
        # save w_opt to a text file 
        # x_pred[:,] = [w_opt[nx+nu+nx*j:nx+nu+nx*(j+1)] for j in range(N)]
        # u_pred[:,] = [w_opt[nx+nu*j:nx+nu*(j+1)] for j in range(N)]
        # plot the predicted trajectory and control in the same figure, and show only current values
        # print('x_ref: ', x_ref)
        

        # Retrieve the solution
        X_mpc[i+1,:] = w_opt[nx+nu:nx+nu+nx]
        U_mpc[i,:] = w_opt[nx:nx+nu]
      
        
        print('X_mpc: ', X_mpc[i+1,:])
        print('U_mpc: ', U_mpc[i,:])
        pose_deviation = np.linalg.norm(X_mpc[i+1,0:3] - x_desired)
        print('Deviation in euclidean distance: ', pose_deviation, 'm')
        deviation.append(pose_deviation)
        total_deviation = np.linalg.norm(X_mpc[i+1,:] - x_ref)

        # Implement stopping conditions
        if i == N_sim-1:
            print('MPC failed to converge after ',i,' steps')
            break

        if total_deviation < 0.01:
            stable_state_counter += 1
            if stable_state_counter == 300:
                print('MPC converged in ', i ,' steps')
                break
        end_time = time.time()
        timer += end_time - start
        print('Time taken for step ', i, ' is ', end_time - start, 's')

    # print the function type and time
    print('Function type: ', function_type)
    print('Time: ', timer, 's')
    print('average time per step for multiple shooting: ', timer/N_sim, 's')

    return X_mpc, U_mpc, deviation, i

def MPC_collocation(degree, Q, R, function_type, x_init, x_desired, N, T, Tf, nlpopts = None):
    # get the Solver
    solver, w0, lbw, ubw, lbg, ubg, x_plot, u_plot = NLP_direct_collocation(degree,Q, R, function_type, N, T, nlpopts)

    # set the number of steps
    N_sim = int(Tf/T/N)
    print('N_sim: ', N_sim)
    print('Length of w0: ', len(w0))

    # set the initial state
    x0 = np.concatenate((x_init, np.zeros(nx-3)))
    x_ref = np.concatenate((x_desired, np.zeros(nx-3)))

    # set optimisation variables
    w_opt = np.zeros(len(w0))
    print('size of w_opt: ', w_opt.shape)
    X_mpc = np.zeros((N_sim+1, nx))
    U_mpc = np.zeros((N_sim, nu))
    stable_state_counter = 0
    # set the start state in X_mpc
    X_mpc[0,:] = x0
    timer = 0
    shift = [*np.zeros(nx+ nu + nx*degree )]
    deviation = []
    for i in range(N_sim):
        start = time.time()
        print('step: ', i)
        #shift initialisation
        w0 = [*w_opt[nx+nu + nx *degree:], *shift]
        # w0 = w_opt
        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=vertcat(X_mpc[i,:], x_ref) )
        w_opt = sol['x'].full().flatten()

        # Retrieve the solution
        X_mpc[i+1,:] = w_opt[nx + nu +nx*degree :nx + nu +nx*degree  + nx]
        print('X_mpc: ', X_mpc[i+1,:])
        U_mpc[i,:] = w_opt[nx:nx+nu] # this is the first control input
        print('U_mpc: ', U_mpc[i,:])

        pose_deviation = np.linalg.norm(X_mpc[i+1,0:3] - x_desired)
        print('Deviation in euclidean distance: ', pose_deviation, 'm')

        deviation.append(pose_deviation)
        total_deviation = np.linalg.norm(X_mpc[i+1,:] - x_ref)

        # Implement stopping conditions
        if i >= N_sim-1:
            print('MPC failed to converge after ',i,' steps')
            break

        if total_deviation < 0.1:
            stable_state_counter += 1
            if stable_state_counter == 30:
                print('MPC converged in ', i, ' steps')
                break

        end_time = time.time()
        timer += end_time - start
        print('Time taken for step ', i, ' is ', end_time - start, 's')


    # print method type, function type and time
    print('Method type: Direct Collocation')
    print('Function type: ', function_type)
    print('Time: ', timer, 's')
    print('average time per step for collocation: ', timer/N_sim, 's')
    return X_mpc, U_mpc, deviation, i


