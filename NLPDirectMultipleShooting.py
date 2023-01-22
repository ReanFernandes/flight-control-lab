from casadi import * 
import numpy as np
from cost_func import cost
def NLP_multiple_shooting(Q, R, function_type, N, T, nlpopts):
    # get the cost
    xdot, x, u, nx, nu = eval("model_" + function_type + "()")
    L, x_ref = cost(Q, R, function_type)


    # define integrator to integratre the dynamics and cost
    # at between the control intervals

    M = 4 # RK4 steps per interval
    DT = T/N/M
    f = Function('f', [x, u], [xdot, L])
    X0 = SX.sym('X0', nx, 1)
    U = SX.sym('U', nu, 1)
    X = X0
    Q = 0
    for j in range(M):
        k1, k1_q = f(X, U)
        k2, k2_q = f(X + DT/2 * k1, U)
        k3, k3_q = f(X + DT/2 * k2, U)
        k4, k4_q = f(X + DT * k3, U)
        X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
        Q=Q+DT/6*(k1_q +2*k2_q +2*k3_q +k4_q)
    F = Function('F', [X0, U], [X, Q], ['x0','p'], ['xf', 'qf'])

    # define the constraints 
    # path constraints
    initial_guess_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    initial_guess_control = [0, 0, 0, 0]
    equality_const = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    upper_pose = [10, 10, 10]
    lower_pose = [-10, -10, 0]
    
    upper_att = [pi/3, pi/3, pi]
    lower_att = [-pi/3, -pi/3, -pi]
    
    upper_vel = [10, 10, 10]
    lower_vel = [-10, -10, -10]
    
    upper_rate = [pi, pi, pi]
    lower_rate = [-pi, -pi, -pi]

    # control constraints
    if function_type == "rpm_control":
        upper_control = [22000, 22000, 22000, 22000]
    elif function_type == "thrust_control":
        upper_control = [10, 10, 10, 10]   

    lower_control = [0, 0, 0, 0]

    #start with empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    g = []
    lbg = []
    ubg = []
    J = 0

    # create the start state parameter 
    # that will take the value of the initial state
    x0_hat = SX.sym('x0_hat', nx, 1)

    # create variable for the beginning of the state
    X0 = SX.sym('X0', nx, 1)
    w += [X0]
    lbw += [ *lower_pose, *lower_att, *lower_vel, *lower_rate]
    ubw += [ *upper_pose, *upper_att, *upper_vel, *upper_rate]
    w0 += [ *initial_guess_state]

    # add constraint to make the start state equal to the initial state
    g += [X0 - x0_hat]
    lbg += [*equality_const]
    ubg += [*equality_const]

    Xk = X0

    # formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = SX.sym('U_' + str(k), nu, 1)
        w += [Uk]
        lbw += [ *lower_control]
        ubw += [ *upper_control]
        w0 += [ *initial_guess_control]

        # Integrate till the end of the interval
        Fk = F(x0=Xk, p=Uk)
        Xk_end = Fk['xf']
        J=J+Fk['qf']

        # New NLP variable for state at end of interval
        Xk = SX.sym('X_' + str(k+1), nx, 1)
        w += [Xk]
        lbw += [ *lower_pose, *lower_att, *lower_vel, *lower_rate]
        ubw += [ *upper_pose, *upper_att, *upper_vel, *upper_rate]
        w0 += [ *initial_guess_state]

        # Add equality constraint
        g += [Xk_end-Xk]
        lbg += [*equality_const]
        ubg += [*equality_const]
    
    # end state constraint
    g += [Xk - x_ref]
    lbg += [*equality_const]
    ubg += [*equality_const]

    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g), 'p':vertcat[x0_hat, x_ref]}
    solver = nlpsol('solver', 'ipopt', prob, nlpopts)

    return solver, w0, lbw, ubw, lbg, ubg
