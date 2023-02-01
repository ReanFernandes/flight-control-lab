from casadi import *
import numpy as np
from cost_func import cost
from dynamics_model import *

def NLP_direct_collocation(degree, Q, R, function_type, N, T, nlpopts = None):
    # get the cost
    print("function_type: " , function_type)
    xdot, x, u, nx, nu = eval("model_" + function_type + "()")
    L, x_ref = cost(Q, R, function_type)

    d = degree

    ########################################
    # build the Interpolation polynomials

    # Get collocation points
    tau_root = np.append(0, collocation_points(d, 'legendre'))

    # Coefficients of the collocation equation
    C = np.zeros((d+1,d+1))

    # Coefficients of the continuity equation
    D = np.zeros(d+1)

    # Coefficients of the quadrature function
    B = np.zeros(d+1)

    # Construct polynomial basis
    for j in range(d+1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d+1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(d+1):
            C[j,r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)

    ########################################
    # define continuous time dynamics
    f = Function('f', [x, u],[xdot, L], ['x', 'u'], ['xdot', 'L'])

    # Time step
    h = T / N

    # define the system bounds
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
        upper_control = [22, 22, 22, 22]
    elif function_type == "force_control":
        upper_control = [0.6292, 0.0102, 0.0102, 0.0076]   

    lower_control = [0, 0, 0, 0]

    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    g = []
    lbg = []
    ubg = []
    J = 0

    # for plotting the trajectory
    x_plot = []
    u_plot = []

    # create the start state parameter 
    # that will take the value of the initial state
    x0_hat = SX.sym('x0_hat', nx, 1)

    # create variable for the beginning of the state
    X0 = SX.sym('X0', nx, 1)
    w += [X0]
    x_plot += [X0]
    lbw += [ *lower_pose, *lower_att, *lower_vel, *lower_rate]
    ubw += [ *upper_pose, *upper_att, *upper_vel, *upper_rate]
    w0 += [ *initial_guess_state]

    # add constraint to make the start state equal to the initial state
    g += [X0 - x0_hat]
    lbg += [*equality_const]
    ubg += [*equality_const]

    Xk = x0_hat

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = SX.sym('U_' + str(k), nu, 1)
        w += [Uk]
        u_plot += [Uk]
        lbw += [*lower_control]
        ubw += [*upper_control]
        if function_type == "rpm_control":
            w0 += [*upper_control]
        else:
            w0 += [*initial_guess_control]

        # state at the collocation points
        Xc = []
        for j in range(d):
            Xkj = SX.sym('X_' + str(k)+'_'+str(j), nx, 1)
            Xc += [Xkj]
            w += [Xkj]
            lbw += [*lower_pose, *lower_att, *lower_vel, *lower_rate]
            ubw += [*upper_pose, *upper_att, *upper_vel, *upper_rate]
            w0 += [*initial_guess_state]

        # Loop over collocation points
        Xk_end = D[0] * Xk
        for j in range(1,d+1):
            # Expression for the state derivative at the collocation point
            xp = C[0,j] * Xk
            for r in range(d):
                xp = xp + C[r+1,j] * Xc[r]

            # Append collocation equations
            fj, qj = f(Xc[j-1], Uk)
            g += [h * fj - xp]
            lbg += [*equality_const]
            ubg += [*equality_const]

            # Add contribution to the end state
            Xk_end = Xk_end + D[j] * Xc[j-1]

            # Add contribution to quadrature function
            J = J + B[j] * qj * h
        
        # New NLP variable for state at end of interval
        Xk = SX.sym('X_' + str(k+1), nx, 1)
        w += [Xk]
        x_plot += [Xk]
        lbw += [*lower_pose, *lower_att, *lower_vel, *lower_rate]
        ubw += [*upper_pose, *upper_att, *upper_vel, *upper_rate]
        w0 += [*initial_guess_state]
        

        # Add equality constraint
        g += [Xk_end-Xk]
        lbg += [*equality_const]
        ubg += [*equality_const]

    # terminal equality constrain for the final state
    g += [Xk - x_ref]
    lbg += [*equality_const]
    ubg += [*equality_const]
    print ( "length of w0: ", len(w0))
    print ( "length of lbw: ", len(lbw))
    print ( "length of ubw: ", len(ubw))
    print ( "length of lbg: ", len(lbg))
    print ( "length of ubg: ", len(ubg))
    print("length of g: ", len(g))
    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g), 'p': vertcat(x0_hat, x_ref)}
    solver = nlpsol('solver', 'ipopt', prob, nlpopts)
    
    return solver, w0, lbw, ubw, lbg, ubg, x_plot, u_plot

def NLP_multiple_shooting(Q, R, function_type, N, T, nlpopts):
    # get the cost
    xdot, x, u, nx, nu = eval("model_" + function_type + "()")
    L, x_ref = cost(Q, R, function_type)


    # define integrator to integratre the dynamics and cost
    # at between the control intervals

    M = 2 # RK4 steps per interval
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
    elif function_type == "force_control":
        upper_control =[0.6292, 0.0102, 0.0102, 0.0076]     

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
        if function_type == "rpm_control":
            w0 += [*upper_control]
        else:
            w0 += [*initial_guess_control]

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
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g), 'p':vertcat(x0_hat, x_ref)}
    solver = nlpsol('solver', 'ipopt', prob, nlpopts)

    return solver, w0, lbw, ubw, lbg, ubg






