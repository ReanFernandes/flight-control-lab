from casadi import * 
import numpy as np
import matplotlib.pyplot as plt

# this class implements the quadrotor dynamics using quaternions and solve the NMPC problem
class NMPC_quat:
    def __init__(self, Q, R, N, T, Tf, method, degree = None, nlpopts_dc = None, nlpopts_dms = None):
        # define the casadi variables for the system dynamics
        self.nx = 13
        self.nu = 4
        self.nlp_opts_dms = nlpopts_dms
        self.nlp_opts_dc = nlpopts_dc
        # state variables
        self.xq = SX.sym('xq')
        self.yq = SX.sym('yq')
        self.zq = SX.sym('zq')
        self.q1 = SX.sym('q1')
        self.q2 = SX.sym('q2')
        self.q3 = SX.sym('q3')
        self.q4 = SX.sym('q4')
        self.vbx = SX.sym('vbx')
        self.vby = SX.sym('vby')
        self.vbz = SX.sym('vbz')
        self.wx = SX.sym('wx')
        self.wy = SX.sym('wy')
        self.wz = SX.sym('wz')

        self.x = vertcat( self.xq, self.yq, self.zq, self.q1, self.q2, self.q3, self.q4, self.vbx, self.vby, self.vbz, self.wx, self.wy, self.wz)
        # control variables
        self.w1 = SX.sym('w1')
        self.w2 = SX.sym('w2')
        self.w3 = SX.sym('w3')
        self.w4 = SX.sym('w4')

        self.u =vertcat(self.w1, self.w2, self.w3, self.w4)

        # system ode 
        self.xdot = None

        # define drone parameters
        self.g0  = 9.8066     # [m.s^2] accerelation of gravity
        self.mq  = 40e-3      # [kg] total mass (with one marker)
        self.Ixx = 1.395e-5   # [kg.m^2] Inertia moment around x-axis
        self.Iyy = 1.395e-5   # [kg.m^2] Inertia moment around y-axis
        self.Izz = 2.173e-5   # [kg.m^2] Inertia moment around z-axis
        self.Cd  = 7.9379e-06 # [N/krpm^2] Drag coef
        self.Ct  = 3.25e-4    # [N/krpm^2] Thrust coef
        self.dq  = 65e-3      # [m] distance between motors' center
        self.l   = self.dq/2       # [m] distance between motors' center and the axis of rotation

        # assign the OCP parameters
        self.Q = Q
        self.R = R
        self.N = N
        self.T = T
        self.Tf = Tf
        self.x_init = None
        self.x_des = None
        self.N_sim = int(Tf * N/T)
        self.nlpopts_dc = nlpopts_dc
        self.nlpopts_dms = nlpopts_dms
        

        # define the system constraints
        self.u_max = 22
        self.initial_guess_control = [ self.u_max, self.u_max, self.u_max, self.u_max]
        # self.initial_guess_control = [ 0, 0, 0, 0]
        self.initial_guess_state = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # state and control bounds
        self.lower_pose = [-1, -1, 0]
        self.upper_pose = [2, 2, 2]
        self.lower_quat = [-1, -1, -1, -1]
        self.upper_quat = [1, 1, 1, 1]
        self.lower_vel = [-1, -1, -1]
        self.upper_vel = [1, 1, 1]
        self.upper_rate = [1, 1, 1]
        self.lower_rate = [-1, -1, -1]
        self.lower_bound_state = [ *self.lower_pose, *self.lower_quat, *self.lower_vel, *self.lower_rate]
        self.upper_bound_state = [ *self.upper_pose, *self.upper_quat, *self.upper_vel, *self.upper_rate]
        # control bounds
        self.lower_bound_control = [0, 0, 0, 0]
        self.upper_bound_control = [self.u_max, self.u_max, self.u_max, self.u_max]

        # define the parameters of the solver 
        self.x_ref = SX.sym('x_ref', self.nx, 1)
        self.x0_hat = SX.sym('x0_hat', self.nx, 1)
        self.u_prev = SX.sym('u_prev', self.nu, 1)
        self.L = None # cost function

        # DMS parameters 
        self.M = 2 # number of ERK4 stages
        self.integrator = None
        # DC parameters 
        
        self.C = None
        self.D = None
        self.B = None

        # define the solver variables
        self.w = []
        self.w0 = []
        self.g = []
        self.lbg = []
        self.ubg = []
        self.lbw = []
        self.ubw = []
        self.J = 0
        self.solver = None

        # define the variables used in the mpc
        self.X_mpc = np.zeros((self.N_sim+1, self.nx))
        # self.X_mpc[0, :] = self.x_init  # set the initial state of the mpc
        self.U_mpc = np.zeros((self.N_sim, self.nu))
        self.w_opt = None

        # define the X_opt, U_opt, X_pred, U_pred for the current step
        self.X_opt_current = None
        self.U_opt_current = None
        self.X_pred_current = None
        self.U_pred_current = None
        self.pose_dev_current = None
        self.total_dev_current = None

        # define the deviation and logging variables to save to text file
        self.X_opt_full = None
        self.U_opt_full = None

        self.pose_dev = []
        self.total_dev = []

        # create the model and the cost function
        self._model()
        self._cost()

        #set the solver based on the method
        if method == "DMS":
            self._rk4()
            self._dms_solver()
        elif method == "DC":
            self._dc_solver(degree)


    def set_values(self, x_init, x_des):
        self.x_init = x_init
        self.x_des = x_des
        self.X_opt_current = self.x_init
    
    def _model(self):
        dxq = self.vbx*(2*self.q1**2 + 2*self.q2**2 - 1) - self.vby*(2*self.q1*self.q4 - 2*self.q2*self.q3) + self.vbz*(2*self.q1*self.q3 + 2*self.q2*self.q4)
        dyq = self.vby*(2*self.q1**2 + 2*self.q3**2 - 1) + self.vbx*(2*self.q1*self.q4 + 2*self.q2*self.q3) - self.vbz*(2*self.q1*self.q2 - 2*self.q3*self.q4)
        dzq = self.vbz*(2*self.q1**2 + 2*self.q4**2 - 1) - self.vbx*(2*self.q1*self.q3 - 2*self.q2*self.q4) + self.vby*(2*self.q1*self.q2 + 2*self.q3*self.q4)
        dq1 = - (self.q2*self.wx)/2 - (self.q3*self.wy)/2 - (self.q4*self.wz)/2
        dq2 = (self.q1*self.wx)/2 - (self.q4*self.wy)/2 + (self.q3*self.wz)/2
        dq3 = (self.q4*self.wx)/2 + (self.q1*self.wy)/2 - (self.q2*self.wz)/2
        dq4 = (self.q2*self.wy)/2 - (self.q3*self.wx)/2 + (self.q1*self.wz)/2
        dvbx = self.vby*self.wz - self.vbz*self.wy + self.g0*(2*self.q1*self.q3 - 2*self.q2*self.q4)
        dvby = self.vbz*self.wx - self.vbx*self.wz - self.g0*(2*self.q1*self.q2 + 2*self.q3*self.q4)
        dvbz = self.vbx*self.wy - self.vby*self.wx - self.g0*(2*self.q1**2 + 2*self.q4**2 - 1) + (self.Ct*(self.w1**2 + self.w2**2 + self.w3**2 + self.w4**2))/self.mq
        dwx = -(self.Ct*self.l*(self.w1**2 + self.w2**2 - self.w3**2 - self.w4**2) - self.Iyy*self.wy*self.wz + self.Izz*self.wy*self.wz)/self.Ixx
        dwy = -(self.Ct*self.l*(self.w1**2 - self.w2**2 - self.w3**2 + self.w4**2) + self.Ixx*self.wx*self.wz - self.Izz*self.wx*self.wz)/self.Iyy
        dwz = -(self.Cd*(self.w1**2 - self.w2**2 + self.w3**2 - self.w4**2) - self.Ixx*self.wx*self.wy + self.Iyy*self.wx*self.wy)/self.Izz

        self.xdot = vertcat(dxq, dyq, dzq, dq1, dq2, dq3, dq4, dvbx, dvby, dvbz, dwx, dwy, dwz)

    def _cost(self):
        self.L = mtimes(mtimes((self.x - self.x_ref).T, self.Q), (self.x - self.x_ref)) + mtimes(mtimes((self.u - self.u_prev).T, self.R), (self.u - self.u_prev))

    def _rk4(self):
        h = self.T / self.N
        DT = h / self.M
        f = Function('f', [self.x, self.u], [self.xdot, self.L])
        X0 = SX.sym('X0', self.nx, 1)
        U = SX.sym('U', self.nu, 1)
        X = X0
        Q = 0
        for j in range(self.M):
            k1, k1_q = f(X, U)
            k2, k2_q = f(X + DT/2 * k1, U)
            k3, k3_q = f(X + DT/2 * k2, U)
            k4, k4_q = f(X + DT * k3, U)
            X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
            Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
        self.integrator = Function('F', [X0, U], [X, Q], ['x0','p'],['xf','qf'])
    
    def _dms_solver(self):
        # create the state variables for the beginning of the state
        X0 = SX.sym('X0', self.nx, 1)
        self.w += [X0]
        self.w0 += [*self.initial_guess_state]
        self.lbw += [*self.lower_bound_state]
        self.ubw += [*self.upper_bound_state]

        # add the equality constraint the make the starting state of the shooting interval equal to the prvious starting state,4
        # which will be passed as parameter x0_hat
        self.g += [X0 - self.x0_hat]
        self.lbg += [0]*self.nx
        self.ubg += [0]*self.nx

        Xk = X0

        # formulate the shooting point nodes
        for k in range(self.N):
            # new nlp variable for the control during the interval
            Uk = SX.sym('U_' + str(k), self.nu, 1)
            self.w += [Uk]
            self.w0 += [*self.initial_guess_control]
            self.lbw += [*self.lower_bound_control]
            self.ubw += [*self.upper_bound_control]

            # Integrate using Runge-Kutta 4 integrator
            Fk, Qk = self.integrator(x0=Xk, p=Uk)
            Xk_end = Fk     # the end state of this interval
            self.J += Qk   # the cost of this interval, added to the total cost

            # New NLP variable for state at end of interval
            Xk = SX.sym('X_' + str(k+1), self.nx, 1)
            self.w += [Xk]
            self.w0 += [*self.initial_guess_state]
            self.lbw += [*self.lower_bound_state]
            self.ubw += [*self.upper_bound_state]

            # Add equality constraint
            self.g += [Xk_end - Xk]
            self.lbg += [0]*self.nx
            self.ubg += [0]*self.nx
        
        # add the equality constraint the make the final state of the shooting interval equal to the desired state
        self.g += [Xk - self.x_ref]
        self.lbg += [0]*self.nx
        self.ubg += [0]*self.nx
        
        self.w_opt = np.zeros(len(self.w0))

        # Create an NLP solver
        prob = {'f': self.J, 'x': vertcat(*self.w), 'g': vertcat(*self.g), 'p': vertcat(self.x0_hat, self.u_prev, self.x_ref)}
        self.solver = nlpsol('solver', 'ipopt', prob, self.nlp_opts_dms)


 
