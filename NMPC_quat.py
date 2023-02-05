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
        # self.hover_speed = np.sqrt(self.mq * self.g0 / (4 * self.Ct)) # [krpm] hover speed
        self.hover_speed = 18.1795 # [krpm] hover speed
        # assign the OCP parameters
        self.Q = Q
        self.R = R
        self.N = N
        self.T = T
        self.h = self.T / self.N
        self.Tf = Tf
        self.x_init = None
        self.x_des = None
        self.N_sim = int(Tf * N/T)
        self.nlpopts_dc = nlpopts_dc
        self.nlpopts_dms = nlpopts_dms
        

        # define the system constraints
        self.u_max = 22 # max rotor speed [krpm]
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

        # Solver related stuff
        self.method = method

        # DMS parameters 
        self.M = 2 # number of ERK4 stages
        self.integrator = None
        # DC parameters 
        self.degree = degree
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

        # variables to store the predictions for the whole mpc
        self.X_pred = np.zeros((self.N_sim+1,self.N-1, self.nx))
        self.U_pred = np.zeros((self.N_sim, self.N-1, self.nu))

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

        # store prediction flag
        self.store_prediction = False

        # list for storing converted phi, theta, psi
        self.phi = None
        self.theta = None
        self.psi = None
        self.phi_list = []
        self.theta_list = []
        self.psi_list = []
        self.thrust_list = []

        # variables for converting mpc output to drone control input
        self.max_thrust = 1.0942e-07 * (self.u_max*1000)**2 - 2.1059e-04 * (self.u_max*1000) + 1.5417e-01 # in grams
        self.thrust = None
        self.roll = None
        self.pitch = None
        self.yawrate = None


        # create the model and the cost function
        self._model()
        self._cost()

        #set the solver based on the method
        if self.method == "DMS":
            self._rk4()
            self._dms_solver()
        elif self.method == "DC":
            self._build_dc_polynomial()
            self.f = Function('f', [self.x, self.u],[self.xdot, self.L], ['self.x', 'self.u'],['xdot', 'self.L'])
            self._dc_solver()

    def set_values(self, x_init, x_des):
        self.x_init = np.concatenate((x_init, 1 , np.zeros(9)))
        self.x_des = np.concatenate((x_des, 1 , np.zeros(9)))
        self.X_opt_current = self.x_init
        if x_init[2] < 0.1:
            self.U_opt_current = np.array([self.u_max, self.u_max, self.u_max, self.u_max]) # drone near ground, give max thrust
        else:
            self.U_opt_current = np.array([self.hover_speed, self.hover_speed, self.hover_speed, self.hover_speed]) # drone hovering, start with hover speed
    
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
        
        DT = self.h / self.M
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

    def _build_dc_polynomial(self):
        degree = self.degree
        tau_root = np.append(0, collocation_points(degree, 'legendre'))
        
        # coefficients of the collocation equation
        self.C = np.zeros((degree+1, degree+1))

        # coefficients of the continuity equation
        self.D = np.zeros(degree+1)

        # coefficients of the quadrature function
        self.B = np.zeros(degree+1)

        # construct polynomial basis
        for j in range(degree+1):
            # construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for r in range(degree+1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

            # evaluate the polynomial at the final time to get the coefficients of the continuity equation
            self.D[j] = p(1.0)

            # evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            pder = np.polyder(p)
            for r in range(degree+1):
                self.C[j, r] = pder(tau_root[r])

            # evaluate the integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(p)
            self.B[j] = pint(1.0)

    def _dc_solver(self):
        # create the variable for the beginning of the state
        X0 = SX.sym('X0', self.nx, 1)
        self.w += [X0]
        self.w0 += [*self.initial_guess_state]
        self.lbw += [*self.lower_bound_state]
        self.ubw += [*self.upper_bound_state]

        # add the equality constraint the make the starting state of the shooting interval equal to the previous end state
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

            # state variable at the collocation points
            Xc = []
            for j in range(self.degree):
                Xkj = SX.sym('X_' + str(k) + '_' + str(j), self.nx, 1)
                Xc += [Xkj]
                self.w += [Xkj]
                self.w0 += [*self.initial_guess_state]
                self.lbw += [*self.lower_bound_state]
                self.ubw += [*self.upper_bound_state]

            # state approximation using the collocattion polynomials 
            Xk_end = self.D[0] * Xk
            for j in range(1, self.degree+1):
                # expression for state derivative at the collocation point
                xp = self.C[0, j] * Xk
                for r in range(self.degree):
                    xp += self.C[r+1, j] * Xc[r]
                
                # append the collocation equations
                fj, qj = self.f(Xc[j-1], Uk)
                self.g += [self.h * fj - xp]
                self.lbg += [0]*self.nx
                self.ubg += [0]*self.nx

                # add contribution to the end state
                Xk_end += self.D[j] * Xc[j-1]

                # add contribution to quadrature function
                self.J += self.h * qj * self.B[j]

            # new variable for the state at the end of the interval
            Xk = SX.sym('X_' + str(k+1), self.nx, 1)
            self.w += [Xk]
            self.w0 += [*self.initial_guess_state]
            self.lbw += [*self.lower_bound_state]
            self.ubw += [*self.upper_bound_state]

            # add equality constraint
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
        self.solver = nlpsol('solver', 'ipopt', prob, self.nlp_opts_dc)

    def solve_for_next_state(self):
        # solve the nlp 
        sol = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=vertcat(self.X_opt_current, self.U_opt_current, self.x_des))
        self.w_opt = sol['x'].full().flatten()

    def solve_open_loop(self):
            # solve the nlp 
            sol = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=vertcat(self.X_opt_current, self.U_opt_current ,self.x_des))
            self.w_opt = sol['x'].full().flatten()
            if self.method == "DMS":
                x_open_loop = np.vstack([self.w_opt[(self.nx+self.nu)*(i): (self.nx+self.nu)*(i) + self.nx] for i in range(self.N)])
                u_open_loop = np.vstack([self.w_opt[(self.nx+self.nu)*(i)+self.nx: (self.nx+self.nu)*(i)+self.nx + self.nu] for i in range(self.N)])
            elif self.method == "DC":
                x_open_loop = np.vstack([self.w_opt[(self.nx + self.nu + self.nx * self.degree)*i: (self.nx + self.nu + self.nx * self.degree)*i + self.nx] for i in range(self.N)])
                u_open_loop = np.vstack([self.w_opt[((self.nx + self.nu + self.degree*self.nx)*(i-1)+self.nx): ((self.nx + self.nu + self.degree*self.nx)*(i-1)+self.nx) + self.nu ] for i in range(self.N)])
            self.X_mpc = x_open_loop
            self.U_mpc = u_open_loop

    def extract_next_state(self, step, extract_predicted_state=False):
        # extract the optimal state from the solution
        if self.method == "DMS":
            self.X_opt_current = self.w_opt[self.nx + self.nu : self.nx + self.nu + self.nx]
            self.U_opt_current = self.w_opt[self.nx : self.nx + self.nu]
        elif self.method == "DC":
            self.X_opt_current = self.w_opt[ self.nx + self.nu + self.nx * self.degree: self.nx + self.nu + self.nx * self.degree + self.nx]
            self.U_opt_current = self.w_opt[self.nx : self.nx + self.nu]
        
        if extract_predicted_state:
            self._extract_prediction(step)
        
        self.X_mpc[step,:] = self.X_opt_current
        self.U_mpc[step,:] = self.U_opt_current
    
    def _extract_prediction(self, step):
        if self.method == "DC":
            # Extract the state and control predicitions over the horizon length N, from the control interval 2 to N
            self.X_pred_current = np.vstack([self.w_opt[(self.nx + self.nu + self.nx * self.degree)*i: (self.nx + self.nu + self.nx * self.degree)*i + self.nx] for i in range(2,self.N)])
            self.U_pred_current = np.vstack([self.w_opt[((self.nx + self.nu + self.degree*self.nx)*(i-1)+self.nx): ((self.nx + self.nu + self.degree*self.nx)*(i-1)+self.nx) + self.nu ] for i in range(2,self.N)])
        elif self.method == "DMS":
            # Extract the state predicitions over the horizon length, from the control interval 2 to N
            self.X_pred_current = np.vstack([self.w_opt[(self.nx+self.nu)*(i): (self.nx+self.nu)*(i) + self.nx] for i in range(2,self.N)])
            self.U_pred_current = np.vstack([self.w_opt[(self.nx+self.nu)*(i)+self.nx: (self.nx+self.nu)*(i)+self.nx + self.nu] for i in range(self.N)])
    
        if self.store_prediction:
            self.X_pred[step,:,:] = self.X_pred_current
            self.U_pred[step,:,:] = self.U_pred_current

    def _quaternion_to_euler(self):
        q = np.array(self.X_opt_current[3:7])
        w, x, y, z = q
        R11 = 2*(w*w + x*x) - 1
        R21 = 2*(x*y - w*z)
        R31 = 2*(x*z + w*y)
        R32 = 2*(y*z - w*x)
        R33 = 2*(w*w + z*z) - 1
        phi = np.arctan2(R32, R33)
        theta = -np.arcsin(R31)
        psi = np.arctan2(R21, R11)

        self.phi = phi
        self.theta = theta
        self.psi = psi

        if self.store_prediction:
            self.phi_list.append(phi)
            self.theta_list.append(theta)
            self.psi_list.append(psi)

    def _rpm_to_thrust(self):
        thrust = 1.0942e-07 * self.U_opt_current**2 - 2.1059e-04 * self.U_opt_current + 1.5417e-01
        return thrust
    
    def _total_thrust(self):
        thrust = self._rpm_to_thrust()
        total_thrust = np.sum(thrust)

        self.thrust = (total_thrust / self.max_thrust) * 50000 +10001 #convert the thurst to an integer value between 10001 and 60000
        if self.store_prediction:
            self.thrust_list.append(self.thrust)
  
    def _rad2deg(self, rad):
        return rad * 180 / np.pi  

    def control_to_drone(self):
        self._quaternion_to_euler()
        self._total_thrust()
        pitch = 1.0 * self._rad2deg(self.theta)
        roll = -1.0 * self._rad2deg(self.phi)
        yaw_rate = self._rad2deg(self.X_opt_current[10])

        return [roll, pitch, yaw_rate, self.thrust]

            


   


 
