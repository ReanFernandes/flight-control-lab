from casadi import * 
import numpy as np
import matplotlib.pyplot as plt

# this class defines the Dynamics, cost, constraints and the NMPC solver

class NMPC:
    def __init__(self, Q, R, function_type, x_init, x_desired, N, T, Tf, nlpopts_dc = None, nlpopts_dms = None):
        # define the variables for system dynamics
        self.nx = 12    
        self.nu = 4     
        self.x = SX.sym('x', self.nx, 1)   # state variable : [x, y, z, phi, theta, psi, vx, vy, vz, phi_dot, theta_dot, psi_dot]
        self.u = SX.sym('u', self.nu, 1)   # control variable : [ Thrust, Mx, My, Mz ]# assign parameters
        
        #assign ocp parameters
        self.Q = Q
        self.R = R
        self.function_type = function_type
        self.x_init = np.concatenate(x_init, np.zeros(self.nx - len(x_init)))
        self.x_desired  = np.concatenate(x_desired, np.zeros(self.nx - len(x_desired)))
        self.N = N
        self.T = T
        self.Tf = Tf
        self.N_sim = int(Tf/T/N)
        self.nlpopts_dc = nlpopts_dc
        self.nlpopts_dms = nlpopts_dms
        self.method = None

        
        # define system constraints
        # set initial guess for state and control
        self.initial_guess_control = [0] * self.nu
        self.initial_guess_state = [0] * self.nx
        #state constraints
        self.upper_pose = [10, 10, 10]
        self.lower_pose = [-10, -10, 0]
        self.upper_att = [pi/3, pi/3, pi]
        self.lower_att = [-pi/3, -pi/3, -pi]
        self.upper_vel = [10, 10, 10]
        self.lower_vel = [-10, -10, -10]
        self.upper_rate = [pi, pi, pi]
        self.lower_rate = [-pi, -pi, -pi]

        #control constraints
        self.upper_control = [0.6292, 0.0102, 0.0102, 0.0076]
        self.lower_control = [0, 0, 0, 0]

        # define the variables to be used in cost function
        self.x_setpoint = SX.sym('x_setpoint', self.nx, 1) # setpoint for state variable
        self.x0_hat = SX.sym('x0_hat', self.nx, 1)         # initial state variable
        self.L = 0                                      # cost function

        # define the number of steps for the RK4 integrator for DMS
        self.M = 2

        #define collocation degree
        self.degree = None

        # define the polynomials used in collocation
        self.C = None
        self.D = None
        self.B = None

        # define the variables to be used in constraints and the solver
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
        self.X_opt_current = self.x_init
        self.U_opt_current = None
        self.X_pred_current = None
        self.U_pred_current = None
        self.pose_dev_current = None
        self.total_dev_current = None

        # define the deviation and logging variables to save to text file
        self.X_opt_full = None
        self.U_opt_full = None

        self.pos_dev = []
        self.total_dev = []
    
    def set_solver(self, method, degree = None):
            # set the solver method
            self.method = method
            if method == "DC":
                if degree is None:
                    raise ValueError("Degree must be specified for DC method")
                self.degree = degree
                self._DCsolver()
            elif method == "DMS":
                self._DMSsolver()
            else:
                raise ValueError("Invalid method. Must be either DC or DMS")
                
    def _model(self):
        # define the system dynamics
        params = {  'm' : 0.0027,   # mass of quadcopter
            'l' : 30e-3,    # dist bw com and motor
            'k_f' : 4.86e-4, #thrust coeff
            'k_m' : 1.5e-7,  #moment coeff
            'I' : np.diag([16.5717e-5, 16.6556e-5, 29.2616e-5]), # Inertia matrix
            'g' : 9.81 } # gravity
        
         

        # define state equations Pose, attitude, velocity, angular velocity

        Pose_dot = vertcat( cos(self.x[5]) * self.x[6] - sin(self.x[5]) * self.x[7] ,
                            sin(self.x[5]) * self.x[6] + cos(self.x[5]) * self.x[7] ,
                            self.x[8] )

        Attitude_dot = vertcat( self.x[9] + sin(self.x[3]) * tan(self.x[4]) * self.x[10] + cos(self.x[3]) * tan(self.x[4]) * self.x[11],
                                cos(self.x[3]) * self.x[10] - sin(self.x[3]) * self.x[11],
                                (self.x[10] * sin(self.x[3])) / cos(self.x[4]) + (self.x[11] * cos(self.x[3])) / cos(self.x[4]) )

        Velocity_dot = vertcat( -sin(self.x[4]) * self.u[0] / params['m'],
                                sin(self.x[3]) * self.u[0] / params['m'],
                                + params['g'] - (cos(self.x[3]) * cos(self.x[4]) * self.u[0]) / params['m'] )

        Angular_velocity_dot = vertcat ( (self.u[1] + self.x[10] * self.x[11] * (params['I'][1,1] - params['I'][2,2])) / params['I'][0,0],
                                        (self.u[2] + self.x[9] * self.x[11] * (params['I'][2,2] - params['I'][0,0])) / params['I'][1,1],
                                        (self.u[3] + self.x[9] * self.x[10] * (params['I'][0,0] - params['I'][1,1])) / params['I'][2,2] )
        # define state derivative 
        xdot = vertcat(Pose_dot, Attitude_dot, Velocity_dot, Angular_velocity_dot)

        return xdot

    def _cost(self):
        # define the cost function
        cost = mtimes((self.x - self.x_setpoint).T, mtimes(self.Q, (self.x - self.x_setpoint))) + mtimes(self.u.T, mtimes(self.R, self.u))

        return cost

    def _DMSsolver(self):
        # create a solver instance for Direct Multiple Shooting
        xdot = self._model()

        # cost function
        self.L = self._cost()

        # Create RK4 integrator
        h = self.T / self.N
        DT = h / self.M
        f = Function('f', [self.x, self.u], [xdot, self.L])
        X0 = MX.sym('X0', self.nx)
        U = MX.sym('U', self.nu)
        X = X0
        Q = 0
        for j in range(self.M):
            k1, k1_q = f(X, U)
            k2, k2_q = f(X + DT/2 * k1, U)
            k3, k3_q = f(X + DT/2 * k2, U)
            k4, k4_q = f(X + DT * k3, U)
            X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
            Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
        F = Function('F', [X0, U], [X, Q], ['x0','p'],['xf','qf'])

        # create the variable for the beginning of the state
        X0 = MX.sym('X0', self.nx)
        self.w += [X0]
        self.lbw += [*self.lower_pose, *self.lower_att, *self.lower_vel, *self.lower_rate]
        self.ubw += [*self.upper_pose, *self.upper_att, *self.upper_vel, *self.upper_rate]
        self.w0 += [*self.initial_guess_state]

        # add constraint to make the start state equal to the initial state
        self.g += [X0 - self.x0_hat]
        self.lbg += [0] * self.nx
        self.ubg += [0] * self.nx

        Xk = X0

        # formulate the NLP
        for k in range(self.N):
            # new NLP variable for the control during the interval
            Uk = SX.sym('U_' + str(k), self.nu, 1)
            self.w += [Uk]
            self.lbw += [*self.lower_control]
            self.ubw += [*self.upper_control]
            self.w0 += [*self.initial_guess_control]

            # Integrate till the end of the interval
            Fk = F(x0=Xk, p=Uk)
            Xk_end = Fk['xf']           # the end state of this interval, integrated with RK4
            self.J = self.J + Fk['qf']  # Adding the cost to the total cost

            # new nlp variable for state at end of interval
            Xk = SX.sym('X_' + str(k+1), self.nx, 1)
            self.w += [Xk]
            self.lbw += [*self.lower_pose, *self.lower_att, *self.lower_vel, *self.lower_rate]
            self.ubw += [*self.upper_pose, *self.upper_att, *self.upper_vel, *self.upper_rate]
            self.w0 += [*self.initial_guess_state]

            # add equality constraint
            self.g += [Xk_end - Xk]
            self.lbg += [0] * self.nx
            self.ubg += [0] * self.nx
        
        # add the final state constraint
        self.g += [Xk - self.x_setpoint]
        self.lbg += [0] * self.nx
        self.ubg += [0] * self.nx
        
        # Initialise the optimal soln vector w_opt
        self.w_opt = np.zeros(len(self.w0))

        # create an NLP solver

        prob = {'f': self.J, 'x':vertcat(*self.w), 'g':vertcat(*self.g), 'p':vertcat(self.x0_hat, self.x_setpoint)}
        self.solver = nlpsol('solver','ipopt', prob, self.nlpopts_dms)

    def _build_dc_polynomials(self):
        # build interpolation polynomials for direct collocation
        #get the collocation points
        tau_root = np.append(0, collocation_points(degree, 'legendre'))
        degree = self.degree
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

    def _DCsolver(self):
        # create a solver instance for Direct Collocation
        xdot = self._model()
        # cost function
        self.L = self._cost()
        degree = self.degree
        # set the collocation polynomials
        self._build_dc_polynomials()

        # define the continuous time dynamics
        f = Function('f', [self.x, self.u], [xdot, self.L], ['x', 'u'], ['xdot', 'L'])

        # define the time step for discrete time dynamics
        h = self.T / self.N

        # create variable for the beginning of the state
        X0 = SX.sym('X0', self.nx, 1)
        self.w += [X0]
        self.lbw += [*self.lower_pose, *self.lower_att, *self.lower_vel, *self.lower_rate]
        self.ubw += [*self.upper_pose, *self.upper_att, *self.upper_vel, *self.upper_rate]
        self.w0 += [*self.initial_guess_state]

        # add constraint to make the start state equal to the initial state
        self.g += [X0 - self.x0_hat]
        self.lbg += [0] * self.nx
        self.ubg += [0] * self.nx

        Xk = self.x0_hat

        # formulate the NLP
        for k in range(self.N):
            # new nlp variable for the control during the interval
            Uk = SX.sym('U_' + str(k), self.nu, 1)
            self.w += [Uk]
            self.lbw += [*self.lower_control]   
            self.ubw += [*self.upper_control]
            self.w0 += [*self.initial_guess_control]

            # state at the collocation points
            Xc = []
            for j in range(degree):
                Xkj = SX.sym('X_' + str(k) + '_' + str(j), self.nx, 1)
                Xc += [Xkj]
                self.w += [Xkj]
                self.lbw += [*self.lower_pose, *self.lower_att, *self.lower_vel, *self.lower_rate]
                self.ubw += [*self.upper_pose, *self.upper_att, *self.upper_vel, *self.upper_rate]
                self.w0 += [*self.initial_guess_state]

            #loop over the collocation points
            Xk_end = self.D[0] * Xk
            for j in range(1, degree+1):
                # expression for the state derivative at the collocation point
                xp = self.C[0, j] * Xk
                for r in range(degree):
                    xp = xp + self.C[r+1, j] * Xc[r]

                # append collocation equations
                # f(Xk, Uk) is the state derivative at the collocation point
                fj, qj = f(Xc[j-1], Uk)
                self.g += [h * fj - xp]
                self.lbg += [0] * self.nx
                self.ubg += [0] * self.nx

                # add contribution to the end state
                # D[j] is the coefficient of the continuity equation
                Xk_end = Xk_end + self.D[j] * Xc[j-1]

                # add contribution to quadrature function
                # B[j] is the coefficient of the quadrature function
                self.J += h * self.B[j] * qj    

            # new NLP variable for state at end of interval
            Xk = SX.sym('X_' + str(k+1), self.nx, 1)
            self.w += [Xk]
            self.lbw += [*self.lower_pose, *self.lower_att, *self.upper_vel, *self.upper_rate]
            self.ubw += [*self.upper_pose, *self.upper_att, *self.upper_vel, *self.upper_rate]
            self.w0 += [*self.initial_guess_state]

            # add equality constraint
            self.g += [Xk_end - Xk]
            self.lbg += [0] * self.nx
            self.ubg += [0] * self.nx
        
        # add the final state constraint
        self.g += [Xk - self.x_setpoint]
        self.lbg += [0] * self.nx
        self.ubg += [0] * self.nx

        #Initialise the optimal soln vector w_opt
        self.w_opt = np.zeros(len(self.w0))

        # create an NLP solver
        prob = {'f': self.J, 'x': vertcat(*self.w), 'g': vertcat(*self.g), 'p': vertcat(self.x0_hat, self.x_setpoint)}
        self.solver = nlpsol('solver','ipopt',prob,self.nlpopts_dc)    

   
    def solve_for_next_state(self):
        # solve the nlp for the current state and the setpoint
        sol = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=vertcat(self.X_opt_current, self.x_setpoint))
        self.w_opt = sol['x'].full().flatten()
        
    def extract_next_state(self, step, extract_preds=False):
        # extract the next state from the optimal solution. If the method is dms, then get the value X_k+1
        # if the method is dc, then get the value X_k+1. The extract_preds is simply to extract the predictions, which is flagged to improve calculation speed
        if self.method == "DC":
            self.X_opt_current = self.w_opt[ self.nx + self.nu + self.nx * self.degree: self.nx + self.nu + self.nx * self.degree + self.nx]
            if extract_preds:
                self._extract_predictions()
        elif self.method == "DMS":
            self.X_opt_current = self.w_opt[self.nx + self.nu : self.nx + self.nu + self.nx]
            if extract_preds:
                self._extract_predictions() 
        else:
            raise ValueError("The method is not defined")
        
        self.U_opt_current = self.w_opt[self.nx : self.nx + self.nu]
        self.X_mpc[step,:] = self.X_opt_current
        self.U_mpc[step,:] = self.U_opt_current # while this variable is called current, it is the implemented control that has given the current state
        self._deviation()

    def _extract_predictions(self):
        if self.method == "DC":
            # Extract the state and control predicitions over the horizon length N, from the control interval 2 to N
            self.X_pred_current = np.vstack([self.w_opt[(self.nx + self.nu + self.nx * self.degree)*i: (self.nx + self.nu + self.nx * self.degree)*i + self.nx] for i in range(2,self.N)])
            self.U_pred_current = np.vstack([self.w_opt[((self.nx + self.nu + self.degree*self.nx)*(i-1)+self.nx): ((self.nx + self.nu + self.degree*self.nx)*(i-1)+self.nx) + self.nu ] for i in range(2,self.N)])
        elif self.method == "DMS":
            # Extract the state predicitions over the horizon length, from the control interval 2 to N
            self.X_pred_current = np.vstack([self.w_opt[(self.nx+self.nu)*(i): (self.nx+self.nu)*(i) + self.nx] for i in range(2,self.N)])
            self.U_pred_current = np.vstack([self.w_opt[(self.nx+self.nu)*(i)+self.nx: (self.nx+self.nu)*(i)+self.nx + self.nu] for i in range(self.N)])



    def _deviation(self):
        # calculate the deviation between the current state and the setpoint
        self.total_dev_current = np.linalg.norm(self.X_opt_current - self.x_setpoint)
        self.total_dev.append(self.total_dev_current)

        self.pose_dev_current = np.linalg.norm(self.X_opt_current[:3] - self.x_setpoint[:3])
        self.pose_dev.append(self.pose_dev_current)

    def plot_drone_trajectory_end(self):
        # plot the drone trajectory in the end
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.X_mpc[:,0], self.X_mpc[:,1], self.X_mpc[:,2], label='Drone trajectory')
        ax.scatter(self.X_mpc[0,0], self.X_mpc[0,1], self.X_mpc[0,2], color='r', label='Start')
        ax.scatter(self.X_mpc[-1,0], self.X_mpc[-1,1], self.X_mpc[-1,2], color='g', label='End')
        ax.scatter(self.x_setpoint[0], self.x_setpoint[1], self.x_setpoint[2], color='k', label='Setpoint')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

   

    def control_input_to_drone(self):
        # the control input to the drone, at the moment are the attitude setpoints [ roll, pitch, yawrate, thrust]
        # we return these from the obtained solution of the solver and convert them to the correct units. this input state 
        # will then be used by the onboard PID controller as setpoints to bring the drone state to

        roll = self.X_opt_current[3] * 180/np.pi # convert to degrees from rad
        pitch = self.X_opt_current[4] * 180/np.pi # convert to degrees from rad
        yawrate = self.X_opt_current[11] * 180/np.pi # convert to degrees/second from rad/second
        # the thrust vallue to be sent to the drone is an integer value between 10001 for 0% thrust and 60000 for 100% thrust
        # the solver gives the value of thrust between 0 to 0.6292, so we need to convert this to the correct value
        thrust = (self.U_opt_current[0] / 0.6292) * 50000 + 10001

        return [roll, pitch, yawrate, thrust]
            
