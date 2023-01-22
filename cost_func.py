from casadi import * 
import numpy as np
from dynamics_model import *

def cost(Q, R, function_type):
    # get the model
    _, x, u, nx, _ = eval( "model_" + function_type + "()")

    # create setpoint variable
    x_ref = SX.sym('x_ref', nx,1)

    # define cost function

    L = 0.5 * mtimes( mtimes( (x - x_ref).T, Q), (x - x_ref)) + 0.5 * mtimes( mtimes( u.T, R), u)

    return L, x_ref

