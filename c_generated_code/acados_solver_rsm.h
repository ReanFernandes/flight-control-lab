/*
 * Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
 * Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
 * Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
 * Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SOLVER_rsm_H_
#define ACADOS_SOLVER_rsm_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define RSM_NX     2
#define RSM_NZ     2
#define RSM_NU     2
#define RSM_NP     3
#define RSM_NBX    0
#define RSM_NBX0   2
#define RSM_NBU    1
#define RSM_NSBX   0
#define RSM_NSBU   0
#define RSM_NSH    0
#define RSM_NSG    0
#define RSM_NSPHI  0
#define RSM_NSHN   0
#define RSM_NSGN   0
#define RSM_NSPHIN 0
#define RSM_NSBXN  0
#define RSM_NS     0
#define RSM_NSN    0
#define RSM_NG     2
#define RSM_NBXN   0
#define RSM_NGN    0
#define RSM_NY0    4
#define RSM_NY     4
#define RSM_NYN    2
#define RSM_N      2
#define RSM_NH     0
#define RSM_NPHI   1
#define RSM_NHN    0
#define RSM_NPHIN  0
#define RSM_NR     2

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct rsm_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */
    // dynamics

    external_function_param_casadi *impl_dae_fun;
    external_function_param_casadi *impl_dae_fun_jac_x_xdot_z;
    external_function_param_casadi *impl_dae_jac_x_xdot_u_z;




    // cost






    // constraints
    external_function_param_casadi *phi_constraint;





} rsm_solver_capsule;

ACADOS_SYMBOL_EXPORT rsm_solver_capsule * rsm_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int rsm_acados_free_capsule(rsm_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int rsm_acados_create(rsm_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int rsm_acados_reset(rsm_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of rsm_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int rsm_acados_create_with_discretization(rsm_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int rsm_acados_update_time_steps(rsm_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int rsm_acados_update_qp_solver_cond_N(rsm_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int rsm_acados_update_params(rsm_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int rsm_acados_update_params_sparse(rsm_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);

ACADOS_SYMBOL_EXPORT int rsm_acados_solve(rsm_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int rsm_acados_free(rsm_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void rsm_acados_print_stats(rsm_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int rsm_acados_custom_update(rsm_solver_capsule* capsule, double* data, int data_len);


ACADOS_SYMBOL_EXPORT ocp_nlp_in *rsm_acados_get_nlp_in(rsm_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *rsm_acados_get_nlp_out(rsm_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *rsm_acados_get_sens_out(rsm_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *rsm_acados_get_nlp_solver(rsm_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *rsm_acados_get_nlp_config(rsm_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *rsm_acados_get_nlp_opts(rsm_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *rsm_acados_get_nlp_dims(rsm_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *rsm_acados_get_nlp_plan(rsm_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_rsm_H_