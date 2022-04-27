## testing for syntax for Tron 

# tron = ExaTron.createProblem(n, xl, xu, nele_hess, eval_f_cb, eval_g_cb, eval_h_cb;
# :tol => gtol, :matrix_type => :Dense, :max_minor => 200,
# :frtol => 1e-12)
# tron.x .= x
# it = 0
# avg_tron_minor = 0
# terminate = false

# while !terminate
# it += 1

# # Solve the branch problem.
# status = ExaTron.solveProblem(tron)
# x .= tron.x
# avg_tron_minor += tron.minor_iter