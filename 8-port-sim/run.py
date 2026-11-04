import v0
import plot
import numpy as np

## Global Experiment Constants
SEED = 0
N_PORTS = 8
MAX_N_TRIALS = 50
T=20  # The prior belief that min_trials is <= some maximum number of trials. This is the prior belief that the agent has about the environment, and is used to initialize the belief states.
TRIAL_TIME_MAX = 4  # Based on experiment setup
RADIUS = 1  # Arbitrary radius for the circular arrangement of ports

params_env = {
    'min_trials': 18-1, # Minimum number of trials for the experiment. Trials are 0-indexed.
    'trial_time_max': TRIAL_TIME_MAX,
    'max_trials': MAX_N_TRIALS,
    'num_ports': N_PORTS,
    'poke_interval': 0.2,
    'alpha': 0.5,
    'radius': RADIUS
}
params_agent = {
    'rho': 0,
    'sigma': 0,
    'q1': 0.98,
    'q2': 0.98,
    'confidence_threshold': 0.3,
    'trial_time_max': 3.5,
    'beta1': 10,
    'beta2': 10
}

params_beliefs = {
    'n_ports': params_env['num_ports'],
    'T': T,
    'f_theta': v0.von_mises(N_PORTS, 1),
    'g_phi': np.array([1/T for _ in range(T)])
}
sim = v0.simulation(params_env=params_env, params_agent=params_agent,params_beliefs=params_beliefs, seed=SEED)

results = sim.run()
save_as = "session.npy"
np.save(save_as, results)
print(results)

data = plot.load_pokes(save_as)
plot.check(data, params_env['min_trials'])

fig = plot.plot_session(np.array(results), params_env['min_trials'], color_by='distance', phase='no_water')

plot.plt.show()