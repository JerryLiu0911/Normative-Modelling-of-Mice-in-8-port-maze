## This is a first draft of the 8-port behavioural model. 
## Potential updates include updating the model to include fitting methods etc.
''' We assume that the port is already aligned to the correct port, i.e. theta=0 is the correct port. This is the convention held for the rest of the script
# Numbering of the ports will follow clockwise from the correct port, with theta=n*pi/4 '''

import numpy as np
# ---- Recording format (structure of arrays; JAX-friendly) ----
KIND_POKE, KIND_STOP, KIND_CAP = 0, 1, 2
FIELDS = {"port": np.int32, "kind": np.int32, "trial": np.int32,
          "t": np.float32, "conf": np.float32, "water": np.bool_}

def pad_session(cols, max_steps):
    """Pad every column to length max_steps; 'valid' marks real records."""
    n = len(cols["kind"])
    out = {}
    for k, dt in FIELDS.items():
        fill = -1 if k in ("port", "kind", "trial") else 0
        arr = np.full(max_steps, fill, dtype=dt); arr[:n] = cols[k][:max_steps]
        out[k] = arr
    out["valid"] = np.arange(max_steps) < n
    return out

def save_session(path, cols):
    np.savez(path, **cols)

def load_session(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}

def pokes_only(cols):
    """Real pokes only (drops STOP / cap records and padding)."""
    m = cols["valid"] & (cols["kind"] == KIND_POKE)
    return {k: v[m] for k, v in cols.items()}

### Helper functions

def sigmoid(x):
    """Computes the sigmoid function."""
    return np.tanh(x/2)*0.5 + 0.5

def softmax(x, beta):
    """Computes the softmax function with temperature parameter beta."""
    e_x = np.exp(beta * (x - np.max(x)))
    return e_x / e_x.sum()

def von_mises(num_port, kappa):
    """Computes the von Mises probability density function."""
    a = kappa * np.cos(2 * np.pi * np.arange(num_port) / num_port)
    f = np.exp(a - a.max())
    return np.array(f / f.sum())

def random_argmax(values, rng, rtol=1e-9, atol=0.0):
    """Argmax with uniform random tie-breaking among (numerically) equal maxima."""
    ties = np.flatnonzero(np.isclose(values, values.max(), rtol=rtol, atol=atol))
    return rng.choice(ties)
class environment:
    """Environment class for the 8-port model. This simulates the environment for a specific recall session,
    up until the first water is received.
        Args:
            min_trials: The minimum number of trials until water is available.
            trial_time_max: The maximum time for each trial.
            max_trials: The maximum number of trials.
            poke_interval: The time it takes for the agent to poke a port.
            radius: The radius of the circular arrangement of ports.
        """
    def __init__(self,params):
        self._min_trials = params['min_trials'] # minimum number of trials until water is available
        self.max_trials = params['max_trials'] # Arbitrary maximum 
        self.trial_time_max = params['trial_time_max']  # Based on experiment setup
        self.num_ports = params['num_ports']
        self.poke_interval = params['poke_interval']
        self.radius = params['radius']
        self._alpha = params['alpha']  # Arbitrary scaling factor for distance to time conversion (1/velocity)
    
    def respondToPoke(self, port, current_trial):
        """Responds to a poke at a given port. Returns True if the poke is at the correct port, False otherwise.
        Args:
            port: The port that was poked.
            current_trial: The current trial number.
        Returns:
            True if the poke is at the correct port, False otherwise."""
        return current_trial >= self._min_trials and port == 0  # Assuming port 0 is the correct port
        
    def distance(self, target_port, prev_port, rng):
        if type(prev_port) is np.ndarray:  # If prev_port is a position vector, compute the distance in Euclidean space
            target_loc = (self.radius * np.cos(target_port *2* np.pi / self.num_ports), self.radius * np.sin(target_port * np.pi * 2 / self.num_ports))
            return np.linalg.norm(np.array(target_loc-prev_port))
        """Computes the distance between two ports in the 8-port model."""
        return min(abs(prev_port - target_port), self.num_ports - abs(prev_port - target_port))*(2 * np.pi / self.num_ports)*self.radius  # Each port is pi/4 radians apart

    def generate_start_position(self, rng):
        """Generates random starting positions for the mice."""
        r, phi = self.radius * np.sqrt(rng.random()), 2 * np.pi * rng.random()
        return np.array([r * np.cos(phi), r * np.sin(phi)])

        # Based on the
        # Placeholder for the actual l(t) implementation
        # This should be replaced with the actual logic for the 8-port model

    def timestamp(self, port, previous_port, t_in_trial, rng):
        """Returns the timestamp of the poke and the total time spent in the trial.
        This is a placeholder until we have the actual timestamps from the data.
        Args:
            port: The port that was poked.
            previous_port: The previous port that was poked.
            current_trial: The current trial number.
            t_sum: The total time spent in the current trial.

        Returns:
            The updated total time spent in the trial.
        t: The timestamp of the poke."""

        t_in_trial += self.distance(port, previous_port, rng) * self._alpha + self.poke_interval  # Add the time taken to move to the port and the poke interval
        return t_in_trial


class agent:
    """Stateless agent class for the 8-port model. This simulates the agent's behavior in the environment."""
    def __init__(self, params):
        self.rho = params['rho']
        self.sigma = params['sigma']

        self._q1 =  params['q1'] # Arbitrary value for q1
        self._q2 =  params['q2'] # Arbitrary value for q2
        self._confidence_threshold = params['confidence_threshold'] # Arbitrary threshold for confidence gate
        self._trial_time_max = params['trial_time_max'] # Arbitrary threshold for time gate
        self._beta1 = params['beta1'] # Strength of confidence_gate firing
        self._beta2 = params['beta2'] # Strngth of time_gate firing

        self._beta = params['beta']  # Arbitrary value for beta
        self._L1 = params['L1']  # Arbitrary value for L1
        self._L2 = params['L2']  # Arbitrary value for L2
        self._lambda = params['lambda']  # Arbitrary value for lambda
        self._alpha = params['alpha']  # Arbitrary value for alpha
        self._poke_interval = params['poke_interval']  # Arbitrary value for poke_interval

        self.env = params['env']  # Environment parameters, including number of ports and radius
        self.num_ports = self.env['num_ports']
        self.radius = self.env['radius']

    def marginal(self, belief_states):
        """Computes the marginal distribution over ports given the belief states.
        Args:
            belief_states: The current belief states.
            current_trial: The current trial number.
        Returns:
            The marginal distribution over ports."""
        return belief_states.sum(axis=1)

    def terminal(self, belief_states, current_trial):
        """Computes the unnormalised terminal distribution over ports given the belief states.
        Args:
            belief_states: The current belief states.
            current_trial: The current trial number.
        Returns:
            The terminal distribution over ports."""
        mask = np.arange(belief_states.shape[1]) <= current_trial 
        return (belief_states * mask).sum(axis=1)

    def distance(self, target_port, prev_port):
        """The agent's perception/prediction of the distance between two ports"""
        if type(prev_port) is np.ndarray:  # If prev_port is a position vector, compute the distance in Euclidean space
            target_loc = (self.radius * np.cos(target_port *2* np.pi / self.num_ports), self.radius * np.sin(target_port * np.pi * 2 / self.num_ports))
            return np.linalg.norm(np.array(target_loc-prev_port))
        """Computes the distance between two ports in the 8-port model."""
        return min(abs(prev_port - target_port), self.num_ports - abs(prev_port - target_port))*(2 * np.pi / self.num_ports)*self.radius  # Each port is pi/4 radians apart


    def policy(self, beliefs, last_conf, current_time, current_trial, current_position, rng):
        """Computes the policy for the agent given the belief states and the current trial number.
        Args:
            beliefs: The current belief states.
            current_trial: The current trial number.
            current_position: The current position of the agent.
            rng: The random number generator.
        Returns:
            The chosen port and its confidence."""

        normalized_terminal = self.terminal(beliefs, current_trial)
        normalized_terminal /= normalized_terminal.sum()  # Normalize the terminal distribution

        # Mechanism 1 inspired confidence cost :   
        conf_term = 0.0 if last_conf is None else self._L1 * sigmoid(self._beta1 * (last_conf - self._confidence_threshold))

        distance = np.array([self.distance(port, current_position) for port in range(self.num_ports)])  # Compute the distance to each port

        # Mechanism 2 inspired time cost:
        # Compute the time cost for each port
        eta = np.array([current_time + self._poke_interval + self._alpha*distance[port] for port in range(self.num_ports)])  # Compute the time cost for each port
        time_cost = eta - self._trial_time_max 

        V_n = normalized_terminal - self._lambda * distance - self._L2*sigmoid(self._beta2 * time_cost) - conf_term # Compute the value function for each port and the "no poke" option

        port = rng.choice(self.num_ports+1, p=softmax(np.append(V_n, 0), self._beta))  # Choose the port with softmax policy, V_n(port=8)=0=STOP
        
        # Get the confidence of the chosen port
        confidence = self.marginal(beliefs)[port] if port < self.num_ports else None
        
        return port,confidence
        
    def update(self, belief_states, port, current_trial):
        """Updates the belief states based on the observed outcome.
        Args:
            belief_states: The current belief states.
            port: The port that was poked.
            current_trial: The current trial number.
        Returns:
            The updated belief states."""

        A = self.rho + (1 - self.rho) * (np.arange(belief_states.shape[0]) == port)                    # (8,)
        B = self.sigma + (1 - self.sigma) * (np.arange(belief_states.shape[1]) <= current_trial)              # (K,)  same index convention as your mask
        belief_states = belief_states * (1 - np.outer(A, B))
        
        if belief_states.sum() != 0:
            belief_states /= belief_states.sum()

        return belief_states

    def _soft_confidence_gate(self, confidence):
        """Determines whether to stop based on the confidence of the chosen port.
        Args:
            confidence: The confidence of the chosen port.
        Returns:
            True if the agent should stop, False otherwise."""
        return self._q1*sigmoid(self._beta1 * (confidence - self._confidence_threshold))

    def _soft_time_gate(self, t_sum):
        """Determines whether to stop based on the total time spent in the trial.
        Args:
            t_sum: The total time spent in the trial.
        Returns:
            True if the agent should stop, False otherwise."""
        return self._q2*sigmoid(self._beta2 * (t_sum - self._trial_time_max))

    def stop_prob(self, port_confidence, t_sum):
        """Returns the probability of stopping P_n, Currently uses a
        dual mechanism , one based on confidence of correct port (notes eq.57), and 
        the other on total time spend on a trial (notes eq.47)
        
        Args:
            port_confidence: The confidence of the chosen port.
            t_sum: The total time spent in the trial.

        Returns:
            The probability of stopping."""
        
        return 1-(1-self._soft_confidence_gate(port_confidence))*(1-self._soft_time_gate(t_sum))


class belief_states:
    
    def __init__(self, params):
        self.n_ports = params['n_ports']
        self.T = params['T'] #The prior belief that min_trials is <= some maximum number of trials. This is the prior belief that the agent has about the environment, and is used to initialize the belief states.
        self.f_theta = params['f_theta'] #probability dist as a vector
        self.g_phi = params['g_phi'] #probability dist as a vector
        self.state = self._initialize(self.f_theta, self.g_phi)

    @property
    def f_theta(self):
        return self._f_theta

    @f_theta.setter
    def f_theta(self, distribution):
        """Sets the f_theta distribution and checks if it is valid."""
        if distribution.shape != (self.n_ports,):
            raise ValueError("f_theta must be a numpy array of size 1*8")
        if not np.allclose(distribution.sum(), 1):
            raise ValueError("f_theta must be a valid probability distribution")
        self._f_theta = distribution

    @property
    def g_phi(self):
        return self._g_phi

    @g_phi.setter
    def g_phi(self, distribution):
        """Sets the g_phi distribution and checks if it is valid."""
        if distribution.shape != (self.T,):
            raise ValueError("g_phi must be a numpy array of size 1*20, over the 20 trials")
        if not np.allclose(distribution.sum(), 1):
            raise ValueError("g_phi must be a valid probability distribution")
        self._g_phi = distribution

    def _initialize(self, f_theta, g_phi):
        return np.outer(f_theta, g_phi)

    def update(self, new_state):
        self.state = new_state

    def get_state(self):
        return self.state


class simulation:
    def __init__(self, params_env, params_agent, params_beliefs, seed=0):
        self.rng = np.random.default_rng(seed)
        self.params_env = params_env
        self.params_beliefs = params_beliefs
        self.params_agent = params_agent

    def run(self):

        self.environment = environment(self.params_env)
        self.agent = agent(self.params_agent)
        self.belief_states = belief_states(self.params_beliefs)  # Reset belief states for each run

                #Draw the min_water trial from the prior distribution g_phi, if it is not provided in the environment parameters. 
        if self.params_env.get('min_trials') is None:
            g_env = np.asarray(self.params_env['g_env'])
            self.environment._min_trials = int(self.rng.choice(len(g_env), p=g_env))
        
        self.min_trials = self.environment._min_trials
        
        rec = {k: [] for k in FIELDS} # Initialize the record dictionary to store simulation data
        def record(kind, port, trial, t, conf, water):
            rec["kind"].append(kind)
            rec["port"].append(-1 if port is None else int(port))
            rec["trial"].append(trial)
            rec["t"].append(t)
            rec["conf"].append(0.0 if conf is None else float(conf))   # 0.0 = not applicable; use 'kind'
            rec["water"].append(bool(water))

        STOP_port = self.agent.num_ports # port 8 is the "stop" option, which is not a real port, but is used to indicate that the agent has decided to stop poking and end the trial. 
        current_trial, t_in_trial, w_t = 0, 0.0, False
        trigger_zone = self.environment.generate_start_position(self.rng)
        last_port, last_conf = None, None
        
        def new_trial():
            nonlocal current_trial, t_in_trial, trigger_zone, last_port, last_conf
            current_trial += 1
            t_in_trial = 0.0
            trigger_zone = self.environment.generate_start_position(self.rng)
            last_port, last_conf = None, None

        """Run the simulation. Currently returns a list of tuples (poke, trial) """
        while not w_t :

            current_position = last_port if last_port is not None else trigger_zone   # xy start or last port index
            
            # Decide where to poke, port_to_poke = $\arg\max_{a_t} \pi(a_t|b_{t-1})$,  confidence = $\pi(port_to_poke |b_{t-1})$
            port_to_poke, confidence = self.agent.policy(self.belief_states.get_state(), last_conf, t_in_trial, current_trial, current_position, self.rng) #Port and confidence for p_t given current state of beliefs b_{t-1}
            # Consider the starting position as a parameter to the policy
        
            if port_to_poke==STOP_port:  # If the agent decides to stop, we increment the trial and reset the time in trial
                record(KIND_STOP, None, current_trial, t_in_trial, None, False)
                new_trial()
                if current_trial >= self.environment.max_trials:
                    break
                continue
           
            t_sum = self.environment.timestamp(port_to_poke, current_position , t_in_trial, self.rng) 

            if t_sum >= self.environment.trial_time_max:                   # hard 4 s cap (environment)
                record(KIND_CAP, port_to_poke, current_trial, t_sum, None, False)  # proposed port, not poked
                new_trial()
                if current_trial >= self.environment.max_trials:
                    break
                continue
 
            w_t = self.environment.respondToPoke(port_to_poke, current_trial)
            record(KIND_POKE, port_to_poke, current_trial, t_sum, confidence, w_t)
            if w_t:
                break

            self.belief_states.update(self.agent.update(self.belief_states.get_state(), port_to_poke, current_trial))
            last_port, last_conf, t_in_trial = port_to_poke, confidence, t_sum
 
        self.sim_data = {k: np.asarray(v, dtype=FIELDS[k]) for k, v in rec.items()}
        self.sim_data["valid"] = np.ones(len(rec["kind"]), dtype=bool)           
        print("Simulation completed. Total trials: ", current_trial)
        return self.sim_data