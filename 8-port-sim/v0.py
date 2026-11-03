## This is a first draft of the 8-port behavioural model. 
## Potential updates include updating the model to include fitting methods etc.
''' We assume that the port is already aligned to the correct port, i.e. theta=0 is the correct port. This is the convention held for the rest of the script
# Numbering of the ports will follow clockwise from the correct port, with theta=n*pi/4 '''

import numpy as np

## Global Experiment Constants
N_PORTS = 8
MAX_N_TRIALS = 20
TRIAL_TIME_MAX = 4  # Based on experiment setup
RADIUS = 1  # Arbitrary radius for the circular arrangement of ports

### Helper functions

def sigmoid(x):
    """Computes the sigmoid function."""
    return np.tanh(x/2)*0.5 + 0.5

def von_mises(num_port, kappa):
    """Computes the von Mises probability density function."""
    a = kappa * np.cos(2 * np.pi * np.arange(num_port) / num_port)
    f = np.exp(a - a.max())
    return np.array(f / f.sum())

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
        
    def distance(self, target_port, prev_port):
        if prev_port is None:
            target_loc = (self.radius * np.cos(target_port *2* np.pi / self.num_ports), self.radius * np.sin(target_port * np.pi * 2 / self.num_ports))
            prev_port = self._generate_start_position()
            return np.linalg.norm(np.array(target_loc-prev_port))
        """Computes the distance between two ports in the 8-port model."""
        return min(abs(prev_port - target_port), self.num_ports - abs(prev_port - target_port))*(2 * np.pi / self.num_ports)*self.radius  # Each port is pi/4 radians apart

    def _generate_start_position(self):
        """Generates random starting positions for the mice."""
        r, phi = self.radius * np.sqrt(np.random.random()), 2 * np.pi * np.random.random()
        return np.array([r * np.cos(phi), r * np.sin(phi)])

        # Based on the
        # Placeholder for the actual l(t) implementation
        # This should be replaced with the actual logic for the 8-port model

    def timestamp(self, port, previous_port, t_in_trial):
        """Returns the timestamp of the poke and the total time spent in the trial.
        Args:
            port: The port that was poked.
            previous_port: The previous port that was poked.
            current_trial: The current trial number.
            t_sum: The total time spent in the current trial.

        Returns:
            The updated total time spent in the trial.
        t: The timestamp of the poke."""

        t_in_trial += self.distance(port, previous_port) * self._alpha + self.poke_interval  # Add the time taken to move to the port and the poke interval
        return t_in_trial

    def l(self, stop, current_trial, t_in_trial):
        """Maps the timestap t to its corresponding trial number. Currently uses a
        dual mechanism , one based on confidence of correct port (notes eq.57), and 
        the other on total time spend on a trial (notes eq.47)
        
        Args:
            stop: The stop condition
            current_trial: The current trial number
            t_in_trial: The time spent in the current trial."""

        new_t_in_trial = t_in_trial
        new_current_trial = current_trial

        if t_in_trial >= self.trial_time_max or stop:
            new_current_trial += 1  # Increment the trial number
            new_t_in_trial = 0  # Reset the time spent in the trial
  
        return new_current_trial, new_t_in_trial


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

    def marginal(self, belief_states):
        """Computes the marginal distribution over ports given the belief states.
        Args:
            belief_states: The current belief states.
            current_trial: The current trial number.
        Returns:
            The marginal distribution over ports."""
        return belief_states.sum(axis=1)

    def terminal(self, belief_states, current_trial):
        """Computes the terminal distribution over ports given the belief states.
        Args:
            belief_states: The current belief states.
            current_trial: The current trial number.
        Returns:
            The terminal distribution over ports."""
        mask = np.arange(belief_states.shape[1]) <= current_trial 
        return (belief_states * mask).sum(axis=1)

    def policy(self, beliefs, current_trial):
        # Placeholder for the actual policy implementation
        # This should be replaced with the actual logic for the 8-port model
        port = np.argmax(self.terminal(beliefs, current_trial))  # Choose the port with the highest marginal probability
        
        # Get the confidence of the chosen port
        confidence = self.marginal(beliefs)[port]
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
    
    def __init__(self, f_theta, g_phi, rho=0, sigma=0):
        self.f_theta = f_theta #probability dist as a vector
        self.g_phi = g_phi #probability dist as a vector
        self.state = self._initialize(self.f_theta, self.g_phi)

    @property
    def f_theta(self):
        return self._f_theta

    @f_theta.setter
    def f_theta(self, distribution):
        """Sets the f_theta distribution and checks if it is valid."""
        if distribution.shape != (N_PORTS,):
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
        if distribution.shape != (MAX_N_TRIALS,):
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
    def __init__(self, params_env, params_agent, f_theta, g_phi):
        self.environment = environment(params_env)
        self.agent = agent(params_agent)
        self.belief_states = belief_states(f_theta, g_phi, self.agent.rho, self.agent.sigma)
        self.sim_data = [] # Outcome of the mice. 

    def run(self):
        self.belief_states = belief_states(self.belief_states.f_theta, self.belief_states.g_phi, self.agent.rho, self.agent.sigma)  # Reset belief states for each run
        self.sim_data = []  # Reset simulation data for each run, follows the form of [(port, trial, t_sum), ...]
        current_trial = 0
        t_in_trial = 0  # Initialize time spent in the current trial
        w_t = False  # Initialize water received flag
        # max_pokes = (self.environment.trial_time_max//self.environment.poke_interval +1)*self.environment.max_trials  # Arbitrary limit on the number of pokes to prevent infinite loops

        """Run the simulation. Currently returns a list of tuples (poke, trial) """
        while not w_t :

            # Decide where to poke, port_to_poke = $\arg\max_{a_t} \pi(a_t|b_{t-1})$,  confidence = $\pi(port_to_poke |b_{t-1})$
            port_to_poke, confidence = self.agent.policy(self.belief_states.get_state(), current_trial) #Port and confidence for p_t given current state of beliefs b_{t-1}

            # mechanism 2 checks if the time to poke will exceed agent's t_max. We assume that the agent's t_max is below the environment's trial_time_max. If the time to poke exceeds this, we increment the trial and reset the time in trial.
            t_sum = self.environment.timestamp(port_to_poke, self.sim_data[-1][0] if self.sim_data and t_in_trial > 0 else None , t_in_trial) 

            mech2_stop_prob = self.agent._soft_time_gate(t_sum)  # Get the probability of stopping based on the time gate (soft mechanism 2)
            
            current_trial, t_in_trial = self.environment.l(np.random.rand() < mech2_stop_prob, current_trial, t_sum)
            
            if current_trial >= self.environment.max_trials:  # Stop the simulation if water is received, max trials reached, or trial ended
                break
            
            if t_in_trial == 0:  # If the trial has been incremented, reset the time in trial
                continue  # Skip the rest of the loop and start the next trial/poke, does not update beliefs

            mech1_stop_prob = self.agent._soft_confidence_gate(confidence)  # Get the probability of stopping based on the confidence gate (soft mechanism 1)
            w_t = self.environment.respondToPoke(port_to_poke, current_trial)
            if w_t:
                break

            self.belief_states.update(self.agent.update(self.belief_states.get_state(), port_to_poke, current_trial)) # Update beliefs first
            self.sim_data.append((port_to_poke, current_trial, t_sum, confidence,w_t))  # Record the poke, trial, and whether water was received
            

            if np.random.rand() < mech1_stop_prob: # Stops if the confidence gate is activated, i.e. if the agent's confidence in the chosen port exceeds the threshold
                current_trial += 1
                t_in_trial = 0
                if current_trial >= self.environment.max_trials:  # Stop the simulation if water is received, max trials reached, or trial ended
                    break
                continue 

            self.belief_states.update(self.agent.update(self.belief_states.get_state(), port_to_poke, current_trial))  # Update the agent's beliefs based on the observed outcome
            
            
        print("Simulation completed. Total trials: ", current_trial)
        return self.sim_data

params_env = {
    'min_trials': 5,
    'trial_time_max': TRIAL_TIME_MAX,
    'max_trials': MAX_N_TRIALS,
    'num_ports': N_PORTS,
    'poke_interval': 0.2,
    'alpha': 0.5,
    'radius': RADIUS
}
params_agent = {
    'rho': 0.01,
    'sigma': 0.01,
    'q1': 0.98,
    'q2': 0.98,
    'confidence_threshold': 0.5,
    'trial_time_max': 3.5,
    'beta1': 10,
    'beta2': 10
}
sim = simulation(params_env=params_env, params_agent=params_agent, f_theta=von_mises(N_PORTS, 5), g_phi=np.array([1/MAX_N_TRIALS for _ in range(MAX_N_TRIALS)]))
results = sim.run()
print(results)