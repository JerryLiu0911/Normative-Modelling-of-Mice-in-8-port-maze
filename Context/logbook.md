# Project logbook — F-ya311-3, 8-port maze normative model


## 2026-10-06 (Tue)

Read the project sources:
- planning form;
- Morales et al. 2020 (task, Memory Index, CPP result);
- Ahmadian's notes "Modeling 8-port maze behavior" (POMDP, belief updates, policies, likelihood, poke-count model);
- Peixoto Moledo 2025 thesis (current protocol; von Mises GLM-HMM with four states; anti-NMDAR and anti-LGI1 results);


## 2026-10-07 (Wed)

Read the CUED "Part IIB projects: second notice" and "Part IIB guide" for 2026–27. 

Key dates:

| Item | Deadline |
|---|---|
| Hazard form | 4 pm Wed 14 Oct |
| Progress reviews | 6 Nov, 4 Dec, 19 Feb, 19 Mar |
| Michaelmas presentation | from Thu 19 Nov |
| TMR | 4 pm Thu 21 Jan 2027 |
| Final report | 4 pm Mon 31 May 2027 |


## 2026-10-08

Prepared the hazard assessment, met with Dr.Ahmadian to discuss meeting details and clarify hazards.

- Computational project; hazards limited to display-screen work, posture, repetitive strain, fatigue and portable equipment. Overall risk low.
- Drafted an email asking Yashar to sign.
- Considered a descriptive project title. Candidate: "Normative Modelling of Memory Interference in Mouse Spatial Behaviour". Keep the registered title on official forms.

## 2026-10-14

**First supervisor meeting** 
- Regular meeting times at 3pm on Wednesdays
- Discussed/clarified concepts of experimental design :
  - HMM-GLM (Descriptive model) : "Descriptive model" as a model that we choose to fit the data to, not assuming a certain behaviour. In general, we assume animal behaviour is based on a discrete set of states which is described by the HMM. We do not necessarily push for orthogonality between states (i.e. forcing each state to characterise an independent behaviour), however, based on our design of the regressors and the weights given to them via the GLM we still tend to be able to characterise distinctive behaviour of each state. However, this could be something we look into. For more info on how the HMM-GLM is used in descriptive behaviour models, read the Ashwood paper which uses a supposedly "simpler model"

  - POMDP (normative model). “Normative” means that, given the task structure, the agent’s goal (obtain water as quickly as possible) and its information, we derive how an agent should behave, i.e. an optimal policy, rather than fitting a description of behaviour. The model in the notes departs from full optimality in three deliberate ways:

    - Imperfect memory: the prior over the rewarded port is a finite-width distribution $f_\theta$ (e.g. von Mises, or delta + uniform) rather than a Dirac delta at the correct port. This is intentional: $\theta$ is the quantity we want to measure (memory precision, interference, drug effects).
    - Approximate policy: actions are chosen greedily, maximising the probability of water on the current poke, rather than by solving the Bellman equation, which would maximise the expected discounted reward over the whole remaining search. Greedy ignores the value of waiting for water to switch on and the value of information, so it may be suboptimal. The size of that gap is an open question.
    - Choice noise: ε-greedy (and optionally non-ideal observation beliefs $\rho, \sigma$) accounts for variability not explained by the belief.

    - Inference itself is exact Bayes given $f_\theta$. The model is therefore bounded-optimal: optimal inference with imperfect memory and an approximately optimal policy. The fitted departures from optimality ($\theta$, $\epsilon$) are the interpretable quantities of interest.
- Computational project; hazards limited to display-screen work, posture, repetitive strain, fatigue and portable equipment. Overall risk low.
- Drafted an email asking Yashar to sign.
- Considered a descriptive project title. Candidate: "Normative Modelling of Memory Interference in Mouse Spatial Behaviour". Keep the registered title on official forms.


## 2026-10-20 (Tue)

Started `8-port-sim/v0.py`, a first draft of the simulator. Scaffolding: `environment`, `agent` and `belief_states` classes, strutctured separation of roles with the agent class containing the belief_states, and the environment to store the sequence of pokes.


## 2026-10-22 (Thu)

**Second Supervisor Meeting**, clarified modelling design choices and set-up of the problem.
- look through Ashwood paper for more information on MDPs

- DQN is fitting the data, needs to recurrent iteration for model parameters, it's easier to just use the greedy policy


- 3 sorts of docs :
	- logbook, lot of detail, notes to my stuff so we remember what we've done, maybe include 
	- results, when things work put them there
	- TMR drafts, polished and more well-written work.

t_n = time of the n_th poke in the current trial. use eq.47, use initial position as uniform over trigger zones

first poke t_1 : delta_x (length of vector from trigger zone to first port, we can use straight lines for first) * alpha
second poke : delta_x_2 (from first port to second port could be arc or straight line. Be careful of angular distance)*alpha + const. t

2 stops : - t_max or reach correct port

limitations :
	- t_max = 4 following experimental procedures, buttt

You're contradicting yourself by assuming you don't poke the correct port after knowing it (since we assume non-ideal memory)
m_n(p_n)-P_conf

3-parts of the model :

1. Bayesian inference (epistemlogical) OPTIMAL : Ideal observer model (knows exactly how the environment works, not necessarily the hidden states) , in order to update this prior belief we update belief based on interaction on reward port/pokes. ASSUMING THAT THE MICE KNOWS THE RULES OF THE TASK, i.e. it knows that the reward is based on r and tau based on the rules of the experiment

2. Decision making : NOT TRUELY DICTATED BY OPTIMALITY esp. Third part
	- Where to poke
		- Greedy policy may not be the most optimal
		- POMDP is reducible to an MDP where the environment state is turned into our beliefs. You should take into account, if I take this action, how is my belief about the environment change, will this lead me to a higher subjective probability
		- Consider Bellman optimality, DQN
	- Whether should carry on poking or stop poking (relaxation of the initial condition where p=p*). 
		- Could be based on energy expenditure too, which is similar to what eq. 57 describes, by stop poking after believing I've poked the correct port.

- TODO :
  - Generate synthetic data (simulate the data first)
  	- model distance to first poke
  	- model t_max (i.e. time limit for the agent) and the stop mechanisms
  
  - Currently ideal observe model, one of our aims could be to relax that, i.e., it might not exactly know how the rules work with respect to r and tau (LOOK AT MODELLING RHO AND SIGMA IN THE NOTES FOR SMALL PROBABILITY OF GETTING REWARD FROM WRONG PORT etc...)
  - Share Overleaf, GitHub Repo
  
## 2026-10-27 to 2026-10-28

**Simulator design.** Discussed the structure of `v0.py` with a view to porting to JAX later.

Conclusions:
- **Pure functions.** Changing state should be passed in and out explicitly, not stored as mutable attributes. Needed for `jit`, `scan` and `vmap`.
- **Environment outside the agent.** The environment should not be an attribute of the agent:
  - the agent must not be able to read $r$ or $\tau$;
  - in fitting there is no environment, because observations come from data.
- **Agent state:** belief (last-visit vector $T$, sufficient for the ideal observer) plus current trial $l$, time in trial and previous port. $l(t)$ is needed by the policy as well as the update.
- **Start positions.**
  - The mouse knows its own position at tone onset, so the first-poke distance $\Delta x_1$ is legitimate agent information.
  - Store the start positions as a per-trial array in the environment, passed to the agent as an observation at trial start. In fitting they come from tracking data.
  - The thesis trigger zone is 26% of arena area (p. 45), not the 1/16 of Morales 2020.

**Bugs noted in `v0.py`.**
- Port spacing should be $\pi/4$ (now fixed in the comment).
- Von Mises normalisation uses $I_0$, which doesn't sum to 1 over eight ports.
- `policy()` samples from the flattened 8×K joint.
- The `g_phi` setter has a typo, the wrong shape check, and writes `_f_theta`.
- `_distance` is not circular.
- The time gate omits $\Delta t$.

**Decision.** I will own the class design and keep building on `v0.py`. An AI-generated structure file (`8-port-sim/skeleton.py`) was produced and is not being used.

**AI use.** Claude Code (Claude Opus 5.5): explained JAX constraints and design trade-offs; reviewed `v0.py`; generated a skeleton that I chose not to use; wrote this logbook. The class design in `v0.py` is my own.

## 2026-10-28 (Wed), later

**Simulator design, continued** (`v0.py`).
- **Ownership of the trial counter.** The environment alone advances ℓ and resets the time-in-trial, when the agent sends "stop" or the 4 s window closes. The agent reads ℓ from the observation, so the two can't drift. Test: agent ℓ = environment ℓ at every step (ideal observer).
- **Event order per poke:**
  1. Choose $p_n$ using $\ell_n$.
  2. Environment returns $w_n$ and $t_n$.
  3. Update the belief.
  4. Compute $P_n$ from $m_n$ (after the update) and $t_n$; draw stop.
  5. Environment sets $\ell_{n+1}$.

  The dependence is sequential, not cyclic.
- **Agent holds parameters only.** The belief and the output record are passed in and out, not stored as attributes.
- **Update.** Signature `update(belief, port, ell) -> belief`. Called only after a miss, because a hit ends the episode. Zero row `port` where τ ≤ ℓ, then renormalise over (r, τ). The belief is the posterior $P(r,\tau\mid\text{history})$; the mask is the likelihood $P(w=0\mid r,\tau,p,\ell)$.
- **Indexing.** Trials and τ are 0-indexed. Consequence: "unvisited" in the last-visit vector $T$ can no longer be 0 (use −1, or keep trials 1-indexed in the model). CLAUDE.md still says 0 = unvisited; to resolve.
- **Stopping.**
  - `stop_prob` returns $P_n$ (Eq. 49 with the smooth gates, Eqs 55, 57, i.e. Eq. 58), not a flag. The loop draws the flag, so the same function serves simulation and LL₂.
  - Keep $q_1, q_2$ just below 1, so LL₂ is finite when real mice poke past a satisfied condition (notes p. 14).
  - Soft gates are chosen over hard indicators for Gate B: the hard version gives zero gradients and zero Fisher information in $t_{\max}$ and κ.
- **Helpers.** Generic numerics (sigmoid, softmax) as module-level functions. The marginal $m$ is a separate function shared by the confidence gate and the max-belief policy. Greedy uses $\mathcal{T}$, not $m$.

**Questions for Yashar** (collected so far):

*Protocol and data*
1. **Pokes per trial:** the table, and how often zero-poke trials occur. Fig. 12 shows 1–4 pokes; is that a bound or just the range shown?
2. **Trigger zones:** geometry and positions. Thesis 26% of arena area (p. 45) vs Morales 1/16. Are tracked start positions (mouse position at cue onset) available for fitting?
3. **Movement between ports:** arc or chord?
4. **First trial:** can water ever be available from the first trial, i.e. can $g$ put mass on τ = first trial? This decides whether greedy faces all-zero hit probabilities at the start (required tests 3 and 4b).
5. **The cue:** how well do trained mice respect the 4 s window? Is there a drop in poke rate at cue offset? This bears on the assumption that the agent knows ℓ.
6. **Code and data access:** Gonzalo's model, Paula's `ssm` fork, the Barcelona DQN, the data.

*Model*

7. **Yesterday's port:** when does it enter $f_\theta$ (weight $w_{24}$)?
8. **Mechanism 2 (Option B):** confirm the environment imposes the hard 4 s window while the agent has its own fitted $t_{\max}\le 4$ s.
9. **The poke that crosses 4 s (Eq. 47):** stopping happens after the poke that exceeds $t_{\max}$, but a poke after the window lies in the ITI, so it isn't in the data. Gate on the current time or on the next poke's time?
10. **Censoring:** for trials ended by the window rather than the agent, drop the $\log P_N$ term in LL₂?
11. **Eq. 57 vs Eq. 46:** the soft confidence gate changes the condition from "poke was the argmax $p^\star$" to "$m_n(p_n) > p_{\text{conf}}$". Intended as a model change, or only as smoothing?
12. **Softmax over $\mathcal{T}$:** its scale shrinks with $S_g(\ell)$, so the effective choice noise changes over trials. Normalise $\mathcal{T}$ first? This affects κ–choice-noise identifiability.
13. **Identifiability of $q_1, q_2$ against β₁, β₂:** they overlap in effect. Fix β at 10 throughout?

*Admin*

14. **Progress review 1** (by Fri 6 Nov): is it booked?
15. **Planning form:** resubmit now that the GLM-HMM is out of scope?

**AI use.** Claude Code (Claude Opus 5.5): reviewed my `v0.py` design and the `update` function; explained the event order, belief vs likelihood, and soft vs hard stopping, with checks against notes Eqs 46–58; drafted `stop_prob` and its two gate helpers on my request (to be reviewed and adapted by me); compiled this question list from our discussion. Class design and `update` are my own.

## 2026-10-28 (Wed), evening: implementing `v0.py`

**Structure decided.** JAX convention: the simulation loop holds everything that changes during a session (belief, current trial, time in trial, previous port, record). The agent and environment hold fixed parameters only, and their methods return new values.

**Done.**
- **Agent.**
  - Stateless.
  - `marginal(belief)`.
  - `update(belief, port, ell)`: zero row `port` for τ ≤ ℓ (0-indexed), renormalise, guard against an all-zero belief.
  - `stop_prob` with soft gates (Eqs 55, 57, 58): β₁ = β₂ = 10, $p_{\text{conf}}=0.5$, $q_1=q_2=0.98$, agent $t_{\max}=3.8$ s.
- **Helpers.** Numerically stable sigmoid (tanh form); circular arc distance between ports.
- **Environment.** `respondToPoke(port, current_trial)` takes the trial instead of storing it. Poke interval Δt = 0.5 s is an environment parameter.
- **Loop.**
  - A `for` loop with a cap $n_{\text{trials}}\,(\lfloor 4/\Delta t\rfloor+1)$, derived from the window, in place of `while`.
  - The rewarded poke is recorded.
  - The stop probability is computed and drawn.
- **Prior.** The f and g setters moved to `belief_states`.

**Open problems in the code** (as of the last review).
- **Crashes.**
  - `run()` overwrites the `belief_states` object with an array.
  - `agent.act` doesn't exist (the method is `policy`).
  - `start_positions[current_trial]` fails from trial 8.
- **Trial bookkeeping.**
  - The trial is advanced in two places: `l` for the window, and the loop for the stop. Both can fire on one poke, skipping a trial. Fix: pass `stop` into `l`, so `l` is the only place that changes the trial.
  - `l` runs before the update, so the update and the record use ℓ + 1 when a trial ends.
- **The window-crossing poke.**
  - Water is checked before the poke's time is computed, so a poke landing after 4 s can still be rewarded.
  - Working fix: compute the poke's time first. If it is past the window, the poke falls in the ITI: no response, update or record, and the trial ends. This matches the data, since ITI pokes are excluded from MI. Pending Yashar's answer to question 9.
- **Stopping.**
  - `stop_prob` gets the confidence from before the update; Eq. 57 and test 6 need `marginal(updated_belief)[port]`.
  - It uses global `np.random`; pass an `rng`.
- **Record.** It lacks w, $P_n$, the stop flag and the policy's probability vector. There is no censoring break at `max_trials`, and no error if the cap is reached.
- **Policy.** `argmax` with no tie-breaking pokes port 0 (the correct port) under a uniform prior, inflating MI. Greedy over $\mathcal{T}$ and the probability vector are still to do (next discussion).
- **Coupling.** The g setter's length check uses `MAX_N_TRIALS`, but K (the support of τ) and the session length are different quantities.
- **Start positions.** Placeholder `np.arange(8)` used as if it were a port.

**Target loop order:**
1. Choose.
2. Compute the poke's time; if past the window, end the trial via `l`.
3. Respond (w).
4. Record; break on water.
5. Update with the current ℓ.
6. $m$ after the update, then $P_n$, then draw stop.
7. `l(stop, …)`: reset the previous port; break if the session is censored.

**Start positions, plan.**
- Work in 2D (arena-radius units) with `port_xy(p)`.
- First-poke distance = Euclidean distance from the start point to the port (straight line, CLAUDE.md item 3). Port to port = arc, or chord via `port_xy`; keep a flag until question 3 is answered.
- Store start positions as a fixed `(max_trials, 2)` array in the environment.
  - Simulation placeholder: uniform in the disk, $r=R\sqrt{u}$.
  - Data: tracked positions at cue onset.
- `timestamp` looks up the start position when there is no previous port; don't pass a point in the `previous_port` slot.
- Check whether the port radius equals the arena radius.

**Relation to the notes.**
- In the notes, ℓ(t) is only a label (p. 1), and trials end only through the §5 mechanisms (Eq. 49).
- The hard 4 s window as a second way to end a trial is our addition, from the protocol (Option B). To record in the report as a modelling choice.
- For fitting, Eq. 48 ($t_n=\tau_n-t_{\text{start}}$ from recorded timestamps) avoids α and Δt.

**AI use.** Claude Code (Claude Opus 5.5): reviewed successive versions of `v0.py` and listed bugs, as well as verify against formulations. At my request it did not write code. All code in `v0.py` is my own. An earlier `stop_prob` draft from the AI was adapted by me.

---

## Open items

- [ ] Hazard form: confirm submitted and signed (due 14 Oct).
- [ ] Book progress review 1 before Fri 6 Nov, with at least a week's notice.
- [ ] Questions for Yashar: full list of 15 in the 2026-10-28 (later) entry.
- [ ] Decide: policy when every port's hit probability is 0; time-gate timing; name for the softmax parameter; "unvisited" value for $T$ under 0-indexing (and update CLAUDE.md); window-crossing poke rule (working choice: drop it as ITI).
- [ ] `v0.py`: fix the open problems listed in the 2026-10-28 evening entry, then policy/act.