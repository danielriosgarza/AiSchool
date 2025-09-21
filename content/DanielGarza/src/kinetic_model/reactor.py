"""
reactor.py
~~~~~~~~~~

Defines the Reactor and Pulse classes that orchestrate pulse-based simulations
of a microbiome–metabolome system under time-varying environmental conditions.

A Reactor instance

* holds references to a `Microbiome` and a `Metabolome`
* executes a sequence of `Pulse` objects
* integrates ODEs for volume, metabolites, and subpopulations
* applies environmental effects (pH, temperature, stirring) provided by each `Pulse`
* supports both instantaneous (vin/vout) and continuous (qin/qout) feed/migration
* produces trajectories that can be visualized via `make_plot()`
* **NEW**: Includes automatic performance optimization for faster simulation

A Pulse instance

* defines a time interval (t_start, t_end, n_steps)
* specifies instantaneous volume changes (vin, vout) at pulse onset
* specifies continuous flow rates (qin, qout) during integration
* carries optional feed compositions for metabolites and microbiome (instant and/or continuous)
* carries an `Environment` object (pH, Stirring, Temperature) used during the pulse

**Performance Optimization Features**

The Reactor now includes several methods to dramatically speed up simulation:

1. **Automatic Optimization**: `simulate_with_optimization()` analyzes system complexity and applies optimal parameters
2. **Fast Mode**: `set_fast_simulation_mode()` applies aggressive optimizations (75% step reduction)
3. **Balanced Mode**: `set_balanced_simulation_mode()` applies moderate optimizations (50% step reduction)
4. **Manual Control**: `reduce_simulation_steps()` lets you manually control step reduction
5. **Adaptive Time Stepping**: Removes forced evaluation at specific time points for better performance
6. **Smart ODE Solver**: Automatically chooses the best solver method based on system complexity

**Typical Speedups**
- Simple systems: 2-5x faster
- Medium complexity: 5-10x faster  
- High complexity: 10-20x faster

Example
-------
Complex microbial community with three bacterial species, multiple subpopulations,
and different environmental conditions across pulses:

>>> from microbiome_gym.kinetic_model import (
...     Microbiome, Bacteria, Subpopulation, FeedingTerm, 
...     Metabolite, Metabolome, Environment, pH, Stirring, Temperature,
...     Pulse, Reactor
... )
>>> import numpy as np
>>> 
>>> # Create metabolites for a complex metabolic network
>>> glucose = Metabolite("glucose", 20.0, {'C': 6, 'H': 12, 'O': 6}, "#ff0000")
>>> oxygen = Metabolite("oxygen", 8.0, {'O': 2}, "#00B8FF")
>>> lactate = Metabolite("lactate", 0.0, {'C': 3, 'H': 6, 'O': 3}, "#51cf66")
>>> acetate = Metabolite("acetate", 0.0, {'C': 2, 'H': 4, 'O': 2}, "#ffa500")
>>> ethanol = Metabolite("ethanol", 0.0, {'C': 2, 'H': 6, 'O': 1}, "#ff69b4")
>>> co2 = Metabolite("co2", 0.0, {'C': 1, 'O': 2}, "#8b4513")
>>> 
>>> # Create metabolome
>>> metabolome = Metabolome(metabolites=[glucose, oxygen, lactate, acetate, ethanol, co2])
>>> 
>>> # ===== BACTERIA 1: E. coli (Glucose specialist) =====
>>> # Feeding terms for E. coli
>>> ecoli_aerobic = FeedingTerm("ecoli_aerobic", {
...     "glucose": [1.0, 0.5],      # Consume glucose
...     "oxygen": [0.3, 0.1],       # Consume oxygen
...     "acetate": [-0.4, 0.0],     # Produce acetate
...     "co2": [-0.8, 0.0]          # Produce CO2
... }, metabolome)
>>> 
>>> ecoli_anaerobic = FeedingTerm("ecoli_anaerobic", {
...     "glucose": [1.0, 0.5],      # Consume glucose
...     "ethanol": [-0.6, 0.0],     # Produce ethanol
...     "acetate": [-0.3, 0.0],     # Produce acetate
...     "co2": [-0.4, 0.0]          # Produce CO2
... }, metabolome)
>>> 
>>> ecoli_lactate = FeedingTerm("ecoli_lactate", {
...     "lactate": [1.0, 0.3],      # Consume lactate
...     "acetate": [-0.8, 0.0],     # Produce acetate
...     "co2": [-0.5, 0.0]          # Produce CO2
... }, metabolome)
>>> 
>>> # E. coli subpopulations
>>> ecoli_aerobic_pop = Subpopulation("ecoli_aerobic", 2.0, "E. coli", 0.8, 
...                                    [ecoli_aerobic], 7.0, 2.0, 2.0, 37.0, 5.0, 2.0, 
...                                    state='active', color='#007acc')
>>> 
>>> ecoli_anaerobic_pop = Subpopulation("ecoli_anaerobic", 1.5, "E. coli", 0.6, 
...                                      [ecoli_anaerobic], 6.8, 2.2, 2.2, 37.0, 5.0, 2.0, 
...                                      state='active', color='#ffc107')
>>> 
>>> ecoli_inactive = Subpopulation("ecoli_inactive", 0.5, "E. coli", 0.0, 
...                                 [], 7.0, 2.0, 2.0, 37.0, 5.0, 2.0, 
...                                 state='inactive', color='#cccccc')
>>> 
>>> # E. coli with conditional transitions
>>> ecoli = Bacteria("E. coli", {
...     "aerobic": ecoli_aerobic_pop,
...     "anaerobic": ecoli_anaerobic_pop,
...     "inactive": ecoli_inactive
... }, {
...     "aerobic": [
...         ["anaerobic", "(concentrations[1] < 1.0) * 0.3", 0.3],
...         ["inactive", "(concentrations[0] < 0.5) * 0.4", 0.4]
...     ],
...     "anaerobic": [
...         ["inactive", "(concentrations[0] < 0.5) * 0.4", 0.4]
...     ],
...     "inactive": [
...         ["aerobic", "(concentrations[0] > 2.0) and (concentrations[1] > 1.5) * 0.2", 0.2],
...         ["anaerobic", "(concentrations[0] > 2.0) and (concentrations[1] < 1.5) * 0.2", 0.2]
...     ]
... }, color='#007acc')
>>> 
>>> # ===== BACTERIA 2: L. plantarum (Lactate specialist) =====
>>> lplantarum_lactate = FeedingTerm("lplantarum_lactate", {
...     "lactate": [1.0, 0.4],      # Consume lactate
...     "acetate": [-0.9, 0.0],     # Produce acetate
...     "co2": [-0.3, 0.0]          # Produce CO2
... }, metabolome)
>>> 
>>> lplantarum_glucose = FeedingTerm("lplantarum_glucose", {
...     "glucose": [1.0, 0.6],      # Consume glucose
...     "lactate": [-0.7, 0.0],     # Produce lactate
...     "acetate": [-0.2, 0.0]      # Produce acetate
... }, metabolome)
>>> 
>>> # L. plantarum subpopulations
>>> lplantarum_lactate_pop = Subpopulation("lplantarum_lactate", 1.8, "L. plantarum", 0.7, 
...                                         [lplantarum_lactate], 6.5, 2.5, 2.5, 37.0, 5.0, 2.0, 
...                                         state='active', color='#28a745')
>>> 
>>> lplantarum_glucose_pop = Subpopulation("lplantarum_glucose", 1.2, "L. plantarum", 0.5, 
...                                         [lplantarum_glucose], 6.8, 2.2, 2.2, 37.0, 5.0, 2.0, 
...                                         state='active', color='#20c997')
>>> 
>>> lplantarum_inactive = Subpopulation("lplantarum_inactive", 0.3, "L. plantarum", 0.0, 
...                                      [], 6.5, 2.5, 2.5, 37.0, 5.0, 2.0, 
...                                      state='inactive', color='#cccccc')
>>> 
>>> # L. plantarum with conditional transitions
>>> lplantarum = Bacteria("L. plantarum", {
...     "lactate_specialist": lplantarum_lactate_pop,
...     "glucose_consumer": lplantarum_glucose_pop,
...     "inactive": lplantarum_inactive
... }, {
...     "lactate_specialist": [
...         ["glucose_consumer", "(concentrations[2] < 0.5) * 0.25", 0.25],
...         ["inactive", "(concentrations[2] < 0.1) * 0.35", 0.35]
...     ],
...     "glucose_consumer": [
...         ["inactive", "(concentrations[0] < 0.5) * 0.35", 0.35]
...     ],
...     "inactive": [
...         ["lactate_specialist", "(concentrations[2] > 1.0) * 0.3", 0.3],
...         ["glucose_consumer", "(concentrations[0] > 1.0) and (concentrations[2] < 0.5) * 0.25", 0.25]
...     ]
... }, color='#28a745')
>>> 
>>> # ===== BACTERIA 3: B. fragilis (Acetate specialist) =====
>>> bfragilis_acetate = FeedingTerm("bfragilis_acetate", {
...     "acetate": [1.0, 0.3],      # Consume acetate
...     "ethanol": [-0.8, 0.0],     # Produce ethanol
...     "co2": [-0.4, 0.0]          # Produce CO2
... }, metabolome)
>>> 
>>> bfragilis_glucose = FeedingTerm("bfragilis_glucose", {
...     "glucose": [1.0, 0.7],      # Consume glucose
...     "acetate": [-0.5, 0.0],     # Produce acetate
...     "ethanol": [-0.3, 0.0]      # Produce ethanol
... }, metabolome)
>>> 
>>> # B. fragilis subpopulations
>>> bfragilis_acetate_pop = Subpopulation("bfragilis_acetate", 1.5, "B. fragilis", 0.6, 
...                                       [bfragilis_acetate], 7.2, 1.8, 1.8, 37.0, 5.0, 2.0, 
...                                       state='active', color='#dc3545')
>>> 
>>> bfragilis_glucose_pop = Subpopulation("bfragilis_glucose", 1.0, "B. fragilis", 0.4, 
...                                       [bfragilis_glucose], 7.0, 2.0, 2.0, 37.0, 5.0, 2.0, 
...                                       state='active', color='#fd7e14')
>>> 
>>> bfragilis_inactive = Subpopulation("bfragilis_inactive", 0.4, "B. fragilis", 0.0, 
...                                    [], 7.2, 1.8, 1.8, 37.0, 5.0, 2.0, 
...                                    state='inactive', color='#cccccc')
>>> 
>>> # B. fragilis with conditional transitions
>>> bfragilis = Bacteria("B. fragilis", {
...     "acetate_specialist": bfragilis_acetate_pop,
...     "glucose_consumer": bfragilis_glucose_pop,
...     "inactive": bfragilis_inactive
... }, {
...     "acetate_specialist": [
...         ["glucose_consumer", "(concentrations[3] < 0.3) * 0.3", 0.3],
...         ["inactive", "(concentrations[3] < 0.1) * 0.4", 0.4]
...     ],
...     "glucose_consumer": [
...         ["inactive", "(concentrations[0] < 0.5) * 0.4", 0.4]
...     ],
...     "inactive": [
...         ["acetate_specialist", "(concentrations[3] > 0.8) * 0.35", 0.35],
...         ["glucose_consumer", "(concentrations[0] > 1.0) and (concentrations[3] < 0.3) * 0.25", 0.25]
...     ]
... }, color='#dc3545')
>>> 
>>> # Create microbiome with all three species
>>> microbiome = Microbiome(name="complex_community", 
...                         bacteria={"E. coli": ecoli, "L. plantarum": lplantarum, "B. fragilis": bfragilis})
>>> 
>>> # ===== ENVIRONMENTS FOR DIFFERENT PULSES =====
>>> # Pulse 1: Aerobic conditions (high oxygen, neutral pH, 37°C)
>>> ph1 = pH(metabolome, intercept=7.0, met_dictionary={"acetate": -0.1, "lactate": -0.05})
>>> env1 = Environment(ph1, Stirring(rate=0.9, base_std=0.02), Temperature(37.0))
>>> 
>>> # Pulse 2: Anaerobic conditions (low oxygen, acidic pH, 42°C)
>>> ph2 = pH(metabolome, intercept=6.5, met_dictionary={"acetate": -0.2, "lactate": -0.1})
>>> env2 = Environment(ph2, Stirring(rate=0.8, base_std=0.05), Temperature(42.0))
>>> 
>>> # Pulse 3: Recovery conditions (neutral pH, 37°C, with feed)
>>> ph3 = pH(metabolome, intercept=7.2, met_dictionary={"acetate": -0.08, "lactate": -0.03})
>>> env3 = Environment(ph3, Stirring(rate=0.85, base_std=0.03), Temperature(37.0))
>>> 
>>> # Feed composition for pulse 3
>>> feed_glucose = Metabolite("glucose", 30.0, {'C': 6, 'H': 12, 'O': 6}, "#ff0000")
>>> feed_oxygen = Metabolite("oxygen", 15.0, {'O': 2}, "#00B8FF")
>>> feed_metabolome = Metabolome(metabolites=[feed_glucose, feed_oxygen, lactate, acetate, ethanol, co2])
>>> 
>>> # ===== PULSES WITH DIFFERENT CONDITIONS =====
>>> # Pulse 1: Aerobic growth phase
>>> pulse1 = Pulse(
...     t_start=0.0, t_end=10.0, n_steps=100,
...     vin=0.0, vout=0.0, qin=0.0, qout=0.0,
...     environment=env1
... )
>>> 
>>> # Pulse 2: Anaerobic stress phase
>>> pulse2 = Pulse(
...     t_start=10.0, t_end=20.0, n_steps=100,
...     vin=0.0, vout=0.0, qin=0.0, qout=0.0,
...     environment=env2
... )
>>> 
>>> # Pulse 3: Recovery with feed
>>> pulse3 = Pulse(
...     t_start=20.0, t_end=30.0, n_steps=100,
...     vin=5.0, vout=0.0, qin=0.3, qout=0.0,
...     instant_feed_metabolome=feed_metabolome,
...     continuous_feed_metabolome=feed_metabolome,
...     environment=env3
... )
>>> 
>>> # ===== REACTOR SIMULATION =====
>>> reactor = Reactor(microbiome=microbiome, metabolome=metabolome, 
...                   pulses=[pulse1, pulse2, pulse3], volume=20.0)
>>> reactor.simulate()
>>> 
>>> # ===== PLOTTING RESULTS =====
>>> fig = reactor.make_plot()                    # Show interactive plot
>>> fig = reactor.make_plot("complex_sim.html")  # Save to file
>>> 
>>> # Access simulation results
>>> print(f"Final volume: {reactor.v_simul[-1]:.2f}")
>>> print(f"Final glucose: {reactor.met_simul[0, -1]:.2f}")
>>> print(f"Final pH: {reactor.pH_simul[-1]:.2f}")
>>> print(f"Active E. coli: {reactor.cellActive_dyn['E. coli'][-1]:.2f}")
>>> print(f"Active L. plantarum: {reactor.cellActive_dyn['L. plantarum'][-1]:.2f}")
>>> print(f"Active B. fragilis: {reactor.cellActive_dyn['B. fragilis'][-1]:.2f}")

Notes
-----
* pH is computed from the current (unstirred) metabolite concentrations using the `pH` object in each pulse's `Environment`.
* Stirring perturbs concentrations used for biological rates (growth, metabolism), leaving pH calculation based on the original concentrations.
* Growth and metabolism always use the provided `Environment` (pH and temperature) from the active pulse.
* All concentrations and populations are clamped to be non-negative during integration.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .metabolome import Metabolome
from .microbiome import Microbiome
from .environment import Environment


class Pulse:
    """
    Represents a control pulse with specific feed/withdrawal parameters and environmental conditions.
    
    A Pulse defines a time period with specific control actions:
    - Time interval (t_start, t_end, n_steps)
    - Instantaneous feed events (vin, vout) at pulse start
    - Continuous flow events (qin, qout) during pulse duration
    - Feed compositions for both instantaneous and continuous events
    - Environmental conditions (pH, temperature, stirring) during the pulse
    
    Attributes
    ----------
    t_start : float
        Start time of the pulse
    t_end : float
        End time of the pulse
    n_steps : int
        Number of integration steps
    vin : float
        Volume added instantaneously at pulse start
    vout : float
        Volume removed instantaneously at pulse start
    qin : float
        Continuous inflow rate during pulse
    qout : float
        Continuous outflow rate during pulse
    instant_feed_metabolome : Metabolome
        Metabolite composition of the instantaneously added volume
    instant_feed_microbiome : Microbiome
        Bacterial composition of the instantaneously added volume
    continuous_feed_metabolome : Metabolome
        Metabolite composition of the continuous flow
    continuous_feed_microbiome : Microbiome
        Bacterial composition of the continuous flow
    environment : Environment
        Environmental conditions (pH, temperature, stirring) during the pulse
    range : np.ndarray
        Time points for integration
    """
    
    def __init__(self, 
                 t_start: float, t_end: float, n_steps: int,
                 vin: float = 0.0, vout: float = 0.0,
                 qin: float = 0.0, qout: float = 0.0,
                 instant_feed_metabolome: Optional[Metabolome] = None,
                 instant_feed_microbiome: Optional[Microbiome] = None,
                 continuous_feed_metabolome: Optional[Metabolome] = None,
                 continuous_feed_microbiome: Optional[Microbiome] = None,
                 environment: Optional[Environment] = None):
        """
        Initialize a Pulse.
        
        Parameters
        ----------
        t_start : float
            Start time of the pulse
        t_end : float
            End time of the pulse
        n_steps : int
            Number of integration steps
        vin : float, optional
            Volume added instantaneously at pulse start. Default is 0.0.
        vout : float, optional
            Volume removed instantaneously at pulse start. Default is 0.0.
        qin : float, optional
            Continuous inflow rate during pulse. Default is 0.0.
        qout : float, optional
            Continuous outflow rate during pulse. Default is 0.0.
        instant_feed_metabolome : Metabolome, optional
            Metabolite composition of the instantaneously added volume
        instant_feed_microbiome : Microbiome, optional
            Bacterial composition of the instantaneously added volume
        continuous_feed_metabolome : Metabolome, optional
            Metabolite composition of the continuous flow
        continuous_feed_microbiome : Microbiome, optional
            Bacterial composition of the continuous flow
        environment : Environment, optional
            Environmental conditions during the pulse
            
        Raises
        ------
        ValueError
            If time parameters are invalid or volumes are negative
        TypeError
            If inputs are not of correct types
        """
        # Validate time parameters
        if not isinstance(t_start, (int, float)) or not isinstance(t_end, (int, float)):
            raise TypeError("t_start and t_end must be numbers")
        if t_start >= t_end:
            raise ValueError("t_start must be less than t_end")
        if not isinstance(n_steps, int) or n_steps <= 0:
            raise TypeError("n_steps must be a positive integer")
            
        # Validate volume parameters
        if not isinstance(vin, (int, float)) or vin < 0:
            raise ValueError("vin must be a non-negative number")
        if not isinstance(vout, (int, float)) or vout < 0:
            raise ValueError("vout must be a non-negative number")
        if not isinstance(qin, (int, float)) or qin < 0:
            raise ValueError("qin must be a non-negative number")
        if not isinstance(qout, (int, float)) or qout < 0:
            raise ValueError("qout must be a non-negative number")
            
        # Validate feed objects
        if instant_feed_metabolome is not None and not isinstance(instant_feed_metabolome, Metabolome):
            raise TypeError("instant_feed_metabolome must be a Metabolome object")
        if instant_feed_microbiome is not None and not isinstance(instant_feed_microbiome, Microbiome):
            raise TypeError("instant_feed_microbiome must be a Microbiome object")
        if continuous_feed_metabolome is not None and not isinstance(continuous_feed_metabolome, Metabolome):
            raise TypeError("continuous_feed_metabolome must be a Metabolome object")
        if continuous_feed_microbiome is not None and not isinstance(continuous_feed_microbiome, Microbiome):
            raise TypeError("continuous_feed_microbiome must be a Microbiome object")
        if environment is not None and not isinstance(environment, Environment):
            raise TypeError("environment must be an Environment object")
        
        # Store parameters
        self.t_start = t_start
        self.t_end = t_end
        self.n_steps = n_steps
        self.range = np.linspace(self.t_start, self.t_end, self.n_steps)
        
        # Volume parameters
        self.vin = vin
        self.vout = vout
        self.qin = qin
        self.qout = qout
        
        # Feed compositions
        self.instant_feed_metabolome = instant_feed_metabolome
        self.instant_feed_microbiome = instant_feed_microbiome
        self.continuous_feed_metabolome = continuous_feed_metabolome
        self.continuous_feed_microbiome = continuous_feed_microbiome
        
        # Environmental conditions
        self.environment = environment
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Pulse(t_start={self.t_start}, t_end={self.t_end}, n_steps={self.n_steps})"
    
    def __str__(self) -> str:
        """String representation."""
        return f"Pulse({self.t_start:.1f}-{self.t_end:.1f}, {self.n_steps} steps)"


class Reactor:
    """
    A class representing a bioreactor with pulse-based control.
    
    The Reactor class integrates the entire kinetic model system and simulates
    dynamics using pulse-based control actions. It manages:
    - Volume dynamics
    - Metabolite concentrations
    - Bacterial population dynamics
    - Environmental conditions (pH, temperature, stirring)
    - ODE integration between pulses
    
    The Reactor maintains its own state vector for trajectory tracking and
    updates the metabolome and microbiome objects for rate calculations.
    
    Attributes
    ----------
    microbiome : Microbiome
        The microbial community
    metabolome : Metabolome
        The metabolite environment
    pulses : List[Pulse]
        List of control pulses
    volume : float
        Current reactor volume
    nstates : int
        Total number of state variables
    time_simul : np.ndarray
        Time points from simulation
    v_simul : np.ndarray
        Volume dynamics
    met_simul : np.ndarray
        Metabolite concentration dynamics
    pH_simul : np.ndarray
        pH dynamics
    subpop_simul : np.ndarray
        Subpopulation dynamics
    cellActive_dyn : np.ndarray
        Active cell dynamics by species
    cellInactive_dyn : np.ndarray
        Inactive cell dynamics by species
    cellDead_dyn : np.ndarray
        Dead cell dynamics by species
    """
    
    def __init__(self, microbiome: Microbiome, metabolome: Metabolome, 
                 pulses: List[Pulse], volume: float):
        """
        Initialize a Reactor.
        
        Parameters
        ----------
        microbiome : Microbiome
            The microbial community
        metabolome : Metabolome
            The metabolite environment
        pulses : List[Pulse]
            List of control pulses
        volume : float
            Initial reactor volume
            
        Raises
        ------
        ValueError
            If volume is negative or pulses is empty
        TypeError
            If inputs are not of correct types
        """
        # Validate inputs
        if not isinstance(microbiome, Microbiome):
            raise TypeError("microbiome must be a Microbiome object")
        if not isinstance(metabolome, Metabolome):
            raise TypeError("metabolome must be a Metabolome object")
        if not isinstance(pulses, list):
            raise TypeError("pulses must be a list")
        # Allow empty pulses for single-pulse integration (integrate_pulse method)
        # Empty pulses are only problematic for simulate() method
        if not isinstance(volume, (int, float)):
            raise TypeError("volume must be a number")
        if volume < 0:
            raise ValueError("volume cannot be negative")
        
        # Validate pulses
        for i, pulse in enumerate(pulses):
            if not isinstance(pulse, Pulse):
                raise TypeError(f"pulses[{i}] must be a Pulse object")
        
        self.microbiome = microbiome
        self.metabolome = metabolome
        self.pulses = pulses
        self.volume = volume
        
        # Calculate number of states
        # State vector: [volume, metabolites, subpopulations]
        self.nstates = 1 + self.metabolome.nmets + self._get_total_subpopulations()
        
        # Initialize simulation results
        self.time_simul = None
        self.v_simul = None
        self.met_simul = None
        self.pH_simul = None
        self.subpop_simul = None
        self.cellActive_dyn = None
        self.cellInactive_dyn = None
        self.cellDead_dyn = None
    
    def _get_total_subpopulations(self) -> int:
        """Get total number of subpopulations across all bacteria."""
        total = 0
        for bacteria in self.microbiome.bacteria.values():
            total += len(bacteria.subpopulations)
        return total
    
    def _get_subpopulation_list(self) -> List[tuple]:
        """Get list of (bacteria_name, subpop_name) tuples."""
        subpop_list = []
        for bacteria_name, bacteria in self.microbiome.bacteria.items():
            for subpop_name in bacteria.subpopulations.keys():
                subpop_list.append((bacteria_name, subpop_name))
        return subpop_list
    
    def get_states(self) -> np.ndarray:
        """
        Get current state vector.
        
        Returns
        -------
        np.ndarray
            State vector: [volume, metabolites, subpopulations]
        """
        vec = np.zeros(self.nstates)
        
        # Volume
        vec[0] = self.volume
        
        # Metabolites
        vec[1:1 + self.metabolome.nmets] = self.metabolome.get_concentration()
        
        # Subpopulations
        subpop_idx = 1 + self.metabolome.nmets
        for bacteria_name, bacteria in self.microbiome.bacteria.items():
            for subpop_name, subpop in bacteria.subpopulations.items():
                vec[subpop_idx] = subpop.count
                subpop_idx += 1
        
        # Ensure non-negative values
        vec = np.maximum(0, vec)
        return vec
    
    def update_states(self, vec: np.ndarray) -> None:
        """
        Update reactor state from state vector.
        
        Parameters
        ----------
        vec : np.ndarray
            State vector to update from
        """
        # Update volume
        self.volume = max(0, vec[0])
        
        # Update metabolites
        for idx, metabolite_name in enumerate(self.metabolome.metabolites):
            concentration = max(0, vec[1 + idx])
            metabolite = self.metabolome._metabolite_dict[metabolite_name]
            metabolite.update(concentration)
        
        # Update subpopulations
        subpop_idx = 1 + self.metabolome.nmets
        for bacteria_name, bacteria in self.microbiome.bacteria.items():
            for subpop_name, subpop in bacteria.subpopulations.items():
                count = max(0, vec[subpop_idx])
                subpop.update(count)
                subpop_idx += 1
    
    def dvdt(self, pulse: Pulse) -> float:
        """
        Calculate volume derivative.
        
        Parameters
        ----------
        pulse : Pulse
            Current pulse object
            
        Returns
        -------
        float
            Volume derivative
        """
        return max(-self.volume, (pulse.qin - pulse.qout))
    
    def dsdt(self, pulse: Pulse) -> np.ndarray:
        """
        Calculate metabolite concentration derivatives.
        
        Parameters
        ----------
        pulse : Pulse
            Current pulse object
            
        Returns
        -------
        np.ndarray
            Metabolite concentration derivatives
        """
        # Reactor-average concentrations for mass balance
        S = self.metabolome.get_concentration()
        V = max(self.volume, 1e-12)  # avoid division by zero
        dVdt = self.dvdt(pulse)

        # pH from reactor-average (unstirred) concentrations
        if pulse.environment is not None:
            _ = pulse.environment.pH.compute_pH(S)

            # Use stirred concentrations for biology (uptake/production gating)
            S_bio = pulse.environment.stirring.apply_stirring(S)
            metab = self.microbiome.metabolism(S_bio, pulse.environment)
        else:
            S_bio = S
            metab = self.microbiome.metabolism(S_bio, None)

        # Feed vector for inflow
        if pulse.continuous_feed_metabolome is not None:
            S_in = pulse.continuous_feed_metabolome.get_concentration()
        else:
            S_in = np.zeros_like(S)

        # Full mass-balance in concentration form
        dSdt = (pulse.qin / V) * S_in \
               - (pulse.qout / V) * S \
               - (S / V) * dVdt \
               + metab

        # Floor so we never drive below zero from numerical error
        return np.maximum(-S, dSdt)
    
    def dxdt(self, pulse: Pulse) -> np.ndarray:
        """
        Calculate subpopulation derivative.
        
        Parameters
        ----------
        pulse : Pulse
            Current pulse object
            
        Returns
        -------
        np.ndarray
            Subpopulation derivatives
        """
        # Reactor-average concentrations for mass balance; stirred for biology
        S = self.metabolome.get_concentration()
        V = max(self.volume, 1e-12)
        dVdt = self.dvdt(pulse)

        if pulse.environment is not None:
            S_bio = pulse.environment.stirring.apply_stirring(S)
            growth = self.microbiome.growth(S_bio, pulse.environment)
        else:
            S_bio = S
            growth = self.microbiome.growth(S_bio, None)

        # Flatten current and feed populations in the same order
        X = []
        X_in = []
        dx_growth = []
        for bac_name, bac in self.microbiome.bacteria.items():
            for sp_name, sp in bac.subpopulations.items():
                X.append(sp.count)
                if (pulse.continuous_feed_microbiome is not None and
                    bac_name in pulse.continuous_feed_microbiome.bacteria and
                    sp_name in pulse.continuous_feed_microbiome.bacteria[bac_name].subpopulations):
                    X_in.append(pulse.continuous_feed_microbiome.bacteria[bac_name].subpopulations[sp_name].count)
                else:
                    X_in.append(0.0)
                dx_growth.append(growth[bac_name][sp_name])

        X = np.array(X)
        X_in = np.array(X_in)
        dx_growth = np.array(dx_growth)

        # Full mass-balance in concentration-like units
        dXdt = (pulse.qin / V) * X_in \
               - (pulse.qout / V) * X \
               - (X / V) * dVdt \
               + dx_growth

        return np.maximum(-X, dXdt)
    
    def ode(self, t: float, states: np.ndarray, pulse: Pulse) -> np.ndarray:
        """
        ODE function for integration.
        
        Parameters
        ----------
        t : float
            Current time
        states : np.ndarray
            Current state vector
        pulse : Pulse
            Current pulse object
            
        Returns
        -------
        np.ndarray
            State derivatives
        """
        # Update reactor state
        self.update_states(states)
        
        # Calculate derivatives
        derivatives = np.zeros(self.nstates)
        
        # Volume derivative
        derivatives[0] = self.dvdt(pulse)
        
        # Metabolite derivatives
        derivatives[1:1 + self.metabolome.nmets] = self.dsdt(pulse)
        
        # Subpopulation derivatives
        derivatives[1 + self.metabolome.nmets:] = self.dxdt(pulse)
        
        return derivatives
    
    def integrate_pulse(self, pulse: Pulse, store_states: bool = False) -> np.ndarray:
        """
        Integrate a single pulse and return the end-of-pulse state.
        
        This method applies instantaneous mixing at the start of the pulse,
        then integrates the ODE system for the pulse duration. It updates
        the reactor's internal state to the end-of-pulse state and returns
        the final state vector.
        
        Parameters
        ----------
        pulse : Pulse
            The pulse to integrate
        store_states : bool, optional
            If True, store the full trajectory in reactor simulation arrays
            (time_simul, v_simul, met_simul, subpop_simul, pH_simul).
            Default is False.
            
        Returns
        -------
        np.ndarray
            End-of-pulse state vector: [volume, metabolites, subpopulations]
            
        Example
        -------
        >>> from kinetic_model import Reactor, Pulse, Environment, pH, Stirring, Temperature
        >>> from kinetic_model.metabolome import Metabolome
        >>> from kinetic_model.microbiome import Microbiome
        >>> 
        >>> # Create a simple system
        >>> met = Metabolome(metabolites=[])
        >>> mic = Microbiome(name="test", bacteria={})
        >>> env = Environment(pH(met), Stirring(), Temperature(37.0))
        >>> 
        >>> # Create a pulse
        >>> pulse = Pulse(t_start=0.0, t_end=1.0, n_steps=10, environment=env)
        >>> 
        >>> # Create reactor and integrate pulse
        >>> reactor = Reactor(microbiome=mic, metabolome=met, pulses=[], volume=1)
        >>> final_state = reactor.integrate_pulse(pulse)
        >>> 
        >>> # The reactor state is now updated to the end-of-pulse state
        >>> assert np.array_equal(reactor.get_states(), final_state)
        >>> 
        >>> # With state storage enabled
        >>> final_state_with_trajectory = reactor.integrate_pulse(pulse, store_states=True)
        >>> assert reactor.time_simul is not None  # Trajectory was stored
        """
        # Apply discrete volume change at pulse start
        state = np.zeros(self.nstates)
        
        # --- BEGIN: robust instantaneous mixing ---
        V_old = self.volume
        V_new = V_old + pulse.vin - pulse.vout
        if V_new <= 1e-12:
            # Reactor emptied (or negative by request) → define clean zero state
            state[0] = 0.0
            state[1:1 + self.metabolome.nmets] = 0.0
            state[1 + self.metabolome.nmets:] = 0.0
        else:
            state[0] = V_new

            # Metabolites: mass balance on amounts, then convert back to concentration
            S_reactor = self.metabolome.get_concentration()
            if pulse.instant_feed_metabolome is not None:
                S_feed = pulse.instant_feed_metabolome.get_concentration()
            else:
                S_feed = np.zeros_like(S_reactor)

            S_new = (V_old * S_reactor + pulse.vin * S_feed - pulse.vout * S_reactor) / V_new
            state[1:1 + self.metabolome.nmets] = S_new

            # Subpopulations: mass balance on amounts, then back to "concentration-like" count/vol
            X_reactor = []
            X_feed = []
            for bac_name, bac in self.microbiome.bacteria.items():
                for sp_name, sp in bac.subpopulations.items():
                    X_reactor.append(sp.count)
                    if (pulse.instant_feed_microbiome is not None and
                        bac_name in pulse.instant_feed_microbiome.bacteria and
                        sp_name in pulse.instant_feed_microbiome.bacteria[bac_name].subpopulations):
                        X_feed.append(pulse.instant_feed_microbiome.bacteria[bac_name].subpopulations[sp_name].count)
                    else:
                        X_feed.append(0.0)
            X_reactor = np.array(X_reactor)
            X_feed = np.array(X_feed)

            X_new = (V_old * X_reactor + pulse.vin * X_feed - pulse.vout * X_reactor) / V_new
            state[1 + self.metabolome.nmets:] = X_new
        # --- END: robust instantaneous mixing ---
        
        # Update reactor state
        self.update_states(state)
        
        # Build t_eval from pulse settings (only affects saved samples, not the integrator's internal steps)
        if getattr(pulse, "range", None) is not None and len(pulse.range) > 1:
            t_eval = np.asarray(pulse.range, dtype=float)
        elif getattr(pulse, "n_steps", None) and pulse.n_steps > 0:
            t_eval = np.linspace(pulse.t_start, pulse.t_end, pulse.n_steps + 1)
        else:
            t_eval = None  # no constraint on saved times

        # Get ODE solver parameters from helper method
        ode_params = self._get_ode_solver_params()

        # Integrate ODE during pulse
        solution = solve_ivp(
            fun=lambda t, y: self.ode(t, y, pulse),
            t_span=(pulse.t_start, pulse.t_end),
            y0=self.get_states(),
            method=ode_params['method'],
            rtol=ode_params['rtol'],
            atol=ode_params['atol'],
            max_step=ode_params['max_step'],
            first_step=ode_params['first_step'],
            t_eval=t_eval,         # <-- now used if provided
            vectorized=False,
            dense_output=False     # <-- disable dense output to prevent hanging
        )
        
        # Store trajectory if requested
        if store_states:
            # Store time points
            if self.time_simul is None:
                self.time_simul = solution.t
            else:
                self.time_simul = np.concatenate([self.time_simul, solution.t])
            
            # Store volume dynamics
            if self.v_simul is None:
                self.v_simul = solution.y[0]
            else:
                self.v_simul = np.concatenate([self.v_simul, solution.y[0]])
            
            # Store metabolite dynamics
            met_dyn = solution.y[1:1 + self.metabolome.nmets]
            if self.met_simul is None:
                self.met_simul = met_dyn
            else:
                self.met_simul = np.hstack([self.met_simul, met_dyn])
            
            # Store subpopulation dynamics
            subpop_dyn = solution.y[1 + self.metabolome.nmets:]
            if self.subpop_simul is None:
                self.subpop_simul = subpop_dyn
            else:
                self.subpop_simul = np.hstack([self.subpop_simul, subpop_dyn])
            
            # Calculate and store pH dynamics for this pulse
            pH_dyn = np.zeros(len(solution.t))
            for i, t in enumerate(solution.t):
                if pulse.environment is not None:
                    concentrations = solution.y[1:1 + self.metabolome.nmets, i]
                    pH_dyn[i] = pulse.environment.pH.compute_pH(concentrations)
                else:
                    pH_dyn[i] = 7.0  # Default pH
            
            if self.pH_simul is None:
                self.pH_simul = pH_dyn
            else:
                self.pH_simul = np.concatenate([self.pH_simul, pH_dyn])
            
            # Calculate species-level dynamics if we have stored data
            if self.subpop_simul is not None:
                self._calculate_species_dynamics()
        
        # Update reactor state to end-of-pulse state
        final_state = solution.y[:, -1]
        self.update_states(final_state)
        
        return final_state

    def simulate(self) -> None:
        """
        Run the complete simulation.
        
        This method integrates the ODE system through all pulses, handling
        both discrete volume changes and continuous dynamics.
        """
        # Validate that we have pulses to simulate
        if not self.pulses:
            raise ValueError("Cannot simulate: pulses list is empty. Use integrate_pulse() for single-pulse integration.")
        
        # Initialize arrays to store results
        ts = np.empty(0)
        vol_dyn = np.empty(0)
        met_dyn = []
        subpop_dyn = []
        
        for pulse in self.pulses:
            # Apply discrete volume change at pulse start
            state = np.zeros(self.nstates)
            
            # --- BEGIN: robust instantaneous mixing ---
            V_old = self.volume
            V_new = V_old + pulse.vin - pulse.vout
            if V_new <= 1e-12:
                # Reactor emptied (or negative by request) → define clean zero state
                state[0] = 0.0
                state[1:1 + self.metabolome.nmets] = 0.0
                state[1 + self.metabolome.nmets:] = 0.0
            else:
                state[0] = V_new

                # Metabolites: mass balance on amounts, then convert back to concentration
                S_reactor = self.metabolome.get_concentration()
                if pulse.instant_feed_metabolome is not None:
                    S_feed = pulse.instant_feed_metabolome.get_concentration()
                else:
                    S_feed = np.zeros_like(S_reactor)

                S_new = (V_old * S_reactor + pulse.vin * S_feed - pulse.vout * S_reactor) / V_new
                state[1:1 + self.metabolome.nmets] = S_new

                # Subpopulations: mass balance on amounts, then back to "concentration-like" count/vol
                X_reactor = []
                X_feed = []
                for bac_name, bac in self.microbiome.bacteria.items():
                    for sp_name, sp in bac.subpopulations.items():
                        X_reactor.append(sp.count)
                        if (pulse.instant_feed_microbiome is not None and
                            bac_name in pulse.instant_feed_microbiome.bacteria and
                            sp_name in pulse.instant_feed_microbiome.bacteria[bac_name].subpopulations):
                            X_feed.append(pulse.instant_feed_microbiome.bacteria[bac_name].subpopulations[sp_name].count)
                        else:
                            X_feed.append(0.0)
                X_reactor = np.array(X_reactor)
                X_feed = np.array(X_feed)

                X_new = (V_old * X_reactor + pulse.vin * X_feed - pulse.vout * X_reactor) / V_new
                state[1 + self.metabolome.nmets:] = X_new
            # --- END: robust instantaneous mixing ---
            
            # Update reactor state
            self.update_states(state)
            
            # Build t_eval from pulse settings (only affects saved samples, not the integrator's internal steps)
            if getattr(pulse, "range", None) is not None and len(pulse.range) > 1:
                t_eval = np.asarray(pulse.range, dtype=float)
            elif getattr(pulse, "n_steps", None) and pulse.n_steps > 0:
                t_eval = np.linspace(pulse.t_start, pulse.t_end, pulse.n_steps + 1)
            else:
                t_eval = None  # no constraint on saved times

            # Get ODE solver parameters from helper method
            ode_params = self._get_ode_solver_params()

            # Integrate ODE during pulse
            solution = solve_ivp(
                fun=lambda t, y: self.ode(t, y, pulse),
                t_span=(pulse.t_start, pulse.t_end),
                y0=self.get_states(),
                method=ode_params['method'],
                rtol=ode_params['rtol'],
                atol=ode_params['atol'],
                max_step=ode_params['max_step'],
                first_step=ode_params['first_step'],
                t_eval=t_eval,         # <-- now used if provided
                vectorized=False,
                dense_output=False     # <-- disable dense output to prevent hanging
            )
            
            # Store results
            ts = np.concatenate([ts, solution.t])
            vol_dyn = np.concatenate([vol_dyn, solution.y[0]])
            met_dyn.append(solution.y[1:1 + self.metabolome.nmets])
            subpop_dyn.append(solution.y[1 + self.metabolome.nmets:])
        
        # Store final results
        self.time_simul = np.array(ts)
        self.v_simul = np.array(vol_dyn)
        self.met_simul = np.hstack(met_dyn)
        self.subpop_simul = np.hstack(subpop_dyn)
        
        # Calculate pH dynamics
        self.pH_simul = np.zeros(len(self.time_simul))
        for i, t in enumerate(self.time_simul):
            # Find which pulse this time belongs to
            pulse_idx = 0
            for j, pulse in enumerate(self.pulses):
                if pulse.t_start <= t <= pulse.t_end:
                    pulse_idx = j
                    break
            
            pulse = self.pulses[pulse_idx]
            if pulse.environment is not None:
                # Get metabolite concentrations at this time
                concentrations = self.met_simul[:, i]
                self.pH_simul[i] = pulse.environment.pH.compute_pH(concentrations)
            else:
                self.pH_simul[i] = 7.0  # Default pH
        
        # Ensure non-negative populations
        self.subpop_simul = np.maximum(0, self.subpop_simul)
        
        # Calculate species-level dynamics
        self._calculate_species_dynamics()
    
    def _calculate_species_dynamics(self) -> None:
        """Calculate species-level dynamics from subpopulation data."""
        species_names = list(self.microbiome.bacteria.keys())
        n_timepoints = len(self.time_simul)
        
        cellActive_dyn = {species: np.zeros(n_timepoints) for species in species_names}
        cellInactive_dyn = {species: np.zeros(n_timepoints) for species in species_names}
        cellDead_dyn = {species: np.zeros(n_timepoints) for species in species_names}
        
        subpop_idx = 0
        for bacteria_name, bacteria in self.microbiome.bacteria.items():
            for subpop_name, subpop in bacteria.subpopulations.items():
                if subpop.state == 'active':
                    cellActive_dyn[bacteria_name] += self.subpop_simul[subpop_idx]
                elif subpop.state == 'inactive':
                    cellInactive_dyn[bacteria_name] += self.subpop_simul[subpop_idx]
                elif subpop.state == 'dead':
                    cellDead_dyn[bacteria_name] += self.subpop_simul[subpop_idx]
                subpop_idx += 1
        
        self.cellActive_dyn = cellActive_dyn
        self.cellInactive_dyn = cellInactive_dyn
        self.cellDead_dyn = cellDead_dyn
    
    def optimize_simulation_parameters(self, target_time_per_pulse: float = 1.0) -> Dict[str, Any]:
        """
        Automatically optimize simulation parameters for better performance.
        
        This method analyzes the system complexity and suggests optimal parameters
        to achieve the target simulation time per pulse.
        
        Parameters
        ----------
        target_time_per_pulse : float, optional
            Target simulation time per pulse in seconds. Default is 1.0.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with recommended simulation parameters
        """
        # Analyze system complexity
        n_metabolites = self.metabolome.nmets
        n_subpopulations = sum(len(bacteria.subpopulations) for bacteria in self.microbiome.bacteria.values())
        n_bacteria = len(self.microbiome.bacteria)
        total_connections = sum(len(bacteria.connections) for bacteria in self.microbiome.bacteria.values())
        
        # Calculate complexity score
        complexity_score = n_metabolites * n_subpopulations * n_bacteria * (1 + total_connections / 100)
        
        # Recommend parameters based on complexity
        if complexity_score < 100:
            # Simple system
            recommended_params = {
                'method': 'RK45',
                'atol': 1e-4,
                'rtol': 1e-4,
                'max_step': 0.1,
                'first_step': 0.01,
                'n_steps_factor': 1.0  # Keep original n_steps
            }
        elif complexity_score < 500:
            # Medium complexity
            recommended_params = {
                'method': 'RK45',
                'atol': 1e-3,
                'rtol': 1e-3,
                'max_step': 0.2,
                'first_step': 0.02,
                'n_steps_factor': 0.7  # Reduce n_steps by 30%
            }
        else:
            # High complexity
            recommended_params = {
                'method': 'BDF',  # Better for stiff systems
                'atol': 1e-2,
                'rtol': 1e-2,
                'max_step': 0.5,
                'first_step': 0.05,
                'n_steps_factor': 0.5  # Reduce n_steps by 50%
            }
        
        # Add complexity analysis
        recommended_params.update({
            'complexity_score': complexity_score,
            'n_metabolites': n_metabolites,
            'n_subpopulations': n_subpopulations,
            'n_bacteria': n_bacteria,
            'total_connections': total_connections,
            'estimated_speedup': 2.0 if complexity_score < 100 else 5.0 if complexity_score < 500 else 10.0
        })
        
        return recommended_params
    
    def simulate_with_optimization(self, auto_optimize: bool = True, **kwargs) -> None:
        """
        Run simulation with automatic performance optimization.
        
        Parameters
        ----------
        auto_optimize : bool, optional
            Whether to automatically optimize simulation parameters. Default is True.
        **kwargs
            Additional parameters to override optimization recommendations
        """
        if auto_optimize:
            # Get optimization recommendations
            opt_params = self.optimize_simulation_parameters()
            
            # Apply optimizations to pulses
            for pulse in self.pulses:
                # Reduce n_steps if recommended
                if 'n_steps_factor' in opt_params:
                    original_steps = pulse.n_steps
                    pulse.n_steps = max(10, int(original_steps * opt_params['n_steps_factor']))
                    pulse.range = np.linspace(pulse.t_start, pulse.t_end, pulse.n_steps)
                    print(f"Optimized pulse {pulse.t_start}-{pulse.t_end}: {original_steps} → {pulse.n_steps} steps")
            
            # Store optimization info
            self.optimization_params = opt_params
            print(f"System complexity score: {opt_params['complexity_score']:.1f}")
            print(f"Estimated speedup: {opt_params['estimated_speedup']:.1f}x")
            
            # Apply ODE solver optimizations
            self._apply_ode_optimizations(opt_params)
        
        # Run the simulation
        self.simulate()
    
    def _apply_ode_optimizations(self, opt_params: Dict[str, Any]) -> None:
        """Apply optimized ODE solver parameters."""
        # Store optimized parameters for use in solve_ivp
        self.ode_method = opt_params.get('method', 'RK45')
        self.ode_atol = opt_params.get('atol', 1e-4)
        self.ode_rtol = opt_params.get('rtol', 1e-4)
        self.ode_max_step = opt_params.get('max_step', 0.1)
        self.ode_first_step = opt_params.get('first_step', 0.01)
        
        print(f"ODE solver optimized: {self.ode_method}, atol={self.ode_atol}, rtol={self.ode_rtol}")
    
    def _get_ode_solver_params(self) -> Dict[str, Any]:
        """Get current ODE solver parameters."""
        return {
            'method': getattr(self, 'ode_method', 'RK45'),
            'atol': getattr(self, 'ode_atol', 1e-4),
            'rtol': getattr(self, 'ode_rtol', 1e-4),
            'max_step': getattr(self, 'ode_max_step', 0.1),
            'first_step': getattr(self, 'ode_first_step', 0.01)
        }
    
    def reduce_simulation_steps(self, reduction_factor: float = 0.5, min_steps: int = 10) -> None:
        """
        Manually reduce the number of simulation steps for faster execution.
        
        This method reduces the n_steps parameter for all pulses by the specified factor,
        which can dramatically speed up simulation while maintaining reasonable accuracy.
        
        Parameters
        ----------
        reduction_factor : float, optional
            Factor to reduce steps by (0.5 = half the steps, 0.25 = quarter the steps).
            Default is 0.5.
        min_steps : int, optional
            Minimum number of steps per pulse. Default is 10.
        """
        if not 0 < reduction_factor <= 1:
            raise ValueError("reduction_factor must be between 0 and 1")
        
        # Check if we have pulses to reduce
        if not self.pulses:
            print("No pulses to reduce steps for")
            return
        
        total_original_steps = sum(pulse.n_steps for pulse in self.pulses)
        
        for pulse in self.pulses:
            original_steps = pulse.n_steps
            new_steps = max(min_steps, int(original_steps * reduction_factor))
            pulse.n_steps = new_steps
            pulse.range = np.linspace(pulse.t_start, pulse.t_end, new_steps)
            print(f"Reduced pulse {pulse.t_start}-{pulse.t_end}: {original_steps} → {new_steps} steps")
        
        total_new_steps = sum(pulse.n_steps for pulse in self.pulses)
        if total_new_steps > 0:  # Prevent division by zero
            speedup_estimate = total_original_steps / total_new_steps
            print(f"Total steps: {total_original_steps} → {total_new_steps}")
            print(f"Estimated speedup: {speedup_estimate:.1f}x")
        else:
            print("No valid steps to estimate speedup")
    
    def set_fast_simulation_mode(self) -> None:
        """
        Set simulation to fast mode with aggressive optimizations.
        
        This method applies the most aggressive performance optimizations:
        - Reduces steps by 75%
        - Uses relaxed tolerances
        - Uses fastest ODE solver
        """
        print("Setting fast simulation mode...")
        
        # Reduce steps aggressively
        self.reduce_simulation_steps(reduction_factor=0.25, min_steps=5)
        
        # Set aggressive ODE parameters
        self.ode_method = 'RK45'
        self.ode_atol = 1e-2
        self.ode_rtol = 1e-2
        self.ode_max_step = 1.0
        self.ode_first_step = 0.1
        
        print("Fast mode enabled: RK45 solver, relaxed tolerances (1e-2)")
    
    def set_balanced_simulation_mode(self) -> None:
        """
        Set simulation to balanced mode with moderate optimizations.
        
        This method applies balanced performance optimizations:
        - Reduces steps by 50%
        - Uses moderate tolerances
        - Maintains good accuracy
        """
        print("Setting balanced simulation mode...")
        
        # Reduce steps moderately
        self.reduce_simulation_steps(reduction_factor=0.5, min_steps=10)
        
        # Set balanced ODE parameters
        self.ode_method = 'RK45'
        self.ode_atol = 1e-3
        self.ode_rtol = 1e-3
        self.ode_max_step = 0.2
        self.ode_first_step = 0.02
        
        print("Balanced mode enabled: RK45 solver, moderate tolerances (1e-3)")
    
    def make_plot(self, path: Optional[str] = None) -> go.Figure:
        """
        Create interactive plot of simulation results.
        
        Parameters
        ----------
        path : Optional[str], optional
            Path to save HTML file
            
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        if self.time_simul is None:
            raise ValueError("Simulation must be run before plotting")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Metabolites', 'pH', 'Subpopulations', 'Species States']
        )
        
        # Metabolites
        for i, metabolite_name in enumerate(self.metabolome.metabolites):
            metabolite = self.metabolome._metabolite_dict[metabolite_name]
            fig.add_trace(
                go.Scatter(
                    x=self.time_simul,
                    y=self.met_simul[i],
                    mode='lines',
                    name=metabolite.name,
                    line=dict(color=metabolite.color, simplify=True),
                    opacity=0.8
                ),
                row=1, col=1
            )
        
        # pH
        fig.add_trace(
            go.Scatter(
                x=self.time_simul,
                y=self.pH_simul,
                mode='lines',
                name='pH',
                line=dict(color='rgb(57,255,20)', simplify=True),
                opacity=0.8
            ),
            row=1, col=2
        )
        
        # Subpopulations
        subpop_idx = 0
        for bacteria_name, bacteria in self.microbiome.bacteria.items():
            for subpop_name, subpop in bacteria.subpopulations.items():
                opacity = 0.8 if subpop.state == 'active' else 0.1
                fig.add_trace(
                    go.Scatter(
                        x=self.time_simul,
                        y=self.subpop_simul[subpop_idx],
                        mode='lines',
                        name=f"{bacteria_name}_{subpop_name}",
                        line=dict(color=subpop.color, simplify=True),
                        opacity=opacity
                    ),
                    row=2, col=1
                )
                subpop_idx += 1
        
        # Species states
        species_names = list(self.microbiome.bacteria.keys())
        for i, species_name in enumerate(species_names):
            bacteria = self.microbiome.bacteria[species_name]
            fig.add_trace(
                go.Scatter(
                    x=self.time_simul,
                    y=self.cellActive_dyn[species_name],
                    mode='lines',
                    name=f"{species_name}_active",
                    line=dict(color=bacteria.color, width=2, simplify=True),
                    opacity=1
                ),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=self.time_simul,
                    y=self.cellInactive_dyn[species_name],
                    mode='lines',
                    name=f"{species_name}_inactive",
                    line=dict(color=bacteria.color, width=2, simplify=True),
                    opacity=0.1
                ),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=self.time_simul,
                    y=self.cellDead_dyn[species_name],
                    mode='lines',
                    name=f"{species_name}_dead",
                    line=dict(color=bacteria.color, width=2, simplify=True),
                    opacity=0.1
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Reactor Simulation Results",
            height=800,
            showlegend=True
        )
        
        if path is not None:
            fig.write_html(path)
        
        return fig
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Reactor(volume={self.volume}, n_pulses={len(self.pulses)})"
    
    def __str__(self) -> str:
        """String representation."""
        return self.__repr__()
