import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.stats import poisson
from typing import List, Tuple
import random

class Neuron:
    def __init__(self, v_rest, sample_rate, thr, tf, t_ref, lmbda, seed=None):
        self.v_rest = v_rest
        self.sample_rate = sample_rate
        self.h = 1 / sample_rate
        self.thr = thr
        self.tf = tf
        self.t_ref = t_ref
        self.lmbda = lmbda
        self.length_v = int(tf * sample_rate)
        self.t_ref_i = int(t_ref * sample_rate)
        self.v_values = np.zeros(self.length_v)
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def f(self, t, v):
        I_0 = 0.01
        R = 10**4
        tau = 0.02
        dv_dt = (- (v-self.v_rest) + R * I_0)/tau
        return dv_dt

    def solve_ode(self, y0, t0):
        sol = solve_ivp(self.f, [t0, self.tf], [y0], method='RK45', t_eval=np.linspace(t0, self.tf, 1000))
        return sol.y[0], sol.t
        
    def identify_neuron_spikes(self, neuron_signal) -> List[int]:
        # the positions of the spikes in the signal
        return [i for i, value in enumerate(neuron_signal) if value >= max(neuron_signal)]

    def simulate(self):
        pass

    def plot_signal(self):
        fig, ax = plt.subplots(figsize=(12, 3))
        neuron_signal = self.v_values
        time = np.arange(len(neuron_signal)) / self.sample_rate
        ax.plot(time, neuron_signal, c="k", lw=0.5)
        ax.axhline(y=self.thr, c="grey", linewidth=0.5, ls=":", zorder=0, label=f"threshold {self.thr}")
        fig.legend()
        ax.set_title('Signal Emitted by Neuron')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (mV)')
        ax.set_xlim(0, np.max(time))
        plt.tight_layout()
        plt.show()
        

class EulerLIF(Neuron):
    def __init__(self, v_rest, sample_rate, thr, tf, t_ref, lmbda, seed=None):
        super().__init__(v_rest, sample_rate, thr, tf, t_ref, lmbda, seed)
        
    def solve_ode(self, y0, t0):
        # uses the euler ODE method
        y_values = []
        t = t0
        y = y0
        while y < self.thr:
            t += self.h
            y += self.h * self.f(t, y)
            y_values.append(y)
        return y_values, t
    
    def simulate(self):
        i = 0
        t = 0
        step = 0
        while i < self.length_v:
            rnd_thr = random.random()
            prob = poisson.cdf(step, mu=self.lmbda)
            step += 1
            if prob < rnd_thr:
                rnd_time = int(self.t_ref_i * 4 * random.random())
                if i + rnd_time < self.length_v:
                    self.v_values[i:i+rnd_time] = self.v_rest
                    i = i + rnd_time
                    t = i * self.h
                else:
                    self.v_values[i:self.length_v] = self.v_rest
                    break
            else:
                v_t, t = self.solve_ode(self.v_rest, t)
                new_i = int(round(t / self.h))
                if new_i + self.t_ref_i + 2 <= self.length_v:
                    self.v_values[i:new_i] = v_t
                    i = new_i + 1
                    self.v_values[i] = self.v_values[i-1] + 80
                    i += 1
                    self.v_values[i] = self.v_rest - 10
                    i += 1
                    new_i = i + self.t_ref_i
                    self.v_values[i:new_i] = self.v_rest
                    i = new_i
                    t = i * self.h
                else:
                    self.v_values[i:self.length_v] = self.v_rest
                    break
        return self.v_values
    

class RKLIF(Neuron):
    def __init__(self, v_rest, sample_rate, thr, tf, t_ref, lmbda, seed=None):
        super().__init__(v_rest, sample_rate, thr, tf, t_ref, lmbda, seed)

    def threshold_event(self, t, v):
        return v[0] - self.thr

    threshold_event.terminal = True
    threshold_event.direction = 1

    def solve_ode(self, y0, t0):
        sol = solve_ivp(self.f, [t0, self.tf], [y0], method='RK45', t_eval=np.linspace(t0, self.tf, self.length_v),
                        events=self.threshold_event)
        return sol.y[0], sol.t

    def simulate(self):
        i = 0
        t = 0
        step = 0
        while i < self.length_v:
            rnd_thr = random.random()
            prob = poisson.cdf(step, mu=self.lmbda)
            step += 1
            if prob < rnd_thr:
                rnd_time = int(self.t_ref_i * 4 * random.random())
                if i + rnd_time < self.length_v:
                    self.v_values[i:i+rnd_time] = self.v_rest
                    i = i + rnd_time
                    t = i * self.h
                else:
                    self.v_values[i:self.length_v] = self.v_rest
                    break
            else:
                v_t, t = self.solve_ode(self.v_rest, t)
                new_i = int(round(t[-1] / self.h))
                if new_i + self.t_ref_i + 2 <= self.length_v:
                    # Interpolate the solution to the original time steps
                    interp_func = interp1d(t, v_t, kind='linear', fill_value='extrapolate')
                    v_t_interp = interp_func(np.linspace(t[0], t[-1], new_i - i))

                    self.v_values[i:new_i] = v_t_interp
                    i = new_i + 1
                    self.v_values[i] = self.v_values[i-1] + 80
                    i += 1
                    self.v_values[i] = self.v_rest - 10
                    i += 1
                    new_i = i + self.t_ref_i
                    self.v_values[i:new_i] = self.v_rest
                    i = new_i
                    t = i * self.h
                else:
                    self.v_values[i:self.length_v] = self.v_rest
                    break
        return self.v_values