import noise
import numpy as np
from functools import partial

def rand(box=(-1.,1.)):
    def with_time_seed(time, seed):
        rand01 = np.modf((12.9348*np.modf(np.sin(time * 58.5453123 + seed*3872.93)*234.293)[0])**2)[0]
        #cheap hash function from any real to reals between 0 and 1
        return np.interp(rand01, (0.0, 1.0), box)
    return with_time_seed

def discretize(signal, freq=1.):
    return lambda time, seed: signal(np.floor(time*freq)/freq, seed) 

def discontinuous(signal, discontinuous_freq=1, time_range=0.15):
    def with_time_seed(time, seed):
        discontiue_amount = discretize(rand, freq=discontinuous_freq)(time, seed)
        return signal(time_range*discontiue_amount**2, seed)
    return with_time_seed

def multi_seeder(f, repetitions):
    return lambda time, seed: np.array([
        f(time, seed+i*3.287382) for i in range(repetitions)])

def simplex(box=(-10.,10.), base_freq=1):
    return lambda time, seed: np.interp(np.vectorize(noise.snoise2)(time*base_freq, seed, 4), (-0.75, 0.75), box)

def combine_signals(f, *signals):
    return lambda time, seed: f(np.array([signal(time, seed) for signal in signals]))

def step(box=(-10, 10)):
    return discretize(rand(box), freq=0.3)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    seed = 34434
    time = np.linspace(10, 30, 1000)
    plt.plot(step((-1,1))(time, seed))
    plt.show()