


Discretizers implement step
Samplers have a discretizer and two methods:
    sampler.simulate(x0, T, dt) calls sampler.step(x, dt) until T is reached