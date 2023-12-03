$$
\newcommand{\dd}{\mathrm{d}}
\newcommand{\abs}[1]{|#1|}
$$

# Plasma dynamics Project
Particle acceleration and radiative cooling in high-energy astrophysical plasmas.

## Synopsis from Course Notes

The leap-frog algorithm in the Boris implementation is what will move the particles around based on some force. This is called a particle pusher. The leap frog algorithm uses a staggered temporal grid: positions at integer time, and velocity at half integers time: $\vec{x}_{n+1} =  \vec{x}_n + \Delta t \vec{v}_{n+1/2}$, and $\vec{v}_{n+1/2} = \vec{v}_{n-1/2} + \Delta t \frac{q}{m} (\vec{E}(\vec{x}_n) + \frac{\vec{v}_{n+1/2} + \vec{v}_{n-1/2}}{2}\cdot \vec{B}(\vec{x}_n)$. The latter can be explicitly solved without some matrix inversion like to solve the Poisson equation; this is the Boris solution. There is a stability to consider: the time steps should be small enough compared to the mass to charge ratio of the species (in our case electrons); see Equation (5.29) in the lecture notes.

We will likely need to a PIC method (because coding a Barnes-Hut tree algorithm is a bit much, which we would need in the strongly coupled case), where 1e5 would, I guess, refer to the amount of computational particles. This means we should address our temperature, so our plasmas are actually weakly coupled.

For the PIC, we'll need shape functions. Velocity shape often Dirac delta, spatial shape functions are b-splines. From the moments of the Vlasov equation for the shape functions one gets the equations of motion (Equation 6.24). We have the luxury of not having to update the fields. The EOMs are discretized with the Boris pusher.

Stability is rediscussed in Section 6.10, and how this plays in with conservation laws to verify and other diagnostics in Section 6.11. 

Recap of discretization: the fields are discretized in space and static, the particles are discretized in time and not in space, which means that at every point in time the fields will have to be interpolated with a `scipy` implementation to get $\vec{E}(\vec{x}_n)$.

## Boris scheme

Follow [(Ripperda et al 2018)](doi.org/10.3847/1538-4365/aab114).
The aim is to apply the Boris pusher first, and then apply a drag force.

## Radiative cooling

The cooling is added as a drag force, i.e. proportional to the velocity. This is a third type of term added to the force, which means the Boris pusher from above is modified. A relativistic Boris scheme is given in the RUNKO paper, Appendix C. We should understand this thing, I would say before starting to make a pusher, since the relativistic framework will make the port a hassle.

$$
    \frac{\dd u}{\dd t} = \frac{\abs{e}}{m_e} \frac{B_\text{norm}}{c} \frac{\beta_\text{rec}}{\gamma_\text{syn}^2} \left[ \kappa_R - \gamma^2 \chi_R^2 \beta \right]
$$