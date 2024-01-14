# Cooling in Astrophysical Plasmas
This is the code related to a project for the course [Introduction to Plasma Dynamics](https://onderwijsaanbod.kuleuven.be//2023/syllabi/e/G0P71BE.htm). In it is implemented a Boris pusher and some cooling methods for Plasma


## Boris scheme
The Boris scheme is explained very well in [(Ripperda et al 2018)](https://doi.org/10.3847/1538-4365/aab114). Our implementation is based on the [Tristan-MP](https://github.com/PrincetonUniversity/tristan-mp-v2) code

## Cooling
The radiation reaction drag force is implemented as prescribed by the [Tristan-MP wiki](https://princetonuniversity.github.io/tristan-v2/tristanv2-radiation.html).
Furthermore, particle escape and injection is implemented.