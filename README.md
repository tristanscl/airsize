# AirSize

## About the project

### High level

* The "tool" itself, in its most versatile setting, i.e. for a general mission architecture, is the AirSize Python library. The UI elements were designed to fit the need of the benchmark and RFP that motivated the creation of the library. Although this UI enables parametric analysis, it has to be adapted in the case of a change on the mission architecture itself (e.g. new phase of flight). In the latter case, the AirSize Python library is the appropriate level of abstraction.
* The Python library is entirely vectorized with NumPy to speed up compute. 

### Low level

* All dimensional quantities are assumed to be given in SI units.
* Docstring writing for the different functions is minimalistic.
* ISA implementation is valid up until 20 km (65,617 ft)
* We conventionally name the weight fractions with lowercase and actual weights with uppercase. For example, the empty weight fraction will be refered to as `wE` in the code, whereas the empty weight will be refered to as `WE`.
* When we talk about `WF`, we generally talk about the net mass of fuel remaining at time $t$. By net, we mean that the minimum amount of fuel that should remain at the end of the mission should be subtracted to the actual pass of fuel to get `WF`.
* We use the value of $\beta$ at the begining of the flight phase for constraint analysis, as this is the most constraining value.
* We use the value of $\alpha$ at maximum altitude during the flight phase, as it decreases with altitude.
* We use the value of $q$ at maximum altitude during the flight phase, as min. $q$ is more constraining than max. $q$, assuming structural integrity.