A. To run the code:
1 write the input.json to specify the parameters and configuration
2 move the input.json to your data directory
3 run 
  julia --threads [N_thread] calculation.jl [data_directory]
  
  e.g julia --threads 2 calculation.jl ./data
(or julia --threads 2 calculation_intg.jl ./data for the integro-diff eq.)
  
  

B. To plot the result

1 plot the density and the sum of density
julia plot.jl [data_directory]

2 plot the density, structure factor, radial structure factor, radial correlation function
julia plot_all.jl [data_directory]

The .mp4 file is generated in your data directory by default

C. a rule of thumb 
lattice spacing / time stepsize < 0.01


* Packages used 
SparseArrays
JSON
InteractiveUtils
Serialization
Plots
FFTW
Printf
Distributions
DifferentialEquations
JLD2
FileIO

* To install julia packages
using Pkg
Pkg.add("Package Name")





