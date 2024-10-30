include("./src/mean_field_model_gradient.jl")
include("./src/utils.jl")
using Random, Distributions
using Printf
if length(ARGS) < 1 
    error("Please specify the data directory containing input.json ")
end
data_dir = ARGS[1] 
input_data = read_input(Base.joinpath(data_dir,"input.json"))
x_max,y_max = input_data["range"]
nx,ny = input_data["npts"]
dt = input_data["dt"]
#model parameters
if haskey(input_data,"phi") && haskey(input_data,"R") 
    error("Please do not specify phi and R at the same time")
end
if haskey(input_data,"rel_epsilon") && haskey(input_data,"phi") 
    S = 2.0*pi
    R = sqrt(input_data["phi"]*x_max*y_max/(input_data["N"]*pi))
    eps = input_data["rel_epsilon"]*2.0*R
    input_data["A"] = S*R^2
    input_data["B"] = 0.0
    input_data["C"] = eps*S/24.0*R^4
    input_data["K"] = S*5.0/24.0*R^4
    input_data["Gamma"] = 1.0/eps
    println("A,B,C,Gamma are overwriten and adjusted based on the vaule of phi and epsilon")
    # overwrite the old input file
    file_path = Base.joinpath(data_dir,"input.json")
    open(file_path, "w") do f
        write(f,JSON.json(input_data,4))
    end
    println(R)
end

if haskey(input_data,"rel_epsilon") && haskey(input_data,"R") 
    S = 2.0*pi
    R = input_data["R"]
    eps = input_data["rel_epsilon"]*2.0*R
    input_data["A"] = S*R^2
    input_data["B"] = 0.0
    input_data["C"] = eps*S/24.0*R^4
    input_data["K"] = S*5.0/24.0*R^4
    input_data["Gamma"] = 1.0/eps
    input_data["R"] = R
    println("A,B,C,Gamma are overwriten and adjusted based on the vaule of R and epsilon")
    # overwrite the old input file
    file_path = Base.joinpath(data_dir,"input.json")
    open(file_path, "w") do f
        write(f,JSON.json(input_data,4))
    end
    @printf("the particle radius is set to %1.4f \n",R)
end

T = input_data["T"] 
A = input_data["A"]
B = input_data["B"]
C = input_data["C"]
K = input_data["K"]
Gamma = input_data["Gamma"]
println(input_data)

model = NumericalMeanField2D(x_max, y_max, nx, ny, dt,input_data["time_scheme"])
set_model_params(model,T,A,B,C,Gamma)
# set the initial condition
if input_data["iv"] == "Gaussian-profile"
    x = model.x
    y = model.y'
    sx = input_data["iv_sx"]
    sy  = input_data["iv_sy"]
    a = 1.0/(pi*sx*sy)*input_data["N"]
    rho0 = @. a*exp(-((x-0.5*x_max)/sx)^2 -((y-0.5*y_max)/sy)^2)
elseif input_data["iv"] == "random-normal"
    sr = input_data["iv_srho"]
    mu = input_data["N"]/(x_max*y_max)
    rho0 = rand(Normal(mu, sr*mu), nx,ny)
end
set_initial_condition(model,rho0)
println("the current step number is $(model.step_counter)")
# save the initial value
save_data(model,data_dir,input_data["compression"]>0)

for s in 1:input_data["n_steps"]
    if model.time_scheme in ["forward-Euler","predictor-corrector","RK2"] 
        one_step(model)
    end
    if mod(s,input_data["n_save"]) == 0
        if model.time_scheme in ["julia-Tsit5","julia-TRBDF2","julia-RK4"]
            n_steps(model,input_data["n_save"])
        end
        @printf("The current time step number: %i \n", s)
        if any(isnan, model.rho)
            error("NaN detected, program stops")
        end
        save_data(model,data_dir,input_data["compression"]>0)
    end
end