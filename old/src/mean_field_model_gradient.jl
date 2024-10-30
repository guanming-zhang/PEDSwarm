import Base.joinpath
using SparseArrays
using JSON
using Serialization
using InteractiveUtils
using Base.Threads
using DifferentialEquations

include("finite_diff.jl")


mutable struct NumericalMeanField2D
    rng::Array{Float64,1}         # size of the domain is rng[1] by rng[2]
    npts::Array{Integer,1}        # number of points per dimension npts
    delta::Array{Float64,1}       # spatial discritization 
    time_scheme::String           # forward-euler or predictor-corrector
    params::Dict{String,Float64}  # model parameters
    dt::Float64                   # time step
    step_counter::Integer         # counter for the current steps
    num_th::Integer               # number of threads
    pts_per_th::Integer           # number of points handled by each thread
    # model variables
    x::Array{Float64,1}           # x coordinate 
    y::Array{Float64,1}           # y coordinate
    #cdiff_mat::Dict{Tuple{Integer,Integer},SparseMatrixCSC{Float64, Int64}} #sparse Totpliz matix for central difference
    block_cdiff_mat::Dict{Tuple{Integer,Integer},Array{SparseMatrixCSC{Float64, Int64},1}}
    # model variables
    rho::Array{Float64,1}         # density, a vector of N*N points (N = rng[1]*rng[2])
    drho::Array{Float64,1}        # change of density at each step
    sigma::Array{Float64,3}       # stress sigma, a 2*2*(N^2) matrix sigma[1,1,:] means sigma_xx
    mu::Array{Float64,1}          # chemical potential, a N^2-elements vector
    j::Array{Float64,2}           # j energy flux, a 2*(N^2) matrix j = div
    f::Array{Float64,2}           # f force density, a 2*(N^2) matrix f = div. sigma
    # auxiliary variables
    rho_store::Array{Float64,1}   # an intermediate variable for predictor-corrector scheme
    drho_store::Array{Float64,1}  # an intermediate varialbe for predictor-corrector scheme
    # flag variables
    is_initialized::Bool          # a flag showing if the initial condition is set
    is_parameters_set::Bool       # a flag showing if the parameters are set 
    # Julia integrator(initilaized as nothing)
    integrator::Any                 
end

function NumericalMeanField2D(x_max, y_max, nx, ny, dt, t_scheme="forward-Euler")
    rng = [x_max, y_max]
    npts = [nx, ny]
    delta = [x_max / nx, y_max / ny]
    time_scheme = t_scheme
    params = Dict{String,Float64}()
    step_counter = 0
    num_th = Threads.nthreads()
    pts_per_th = div(npts[1] * npts[2],num_th)
    if mod(npts[1]*npts[2],num_th) != 0
        error("the number of points must be divisible by the number of thread")
    end
    x = range(delta[1], x_max;length = nx)
    y = range(delta[2], y_max;length = ny)
    #=
    Dx = diff_mat2D(nx, ny, 1, 1) / delta[1]
    Dy = diff_mat2D(nx, ny, 2, 1) / delta[2]
    Dxy = Dx * Dy
    Dxx = diff_mat2D(nx, ny, 1, 2) / (delta[1]^2)
    Dyy = diff_mat2D(nx, ny, 2, 2) / (delta[2]^2)
    cdiff_mat = Dict((1, 0) => Dx, (0, 1) => Dy, (2, 0) => Dxx, (0, 2) => Dyy, (1, 1) => Dxy)
    =#
    if mod(nx*ny,num_th) != 0
        error("the number of points must be divisible by the number of thread")
    end
    block_cdiff_mat=Dict{Tuple{Integer,Integer},Array{SparseMatrixCSC{Float64, Int64},1}}()
    for odiff_x in 0:2
        for odiff_y in 0:2
            if odiff_x + odiff_y <=2 && odiff_x + odiff_y>0 
                odiff = (odiff_x,odiff_y)
                block_cdiff_mat[odiff] = cut_into_blocks(mixed_diff_mat2d(odiff,nx,ny,delta[1],delta[2]),num_th)
            end
        end
    end
    rho = zeros(Float64, nx * ny)
    drho = zeros(Float64, nx * ny)
    sigma = zeros(Float64, 2, 2, nx * ny)
    mu = zeros(Float64, nx * ny)
    j = zeros(Float64, 2, nx * ny)
    f = zeros(Float64, 2, nx * ny)
    rho_store = zeros(Float64, nx * ny)
    drho_store = zeros(Float64, nx * ny)

    NumericalMeanField2D(rng, npts, delta, time_scheme, params, dt, step_counter, num_th, pts_per_th,
        x, y,block_cdiff_mat, rho, drho, sigma, mu, j, f, rho_store, drho_store,false,false,nothing)
end


function set_model_params(model::NumericalMeanField2D, T, A2, A3, C, K, Gamma=1.0)
    model.params["T"] = T
    model.params["A2"] = A2
    model.params["A3"] = A3
    model.params["C"] = C
    model.params["K"] = K
    model.params["Gamma"] = Gamma
    model.is_parameters_set = true
end

function set_initial_condition(model::NumericalMeanField2D, rho::Array{Float64,2})
    model.rho = reshape(rho, model.npts[1] * model.npts[2])
    model.rho_store = reshape(rho, model.npts[1] * model.npts[2])
    model.step_counter = 0
    # we warp the model into julia ODE solver
    # tmax = 10000.0 is a large number to make sure t is in [0,tmax]
    # To avoid out-of-memory error caused by storing a large amount of data 
    # to the computer memory, we set save_on = false to avoid the aforementioned problem 
    # and then save the data to the hard disk at a regular time interval in calculation.jl.
    if model.time_scheme == "julia-Tsit5"
        prob = ODEProblem(wrapped_update!,model.rho,(0.0,1000.0),model)
        model.integrator = init(prob,Tsit5();save_on=false)
    elseif model.time_scheme == "julia-RK4"
        prob = ODEProblem(wrapped_update!,model.rho,(0.0,1000.0),model)
        model.integrator = init(prob,RK4();save_on=false)
    elseif model.time_scheme =="julia-TRBDF2"
        prob = ODEProblem(wrapped_update!,model.rho,(0.0,1000.0),model)
        model.integrator = init(prob,TRBDF2();save_on=false) 
    end
    model.is_initialized = true
end

function update_chemical_pot_parallel!(model::NumericalMeanField2D,npts_per_th,th_num)
    laplace_rho = zeros(Float64, npts_per_th)
    idx_rng = npts_per_th*(th_num-1)+1:npts_per_th*th_num
    for i in 1:2
        odiff = [0, 0]
        odiff[i] = 2
        laplace_rho += model.block_cdiff_mat[tuple(odiff...)][th_num] * model.rho
    end
    # mu is the chemical potential due to the interacting free energy
    model.mu[idx_rng] = (2.0*model.params["A2"] * model.rho[idx_rng] 
                        + 3.0*model.params["A3"] * model.rho[idx_rng] .^ 2
                        - 2.0*model.params["K"] * laplace_rho)
end
function update_stress_parallel!(model::NumericalMeanField2D,npts_per_th,th_num)
    idx_rng = npts_per_th*(th_num-1)+1:npts_per_th*th_num
    # calculate the stress
    for alpha in 1:2
        for beta in 1:2
            odiff = [0, 0]
            odiff[alpha] += 1
            odiff[beta] += 1
            # we should try to avoid boradcasting(e.g. .* ) when using sparse matrices
            # it will be out of memory error !!!
            model.sigma[alpha, beta, idx_rng] =  model.block_cdiff_mat[tuple(odiff...)][th_num]*model.rho
            model.sigma[alpha, beta, idx_rng] .*=  2.0*model.params["C"]*model.rho[idx_rng]
        end
    end
end

function calculate_flux_parallel!(model::NumericalMeanField2D,npts_per_th,th_num)
    idx_rng = npts_per_th*(th_num-1)+1:npts_per_th*th_num
    # calculate the free energy flux j
    for alpha in 1:2
        odiff = [0, 0]
        odiff[alpha] = 1
        grad_mu = model.block_cdiff_mat[tuple(odiff...)][th_num] * model.mu
        grad_rho = model.block_cdiff_mat[tuple(odiff...)][th_num] * model.rho
        model.j[alpha, idx_rng] = (model.rho[idx_rng] .* grad_mu / model.params["Gamma"]
                                +model.params["T"] * grad_rho / model.params["Gamma"])
    end
        
    # calculate the force density
    # try not to use fill since it will change the entire matrix
    for i in idx_rng
        for alpha in 1:2
            model.f[alpha,i] = 0.0
        end
    end
    
    for alpha in 1:2
        for beta in 1:2
            odiff = [0, 0]
            odiff[beta] = 1
            model.f[alpha,idx_rng] += model.block_cdiff_mat[tuple(odiff...)][th_num] * model.sigma[alpha, beta, :]
        end
    end
end

function update_drho_parallel!(model::NumericalMeanField2D,npts_per_th,th_num)
    idx_rng = npts_per_th*(th_num-1)+1:npts_per_th*th_num
    model.drho[idx_rng] .= 0.0
    # calculate the ∇rho.∇mu
    for alpha in 1:2
        odiff = [0, 0]
        odiff[alpha] = 1
        grad_mu = model.block_cdiff_mat[tuple(odiff...)][th_num] * model.mu
        grad_rho = model.block_cdiff_mat[tuple(odiff...)][th_num] * model.rho
        model.drho[idx_rng] += grad_mu.*grad_rho
    end
    
    # calculate (rho (∇^2)mu) + T(∇^2)rho
    for alpha in 1:2
        odiff = [0, 0]
        odiff[alpha] = 2
        # calculate rho (∇^2)mu
        model.drho[idx_rng] += model.rho[idx_rng].*(model.block_cdiff_mat[tuple(odiff...)][th_num] * model.mu)
        # calcuate T(∇^2)rho
        model.drho[idx_rng] += model.params["T"]*model.block_cdiff_mat[tuple(odiff...)][th_num] * model.rho
    end
    
    # divided by the mobility coeff,Gamma
    model.drho /= model.params["Gamma"]

    # calculate the stress part ∇∇:sigma
    for alpha in 1:2
        for beta in 1:2
            odiff = [0, 0]
            odiff[alpha] += 1
            odiff[beta] += 1
            model.drho[idx_rng] += model.block_cdiff_mat[tuple(odiff...)][th_num] * model.sigma[alpha, beta, :]
        end
    end
    
end

function update_parallel!(model::NumericalMeanField2D)
    # note that these parallel bolcks cannot be merged in one thread for loop
    Threads.@threads for th_num in 1:model.num_th
        update_chemical_pot_parallel!(model,model.pts_per_th,th_num)
    end
    Threads.@threads for th_num in 1:model.num_th
        update_stress_parallel!(model,model.pts_per_th,th_num)
    end
    Threads.@threads for th_num in 1:model.num_th
        update_drho_parallel!(model,model.pts_per_th,th_num)
    end
end

function wrapped_update!(drho,rho,model,t)
    """
    we wrap the update_drho_parallel function to make it compatable to Julia ODE solver
    which is of the form du/dt = f(du,u,p,t) where u is the unknown, p is the parameter
    in our case we update drho/dt = wrapped_update(drhorho,model,t) where we use the model as
    the parameter
    """
    model.rho = rho
    update_parallel!(model)
    drho .= model.drho
end


function one_step(model::NumericalMeanField2D)
    if !model.is_initialized
        error("Please set the initialcondition before fowarding in time")
    elseif !model.is_parameters_set
        error("Please specify the value of model prameters before fowarding in time")
    end
    if model.time_scheme == "predictor-corrector"
        model.rho_store = copy(model.rho)
        update_parallel!(model)
        model.rho += model.drho * model.dt
        model.drho_store = copy(model.drho) # copy by value
        Threads.@threads for th_num in 1:model.num_th
            update_drho_parallel!(model,model.pts_per_th,th_num)
        end
        model.rho = model.rho_store + 0.5 * model.dt * (model.drho + model.drho_store)
    elseif model.time_scheme == "RK2"
        model.rho_store = copy(model.rho)
        # calclate k1 and update rho_new
        update_parallel!(model)
        k1 = model.dt*model.drho
        # calclate k2 and update rho_new
        model.rho = model.rho_store + k1
        update_parallel!(model)
        k2 = model.dt*model.drho
        model.rho = model.rho_store + 0.5*(k1 + k2)
    elseif model.time_scheme == "forward-Euler"
        update_parallel!(model)
        model.rho += model.drho * model.dt
    else
        error("one_step() only works for time-scheme = 
        [predictor-corrector, forward-Euler, RK2]")
    end
    model.step_counter += 1
end

function n_steps(model::NumericalMeanField2D,n)
    if !model.is_initialized
        error("Please set the initialcondition before fowarding in time")
    elseif !model.is_parameters_set
        error("Please specify the value of model prameters before fowarding in time")
    end

    if model.time_scheme in ["julia-Tsit5","julia-TRBDF2","julia-RK4"]
        step!(model.integrator,n*model.dt,true)
        model.rho = model.integrator.u
    else 
        error("n_steps() only works for time-scheme = 
              [julia-Tsit5, julia-TRBDF2, julia-RK4]")
    end
    model.step_counter += n
end

function save_data(model::NumericalMeanField2D, dir_str::String,compression::Bool)
    pts_per_thread = div(model.npts[1] * model.npts[2],model.num_th)
    if mod(model.npts[1]*model.npts[2],model.num_th) == 0
        Threads.@threads for th_num in 1:model.num_th
            calculate_flux_parallel!(model,pts_per_thread,th_num)
        end
    else
        error("the number of points must be divisible by the number of thread")
    end
    file_str = "Frame_$(model.step_counter).json"
    file_path = joinpath(dir_str, file_str)
    dict_data = Dict("rho" => model.rho, "j" => model.j, "f" => model.f,
                     "t"=>model.dt*model.step_counter,"step_num"=>model.step_counter)
    json_data = JSON.json(dict_data)
    open(file_path, "w") do f
        write(f, json_data)
    end
    if compression
        zip_file = joinpath(dir_str, "Frame_$(model.step_counter).zip")
        zip_cmd = ["zip","-m", "-j",zip_file,file_path]
        run(`$zip_cmd`)
    end
end


#############--------some notes--------############
# Try not to use boradcasting(.*) on sparse matrices
# or you will get an out-of-memory error in julia
# Try not using indexing on sparse matrices since 
# it is very slow
###################################################

# unparalleled update function 
#=
function update_drho!(model::NumericalMeanField2D)
    laplace_rho = zeros(Float64, model.npts[1] * model.npts[2])
    for i in 1:2
        odiff = [0, 0]
        odiff[i] = 2
        laplace_rho += model.cdiff_mat[tuple(odiff...)] * model.rho
    end
    # mu is the chemical potential due to the interacting free energy
    model.mu = -2.0*model.params["A2"] * model.rho + 3.0*model.params["A3"] * model.rho .^ 2- 2.0*model.params["K"] * laplace_rho
    # calculate the free energy flux j
    for alpha in 1:2
        odiff = [0, 0]
        odiff[alpha] = 1
        grad_mu = model.cdiff_mat[tuple(odiff...)] * model.mu
        grad_rho = model.cdiff_mat[tuple(odiff...)] * model.rho
        model.j[alpha, :] = model.rho .* grad_mu / model.params["Gamma"]+model.params["T"] * grad_rho / model.params["Gamma"]
    end
    # calculate the stress
    for alpha in 1:2
        for beta in 1:2
            odiff = [0, 0]
            odiff[alpha] += 1
            odiff[beta] += 1
            # we should try to avoid boradcasting(e.g. .* ) when using sparse matrices
            # it will be out of memory error !!!
            model.sigma[alpha, beta, :] =  model.cdiff_mat[tuple(odiff...)]*model.rho
            model.sigma[alpha, beta, :] .*=  model.params["C"]*model.rho
        end
    end
    
    # calculate the force density
    fill!(model.f, 0.0)
    for alpha in 1:2
        for beta in 1:2 
            odiff = [0, 0]
            odiff[beta] = 1
            model.f[alpha,:] += model.cdiff_mat[tuple(odiff...)] * model.sigma[alpha, beta, :]
        end
    end
    fill!(model.drho, 0.0)
    for alpha in 1:2
        odiff = [0, 0]
        odiff[alpha] = 1
        model.drho += model.cdiff_mat[tuple(odiff...)] * (model.j[alpha, :] - model.f[alpha, :])
    end
end
=#
