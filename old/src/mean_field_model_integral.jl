include("finite_diff.jl")
include("potentials.jl")
include("utils.jl")

import Base.joinpath
using SparseArrays
using JSON
using Serialization
using InteractiveUtils
using Base.Threads
using DifferentialEquations
using LinearAlgebra
using Plots
using Infiltrator

debug_idx = 149


mutable struct NumericalMeanField2D
    rng::Array{Float64,1}         # size of the domain is rng[1] by rng[2] for [0,rng[1]]*[0,rng[2]]
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
    mesh_x::Array{Float64,2}      # meshgrid for x
    mesh_y::Array{Float64,2}      # meshgrid for y
    # sparse central difference matrix 
    block_cdiff_mat::Dict{Tuple{Integer,Integer},Array{SparseMatrixCSC{Float64, Int64},1}}
    # sparse froward difference matrix 
    block_fdiff_mat::Dict{Tuple{Integer,Integer},Array{SparseMatrixCSC{Float64, Int64},1}}
    # sparse backward difference matrix 
    block_bdiff_mat::Dict{Tuple{Integer,Integer},Array{SparseMatrixCSC{Float64, Int64},1}}
    # sparse square stencil laplacian matrix 
    block_laplacian_mat::Array{SparseMatrixCSC{Float64, Int64},1}
    # model variables
    rho::Array{Float64,1}         # density, a vector of N*N points (N = rng[1]*rng[2])
    drho::Array{Float64,1}        # change of density at each step
    sigma::Array{Float64,3}       # stress sigma, a 2*2*(N^2) matrix sigma[1,1,:] means sigma_xx
    potential_grad_x::Array{Float64,1}  # the potential grad for the x direction
    potential_grad_y::Array{Float64,1}  # the potential grad for the y direction
    center_of_mass::Array{Float64, 1}
    intermediate::Dict{String, Array{Float64, 1}}
    potential_kernel::Array{Float64,2}  # the kernel for potential functional
    stress_kernel::Dict{Tuple{Int64,Int64},Array{Float64,2}}  # the kernel for stress calculation
    mu::Array{Float64,1}          # chemical potential, a N^2-elements vector
    j::Array{Float64,2}           # j energy flux, a 2*(N^2) matrix j = rho grad mu
    f::Array{Float64,2}           # f force density, a 2*(N^2) matrix f = div. sigma
    #noise term addition
    #white_noise::Array{Float64,1}                                # a Gaussian white noise vector of N*N points (N = rng[1]*rng[2])
    multi_noise::Dict{Tuple{Int64},Array{Float64,2}}             # multiplicative noise =white noise*sqrt(rho), a 2*(N^2) matrix, multi_noise[1,:] means multi_noise_x
    force_noise::Array{Float64,2}                                # noise force, a 2*(N^2) matrix, force_noise[1,:] means force_noise_x
    noise_kernel::Dict{Tuple{Int64,Int64},Array{Float64,2}}      # the kernel for noise calculation
    t::Float64                    # time elapsed
    # auxiliary variables
    rho_store::Array{Float64,1}         # an intermediate variable for predictor-corrector scheme
    drho_store::Array{Float64,1}        # an intermediate varialbe for predictor-corrector scheme
    # auxiliary variables for reducing runtime memory allocation 
    # usage: grad_mu_fd[thread_id][lattice_index]
    grad_mu_fd::Array{Float64,2}        # an intermediate variable for gradient of mu using forward difference scheme
    grad_rho_fd::Array{Float64,2}       # an intermediate variable for gradient of rho using forward difference scheme
    grad_rho_bd::Array{Float64,2}       # an intermediate variable for gradient of rho using backward difference scheme
    grad_mu_cd::Array{Float64,2}        # an intermediate variable for gradient of mu using central difference scheme
    grad_rho_cd::Array{Float64,2}       # an intermediate variable for gradient of rho using central difference scheme
    grad_sigma_cd::Array{Float64,2}     # an intermediate variable for gradient of sigma using central difference scheme
    grad_force_noise_cd::Array{Float64,2}# an intermediate variable for gradient of force_noise using central difference scheme
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
    t = 0
    if mod(npts[1]*npts[2],num_th) != 0
        error("the number of points must be divisible by the number of thread")
    end
    x = range(delta[1], x_max;length = nx)
    y = range(delta[2], y_max;length = ny)
    mesh_x = repeat(x, 1, length(y))
    mesh_y = repeat(transpose(y), length(x), 1)

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

    # generate block central difference matrix
    block_cdiff_mat=Dict{Tuple{Integer,Integer},Array{SparseMatrixCSC{Float64, Int64},1}}()
    for odiff_x in 0:2
        for odiff_y in 0:2
            if odiff_x + odiff_y <=2 && odiff_x + odiff_y>0 
                odiff = (odiff_x,odiff_y)
                block_cdiff_mat[odiff] = cut_into_blocks(mixed_diff_mat2d(odiff,nx,ny,delta[1],delta[2],"central",2),num_th)
            end
        end
    end

    # generate block forward difference matrix
    block_fdiff_mat=Dict{Tuple{Integer,Integer},Array{SparseMatrixCSC{Float64, Int64},1}}()
    for odiff_x in 0:2
        for odiff_y in 0:2
            if odiff_x + odiff_y <=2 && odiff_x + odiff_y>0 
                odiff = (odiff_x,odiff_y)
                block_fdiff_mat[odiff] = cut_into_blocks(mixed_diff_mat2d(odiff,nx,ny,delta[1],delta[2],"forward",1),num_th)
            end
        end
    end

    # generate block backward difference matrix
    block_bdiff_mat=Dict{Tuple{Integer,Integer},Array{SparseMatrixCSC{Float64, Int64},1}}()
    for odiff_x in 0:2
        for odiff_y in 0:2
            if odiff_x + odiff_y <=2 && odiff_x + odiff_y>0 
                odiff = (odiff_x,odiff_y)
                block_bdiff_mat[odiff] = cut_into_blocks(mixed_diff_mat2d(odiff,nx,ny,delta[1],delta[2],"backward",1),num_th)
            end
        end
    end
    # use 2nd order accuary central difference and 1st order forward difference to make use the mass is conserved
    
    block_laplacian_mat = Array{SparseMatrixCSC{Float64, Int64},1}()
    block_laplacian_mat = cut_into_blocks(laplacian_square_stencil(nx,ny,delta[1]),num_th)

    rho = zeros(Float64, nx * ny)
    drho = zeros(Float64, nx * ny)
    sigma = zeros(Float64, 2, 2, nx * ny)
    mu = zeros(Float64, nx * ny)
    j = zeros(Float64, 2, nx * ny)
    f = zeros(Float64, 2, nx * ny)
    rho_store = zeros(Float64, nx * ny)
    drho_store = zeros(Float64, nx * ny)

    potential_grad_x = zeros(Float64, nx*ny)
    potential_grad_y = zeros(Float64, nx*ny)

    center_of_mass = zeros(2)
    grad_mu_fd = zeros(Float64, num_th, pts_per_th)
    grad_rho_fd = zeros(Float64, num_th, pts_per_th)
    grad_rho_bd = zeros(Float64, num_th, pts_per_th)
    grad_mu_cd = zeros(Float64, num_th, pts_per_th)
    grad_rho_cd = zeros(Float64, num_th, pts_per_th)
    grad_sigma_cd = zeros(Float64, num_th, pts_per_th)
    grad_force_noise_cd = zeros(Float64, num_th, pts_per_th)
    # initialize the kernel, they will be reset later
    potential_kernel = zeros(Float64,21,21)
    stress_kernel = Dict{Tuple{Int64,Int64},Array{Float64,2}}()
    #noise_term_additions
    multi_noise = Dict{Tuple{Int64},Array{Float64,2}}()        
    force_noise = zeros(Float64, 2, nx * ny)       
    noise_kernel = Dict{Tuple{Int64,Int64},Array{Float64,2}}()
    intermediate = Dict(
        "potential_force" => [0.0, 0.0]
    )
    NumericalMeanField2D(rng, npts, delta, time_scheme, params, dt, step_counter, num_th, pts_per_th,
        x, y, mesh_x, mesh_y, block_cdiff_mat, block_fdiff_mat, block_bdiff_mat, block_laplacian_mat, rho, drho, sigma, potential_grad_x, potential_grad_y, center_of_mass, intermediate, potential_kernel, stress_kernel, mu, j, f,
        multi_noise, force_noise, noise_kernel, t, rho_store, drho_store, grad_mu_fd, grad_rho_bd, grad_rho_fd, grad_mu_cd,
        grad_rho_cd, grad_sigma_cd, grad_force_noise_cd, false, false, nothing)
end


function set_model_params(model::NumericalMeanField2D, T, D, R, alpha, Gamma=1.0)
    model.params["T"] = T
    model.params["D"] = D # strength of the stress
    model.params["R"] = R # the radius of the particles
    model.params["Gamma"] = Gamma
    model.params["alpha"] = alpha # The strength of the attraction force

    if sqrt(model.rng[1]*model.rng[2]) < 4.0*R
        error("The domain size must be larger than 4R")
    end
    set_kernels(model)
    model.is_parameters_set = true
end

function set_kernels(model::NumericalMeanField2D)
    # the kernel is initialized here
    ackley_params = (20, 0.2, 2π)
    ackley_center = 0.2 .* model.rng

    # shift the center of the ackley potential
    @views ackley_x = model.mesh_x .- ackley_center[1]
    @views ackley_y = model.mesh_y .- ackley_center[2]

    # TODO: for some reason, "@." doesn't work, but the following one does
    # potential_grad_temp = ackley_grad_x.(ackley_x, ackley_y, Ref(ackley_params))
    # model.potential_grad_x .= potential_grad_temp
    potential_grad_x = ackley_grad_x.(ackley_x, ackley_y, Ref(ackley_params))
    potential_grad_y = ackley_grad_y.(ackley_x, ackley_y, Ref(ackley_params))
    
    # visualize the potential
    potential = ackley.(ackley_x, ackley_y, Ref(ackley_params))
    heatmap(potential, title="Potential", xlabel="X index", ylabel="Y index")
    savefig("potential.png")  # Save the plot to a file

    # visualize the potential gradient
    heatmap(potential_grad_x, title="Potential Gradient X", xlabel="X index", ylabel="Y index")
    savefig("potential_grad_x.png")    
    heatmap(potential_grad_y, title="Potential Gradient X", xlabel="X index", ylabel="Y index")    
    savefig("potential_grad_y.png")

    grad = reshape(model.block_fdiff_mat[tuple(1, 0)][1] * vec(potential), (model.npts[1], model.npts[2]))
    heatmap(grad)
    savefig("grad_diff.png")

    model.potential_grad_x = vec(potential_grad_x)
    model.potential_grad_y = vec(potential_grad_y)

end


function set_initial_condition(model::NumericalMeanField2D, rho::Array{Float64,2})
    model.rho = reshape(rho, model.npts[1] * model.npts[2])
    model.rho_store = reshape(rho, model.npts[1] * model.npts[2])
    model.step_counter = 0
    model.t = 0
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
    idx_rng = npts_per_th*(th_num-1)+1:npts_per_th*th_num
    # evaluate the chemical potential
    rho = reshape(model.rho,(model.npts[1],model.npts[2]))
    for idx in idx_rng
        # the chemical potential at (ix*dx,iy*dy)
        ix = mod_idx(idx,model.npts[1]) 
        iy = div(idx - ix,model.npts[1]) + 1
        model.mu[idx] = model.rho[idx]*convol2d(rho,model.potential_kernel,(ix,iy),model.delta[1],model.delta[2]) 
    end
end

function update_center_of_mass!(model::NumericalMeanField2D)
    rho = reshape(model.rho,(model.npts[1],model.npts[2]))
    
    model.center_of_mass[1] = sum(model.mesh_x.*rho) / sum(rho)
    model.center_of_mass[2] = sum(model.mesh_y.*rho) / sum(rho)
end

function update_stress_parallel!(model::NumericalMeanField2D,npts_per_th,th_num)
    # calculate the stress
    rho = reshape(model.rho,(model.npts[1],model.npts[2]))
    idx_rng = npts_per_th*(th_num-1)+1:npts_per_th*th_num
    for alpha in 1:2
        for beta in 1:2
            for idx in idx_rng
                # the stress at (ix*dx,iy*dy)
                ix = mod_idx(idx,model.npts[1]) 
                iy = div(idx - ix,model.npts[1]) + 1
                model.sigma[alpha, beta, idx] =  -0.5*model.params["D"]*model.rho[idx]*convol2d(
                                                 rho,model.stress_kernel[(alpha,beta)],(ix,iy),model.delta[1],model.delta[2])
            end
        end
    end
end

function update_noise_parallel!(model::NumericalMeanField2D,npts_per_th,th_num)
    # calculate the force in noise term
    idx_rng = npts_per_th*(th_num-1)+1:npts_per_th*th_num
    # assign zero to model.force_noise
    # try not to use fill since it will change the entire matrix
    for i in idx_rng
        for alpha in 1:2
            model.force_noise[alpha,i] = 0.0
        end
    end
    rho = reshape(model.rho,(model.npts[1],model.npts[2]))

    for alpha in 1:2
        for idx in idx_rng
            ix = mod_idx(idx,model.npts[1]) 
            iy = div(idx - ix,model.npts[1]) + 1
            model.multi_noise[(alpha,)][ix,iy] = sqrt(rho[ix,iy])*randn(Float64)*sqrt(model.dt)
        end
        for beta in 1:2
            for idx in idx_rng
                # the noise stress at (ix*dx,iy*dy)
                ix = mod_idx(idx,model.npts[1]) 
                iy = div(idx - ix,model.npts[1]) + 1
                model.force_noise[alpha, idx] += sqrt(model.params["D"])*model.rho[idx]*convol2d(
                    model.noise_kernel[(alpha,beta)],model.multi_noise[(beta,)],(ix,iy),model.delta[1],model.delta[2])
            end
        end
    end
end

function calculate_flux_parallel!(model::NumericalMeanField2D,npts_per_th,th_num)
    idx_rng = npts_per_th*(th_num-1)+1:npts_per_th*th_num
    # calculate the free energy flux j
    for alpha in 1:2
        odiff = [0, 0]
        odiff[alpha] = 1
        # grad_mu = model.block_cdiff_mat[tuple(odiff...)][th_num] * model.mu
        # grad_rho = model.block_cdiff_mat[tuple(odiff...)][th_num] * model.rho
        # model.j[alpha, idx_rng] = (model.rho[idx_rng] .* grad_mu / model.params["Gamma"]
        #                         + model.params["T"] * grad_rho / model.params["Gamma"])
        @views mul!(model.grad_mu_cd[th_num,:], model.block_cdiff_mat[tuple(odiff...)][th_num], model.mu)
        @views mul!(model.grad_rho_cd[th_num,:], model.block_cdiff_mat[tuple(odiff...)][th_num], model.rho)
        @views @. model.j[alpha, idx_rng] = (model.rho[idx_rng] .* model.grad_mu_cd[th_num,:]
                                            + model.params["T"] .* model.grad_rho_cd[th_num,:]) ./ model.params["Gamma"]
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
            # model.f[alpha,idx_rng] += model.block_cdiff_mat[tuple(odiff...)][th_num] * model.sigma[alpha, beta, :]
            @views mul!(model.grad_sigma_cd[th_num,:], model.block_cdiff_mat[tuple(odiff...)][th_num], model.sigma[alpha, beta, :])
            @views @. model.f[alpha,idx_rng] += model.grad_sigma_cd[th_num,:]
        end
    end
end


function update_drho_parallel!(model::NumericalMeanField2D,npts_per_th,th_num)
    idx_rng = npts_per_th*(th_num-1)+1:npts_per_th*th_num
    @views model.drho[idx_rng] .= 0.0
    
    # use chain rule to break ∇.(rho ∇mu) into ∇rho.∇mu + rho ∇^2 mu 
    # also calculate T(∇^2)rho
    # calculate the ∇rho.∇mu using the forward difference scheme

    N = sum(model.rho) * prod(model.delta)
    # println("N: ", N)

    rho_sq = reshape(model.rho, model.npts...)

    @views positions = (vec(model.mesh_x), vec(model.mesh_y))
    position_sq = (model.mesh_x, model.mesh_y)
    for alpha in 1:2
        odiff = [0, 0]
        odiff[alpha] = 1

        # Compute the gradient of rho
        @views mul!(model.grad_rho_fd[th_num,:], model.block_fdiff_mat[tuple(odiff...)][th_num], model.rho)
        @views mul!(model.grad_rho_bd[th_num,:], model.block_bdiff_mat[tuple(odiff...)][th_num], model.rho)

        velocity = -model.intermediate["potential_force"][alpha] .+ model.params["alpha"] .* (model.center_of_mass[alpha] .- positions[alpha][idx_rng])
        
        # Update using upwinding scheme
        model.drho[idx_rng] .+= -max.(velocity, 0) .* model.grad_rho_bd[th_num, :] 
        model.drho[idx_rng] .+= -min.(velocity, 0) .* model.grad_rho_bd[th_num, :] 

        # MARK: The following work
        # velocity_sq = model.center_of_mass[alpha] .- position_sq[alpha]
        # backward_diff = (rho_sq - circshift(rho_sq, odiff)) / model.delta[alpha]
        # forward_diff = (circshift(rho_sq, -odiff) - rho_sq) / model.delta[alpha]
        # model.drho[idx_rng] .+= vec(-model.params["alpha"] * backward_diff .* max.(velocity_sq, 0))
        # model.drho[idx_rng] .+= vec(-model.params["alpha"] * forward_diff .* min.(velocity_sq, 0))
        
        model.drho[idx_rng] += model.params["alpha"] * model.rho[idx_rng]
        
    end

    # Check if any value is less than a threshold
    # Get the index
    idx = findall(x->x<-10, model.drho[idx_rng])
    if length(idx) > 0 && false
        println("Negative value found in drho")
        println("Index: ", idx)
        println("Value: ", model.drho[idx_rng][idx])

        println("rho: ", model.rho[idx])
        println("grad rho fd: ", model.grad_rho_fd[th_num,:][idx])
        println("my position: ", positions[alpha][idx_rng][idx])
        println("center of mass: ", model.center_of_mass[alpha])

        # exit(0)
    end


end

function update_potential_force!(model::NumericalMeanField2D)
    model.intermediate["potential_force"][1] = sum(model.rho .* model.potential_grad_x) / sum(model.rho)
    model.intermediate["potential_force"][2] = sum(model.rho .* model.potential_grad_y) / sum(model.rho)

end

function update_parallel!(model::NumericalMeanField2D)

    update_center_of_mass!(model)
    update_potential_force!(model)


    # TODO: change back to multi thread
    # Threads.@threads for th_num in 1:model.num_th
        update_drho_parallel!(model,model.pts_per_th,1)
    # end

    # This doesn't seem to be necessary
    # model.drho[model.rho .< 0] .= 0

    # Clip small rhos
    # model.drho[abs.(model.rho) .< 1e-5] .= 0

    # println("Step ", model.step_counter, ": rho at index $(debug_idx) is ", model.rho[debug_idx], 
    # " \t the grad is ", model.drho[debug_idx])

    # Check peripheral values
    function get_max_abs_value(arr)
        return maximum(abs.(arr))
    end
    rho = reshape(model.rho, model.npts...)
    max_val = get_max_abs_value(rho[1:2, :])
    max_val = max(max_val, get_max_abs_value(rho[end-1:end, :]))
    max_val = max(max_val, get_max_abs_value(rho[:, 1:2]))
    max_val = max(max_val, get_max_abs_value(rho[:, end-1:end]))
    if max_val > 1e-5
        println("Step ", model.step_counter, ": max value is ", max_val)
    end
    # println("Set rho to zero")
    # rho[1:2, :] .= 0
    # rho[end-1:end, :] .= 0
    # rho[:, 1:2] .= 0
    # rho[:, end-1:end] .= 0
    # model.rho = vec(rho)

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
        println("Still in step ", model.step_counter, ", the rho is ", model.rho[debug_idx])
        Threads.@threads for th_num in 1:model.num_th
            update_drho_parallel!(model,model.pts_per_th,th_num)
        end
        
        println("Still in step ", model.step_counter, ", the grad_rho_fd is ", model.grad_rho_fd[1,:][debug_idx])
        println("Still in step ", model.step_counter, ", the grad is ", model.drho[debug_idx])
        # Clip small drho
        # model.drho[abs.(model.rho) .< 1e-5] .= 0
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
        @. model.rho += model.drho .* model.dt

        #increase elapsed time by dt
        model.t += model.dt

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
                     "t"=>model.t,"step_num"=>model.step_counter)
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
# Try not to use indexing on sparse matrices since 
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
