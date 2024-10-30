include("./src/utils.jl")
using Plots
using Printf

# this line sets the headless mode for julia plotting
# since there is no graphic interface in the cluster
# seed https://discourse.julialang.org/t/plotting-from-a-server/74345
ENV["GKSwstype"] = "100"

if length(ARGS) < 1 
    error("Please specify the data directory")
end
data_dir = ARGS[1] 
dl = DataLoader(data_dir)
x_max,y_max = dl.info["range"][1],dl.info["range"][2]
delta_x = dl.info["range"][1] / dl.info["npts"][1]
delta_y = dl.info["range"][2] / dl.info["npts"][2]
nx, ny = dl.info["npts"][1],dl.info["npts"][2]
x = range(delta_x, x_max;length = nx)
y = range(delta_y, y_max;length = ny)
anim = @animate while !is_eod(dl)
    n_step,data = load(dl)
    t = @sprintf("%4.3f",n_step*dl.info["dt"])
    rho = reshape(Array{Float64}(data["rho"]),(nx,ny))
    # plot density
    p1 = heatmap(x,y,rho',aspect_ratio=:equal,tellheight=true,interpolate=true,title="t = $t")
    xlims!(p1,0,x_max)
    ylims!(p1,0,y_max)
    kx,ky,s_2d = get_structure_factor(rho,nx,ny,x_max,y_max)
    # plot the struture factor
    p2 = heatmap(kx,ky,log10.(s_2d)',aspect_ratio=:equal,tellheight=true,interpolate=true,title="log10 S(kx,ky)")
    xlims!(p2,kx[1],kx[end])
    ylims!(p2,kx[1],ky[end])

    #plot the radial structure factor
    k,s = get_radial_structure_factor(kx,ky,s_2d)
    p3 = plot(k,s,xscale=:log10, yscale=:log10,title="S(k)")

    #plot the radial correlation function
    r,corr = get_radial_cross_corr(rho,nx,ny,x_max,y_max)
    p4 = plot(r,corr,title="Corr(r)")
    
    p = plot(p1,p2,p3,p4,layout=(2,2))

end
mp4(anim, joinpath(data_dir,"all.mp4")) 

