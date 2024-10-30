using JSON
using FFTW
mutable struct DataLoader
    data_root_dir::String
    info::Dict{String,Any}
    counter::Integer
end

function DataLoader(root_dir::String)
    info = read_input(joinpath(root_dir,"input.json"))
    counter = 0
    DataLoader(root_dir,info,counter)
end

function is_frame_existed(dl::DataLoader,frame_num)
    json_file_path = joinpath(dl.data_root_dir, "Frame_$frame_num.json")
    zip_file_path = joinpath(dl.data_root_dir, "Frame_$frame_num.zip")
    if isfile(json_file_path) || isfile(zip_file_path)
        return true
    end
    return false
end
function is_eod(dl::DataLoader)
    """
    check if reaching the last existed frame
    """
    if is_frame_existed(dl,dl.counter)
        return false
    elseif dl.counter < dl.info["n_steps"] 
        println("reaches EOD, there are frames not calculated")
    end
    return true
end

function remove_old_frame()
    files = filter(f -> endswith(f, ".zip"), readdir("."))
    matching_files = filter(f -> occursin(r"^Frame.*", f), files)
    for f in matching_files
        run(`rm $f`)
    end
end

function reset_dl(dl::DataLoader)
    dl.counter = 0
end

function load(dl::DataLoader)
    """
    load and return the current frame
    """
    data = read_data_frame(dl,dl.counter)
    dl.counter += dl.info["n_save"]
    return (dl.counter - dl.info["n_save"],data)
end

function read_data_frame(dl::DataLoader,frame_num)
    json_file_path = joinpath(dl.data_root_dir, "Frame_$frame_num.json")
    zip_file_path = joinpath(dl.data_root_dir, "Frame_$frame_num.zip")
    if dl.info["compression"] > 0
        if isfile(json_file_path)
            rm_json_cmd = ["rm",json_file_path]
            run(`$rm_json_cmd`)
        end
        unzip_cmd = ["unzip",zip_file_path,"-d",dl.data_root_dir]
        run(`$unzip_cmd`)
    end
    data = JSON.parsefile(json_file_path)
    if dl.info["compression"] > 0
        rm_json_cmd = ["rm",json_file_path]
        run(`$rm_json_cmd`)
    end
    return data
end

function read_input(file_str)
    info = JSON.parsefile(file_str)
    return info
end

####### ------- data analysis ------- #####
function get_structure_factor(rho,nx,ny,Lx,Ly)
    if ndims(rho) == 1
        rho = reshape(rho,(nx,ny))
    end
    v = sum(rho)
    # remove the mean value 
    rho = (rho .- v/(nx*ny))
    f_rho = fftshift(fft(rho))
    # normalised structure factor
    s_rho = real(f_rho.*conj(f_rho))/v
    # original frequence 1..N+1 is mapped to 0 to 2pi
    # only N points are outputed since fft[1] and fft[N+1] are the same
    kx = fftshift(fftfreq(nx,nx))*2.0*pi/Lx
    ky = fftshift(fftfreq(ny,ny))*2.0*pi/Ly
    return (kx,ky,s_rho)
end

function get_radial_structure_factor(kx,ky,s_rho,n_bins=0,cut_off = 0.707)
    nx = length(kx)
    ny = length(ky)
    k_max = sqrt(kx[end]^2 + ky[end]^2)*cut_off
    if n_bins == 0
        n_bins = trunc(Int,sqrt(nx*ny)*0.5)
    end
    n_bins = trunc(Int,n_bins*cut_off + 0.5)
    delta_k = k_max/n_bins
    dA = (kx[2]-kx[1])*(ky[2]-ky[1])
    s_k = zeros(Float64,n_bins)
    bin_count = zeros(Float64,n_bins)
    for i in 1:nx
        for j in 1:ny
            k = sqrt(kx[i]*kx[i] + ky[j]*ky[j])
            bin_num = trunc(Int,k/delta_k) + 1
            if bin_num <= n_bins
                s_k[bin_num] += s_rho[i,j]*dA
                bin_count[bin_num] += 1.0
            end
        end
    end
    for b in 1:n_bins
        k = delta_k*0.5 + delta_k*(b-1)
        area = 2.0*pi*k*delta_k
        s_k[b] /= area
    end
    k = collect(0.5*delta_k:delta_k:k_max)
    return (k,s_k)
end

function get_radial_cross_corr(rho,nx,ny,Lx,Ly,n_bins = 0)
    if ndims(rho) == 1
        rho = reshape(rho,(nx,ny))
    end
    rho = rho .- sum(rho)/(nx*ny) # remove the average value
    f_rho = fft(rho)
    corr = real(ifft(f_rho.*conj(f_rho)))
    r_max = 0.5*sqrt(Lx*Ly) # due to the PBC, maximum corr-length = L/2
    if n_bins == 0 
        n_bins = trunc(Int,sqrt(nx*ny)*0.5)
    end
    radial_corr = zeros(Float64,n_bins)
    bin_count = zeros(Float64,n_bins)
    dx = Lx/nx
    dy = Ly/ny
    dr = r_max/n_bins
    for ix in 1:nx
        x = (ix-1)*dx
        for iy in 1:ny
            y = (iy-1)*dy
            r = sqrt(x^2 + y^2)
            bin_num = trunc(Int,r/dr) + 1
            if bin_num <= n_bins
                radial_corr[bin_num] += corr[ix,iy]
                bin_count[bin_num] += 1.0
            end
        end
    end
    for b in 1:n_bins
        r = dr*0.5 + dr*(b-1)
        area = 2.0*pi*r*dr
        radial_corr[b] /= area
    end
    r = collect(0.5*dr:dr:r_max)
    #nomarlisation
    radial_corr /= radial_corr[1]
    return (r,radial_corr)
end




