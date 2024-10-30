using SparseArrays
using JLD2
using FileIO
function mod_idx(idx,n)
    while idx < 1
        idx += n
    end
    while idx > n
        idx -= n
    end
    return idx
end

function diff_mat2d(nx,ny,along,mode ="forward")
    """
    return the first order difference matrix (d/dx or d/dy)
    nx,ny : the number of points in each dimension
    along : derivative along x-axis(along=1) or y-axis(along=2)
    model : "central" for central difference (line stencil)
            "foward" for foward difference (line stencil)
            "backward" for backward difference (line stencil)

    return: the difference matrix
    how the 2d lattice is flatted
              x=1 x=2  x=3  x=4  x=5 (row)
       y=1:    1    2    3    4    5
       y=2:    5    6    7    8    9
     (column)
    Dx*rho ~ drho/dx, where Dx is the difference matrix
    *note that the marix if for Dirchlet boundary condition
    where \rho(boundary) = 0,here we do not calculate the gradient for \rho(boundary)
    because our equation cotains flux = \rho*j, there is no dynamics at the boundary
    *boundary points are rho(i=1,j=1...ny),rho(i=nx,j=1...ny),rho(i=1...nx,j=1),rho(i=1...nx,j=ny)
    """
        
    N = nx*ny
    if mode == "central"
        diff_coeff = [-0.5  0.0  0.5]

    elseif mode == "forward"
        diff_coeff = [0.0 -1.0  1.0]
    else
        error("model = central or forward, no other schemes implemented")
    end
    nrow,ncol = size(diff_coeff)
    nbrs = ncol ÷ 2         # integer division
    diff_mat = spzeros(Float64, N, N)
    line_stencil_ind =  Array{Int64,1}(undef,ncol)
    
    for i in 2:nx-1
        for j in 2:ny-1   
            if along == 1 
                stencil_ind[2] = i + (j-1)*nx     # center
                stencil_ind[1] = i - 1 + (j-1)*nx # left one
                stencil_ind[3] = i + 1 + (j-1)*nx # right one
            elseif along == 2
                stencil_ind[2] = i + (j-1)*nx   # center
                stencil_ind[1] = i + (j-2)*nx   # upper one
                stencil_ind[3] = i + j*nx       # lower one
            else
                error("along = 1 or 2, no other values allowed")
            end
            for is in 1:3
                diff_mat[stencil_ind[2],stencil_ind[is]] = cdiff_coeff[odiff,is]
            end
        end
    dropzeros!(diff_mat)
    return sparse(diff_mat)
end


function cut_into_blocks(M::SparseMatrixCSC{Float64, Int64},n)
    """
        cut the Nx-by-Ny sparse matrix,M, into n matrices, B1,B2 ... Bn,
        where B1 = M[1:s,:] B2 = M[s+1:2s,:],Bi = M[(i-1)*s+1:i*s,:]
        s = div(Nx,n)
    """
    Nx,Ny = size(M)
    s = div(Nx,n)
    diff_mat_list = Array{SparseMatrixCSC{Float64, Int64},1}(undef,n) 
    for i in 1:n 
        diff_mat_list[i] = M[(i-1)*s+1:i*s,:]
    end
    return diff_mat_list
end

function simpson_int1d(f::Array{Float64,1},rng)
"""
    1d numerical integration using Simpson's rule
    f is the integrand of N+1 points
    rng contains the limits, a and b, of the integration
    f[i] is the value of f(a + (i-1)*h), i = 1...N+1 
    where h = (b-a)/N 
"""
    if ndims(f) > 1
        error("ndims(f) must be 1 for 1d numerical integration")
    end
    N = size(f)[1] - 1
    h = (rng[2] - rng[1])/N
    s = f[1] + f[N+1]
    for j in 2:2:N
        s += 4.0*f[j]
    end
    for j in 3:2:N
        s += 2.0*f[j]
    end
    return s*h/3.0
end

function simpson_int2d(f::Array{Float64,2},rng_x,rng_y)
"""
    2d numerical integration using Simpson's rule
    f     :the integrand of (Nx+1)*(Ny+1) points
    rng_x :the limit for x direction, xi and xf
    rng_y :the limit for y direction, yi and yf
    f[i,j] is the value of f(xi + (i-1)*hx, yi + (j-1)*hy)
           , i = 1...Nx+1 j = 1...Ny+1
    where hx = (xf-xi)/Nx and hy = (yf-yi)/Ny 
"""
    Nx,Ny = size(f) .- 1
    # the auxiliary varible fy = int dx f(x,y)
    fy = zeros(Float64,Ny + 1)
    for y in 1:Ny+1
        fy[y] = simpson_int1d(f[:,y],rng_x)
    end
    return simpson_int1d(fy,rng_y)
end
