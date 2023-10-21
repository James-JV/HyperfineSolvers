"""
Required functions for 'HyperfineSolvers' Module.

Author: James Vandeleur
Contact: jamesvandeleur@proton.me
UQ Physics Honours Project 2023
"""

using QuadGK
include("PhysicalConstants.jl")

"""
    (d::Integer, a::Vector{Integer}) = adams_coeff(k)
Returns the '`k+1`'-order Adams-Moulton coefficients and their integer divisor `d`.

**Arguments:**

- `k::Integer`: order parameter for Adams-Moulton.

**Returns:**

- `d::Integer`: integer divisor

- `a::Vector{Int64}`: - vector of `k+1` integer coefficients.

# Examples
```jl-repl
julia> adams_coeff(3)
(24, [1, -5, 19, 9])
```
"""
function adams_coeff(k)
    
    a = zeros(k+1) # initialise array

    # General expression for generating coefficients
    b(s,j) = quadgk(u -> prod(u .+ (0:s) .- 1)/(u+j-1),0,1)[1] * (-1)^j/(factorial(big(j))*factorial(big(s-j)))
    # corretly ordered
    for J = eachindex(a)
        j = J - 1
        a[J] = b(k, k - j)
    end

    # prepare output
    a = rationalize.(BigInt, a)
    d = Integer(1/gcd(a))
    a = Integer.(d.*a)

    return (d,a)
end

"""
    (P,Q) = adams(t, P, Q, G, k=3)
Uses the Adams-Moulton method to solve the Dirac equation to find the upper and lower radial components of the wavefunction `[ P(r(t)) ; Q(r(t)) ]`, over the uniform grid `t = (1:Ngrid)*h`.

**Arguments:**

- `t::StepRangeLen`: array containing uniform grid.

- `P_init::Array, Q_init::Array`: arrays to store upper and lower wave function components respectivley. Must contain initial `k` values where `k` is the order parameter of the method.

- `G::Tuple`: 4-tuple that contains arrays A,B,C,D which are components from the coupling matrix of the differential equations.

- `k::Integer` - AM order parameter of method.

**Returns:**

- `(P,Q)::Tuple{Array, Array}`: tuple containing arrays for each component of solution wavefunction. 
"""
function adams(t, P, Q, G, k=3)

    # calculate coefficients
    (d,a) = adams_coeff(k)

    # collect parameters from uniform position grid.
    Ngrid = length(t)
    Δt = t[2]-t[1]

    # separate matric components
    A = G[1]
    B = G[2]
    C = G[3]
    D = G[4]
  
    # carry out integration step for each element.
    for i = k:(Ngrid-1)
        p = P[i] + sum( j -> a[j]*(A[i-k+j]*P[i-k+j] + B[i-k+j]*Q[i-k+j]), 1:k)*Δt/d
        q = Q[i] + sum( j -> a[j]*(C[i-k+j]*P[i-k+j] + D[i-k+j]*Q[i-k+j]), 1:k)*Δt/d
        
        λ = Δt*a[k+1]/d
        Δ = 1 - λ^2*(B[i+1]*C[i+1] - A[i+1]*D[i+1])
        
        P[i+1] = ((1-λ*D[i+1])*p + λ*B[i+1]*q) / Δ
        Q[i+1] = (λ*C[i+1]*p + (1-λ*A[i+1])*q) / Δ
    end

    # return solution components.
    return (P,Q)
end

"""
    (P,Q) = adams_init(t, P, Q, G::Function, K=3, ki=1)
Calculates the first `k` values from `ki` to `K` using incrementally higher order AM methods.

**Arguments:**

- `t::StepRangeLen`: array containing uniform grid.

- `P_init::Array, Q_init::Array`: arrays to store upper and lower wave function components respectivley. Must contain initial `k` values where `k` is the order parameter of the method.

- `G::Tuple`: 4-tuple that contains arrays A,B,C,D which are components from the coupling matrix of the differential equations.

- `k::Integer` - AM order parameter of method.

**Returns:**

- `(P,Q)::Tuple{Array, Array}`: tuple containing arrays for each component of solution wavefunction with initial `k` values populated. 
"""
function adams_init(t, P, Q, G, K=3, ki=1)
    k = ki

    while (k < K) 
        (P,Q) = adams(t[1:(k+1)],P,Q,G,k)
        k += 1
    end

    return (P,Q)
end

"""
    n = nodes(f)
Returns the number of nodes, `n` in an array `f`, by counting the number of times f(x) crosses the x axis.
Excludes tail end of the domain (where function close to 0).

**Arguments:**

- `f::Vector`: discrete representation of the radial component of some wavefunction. 

**Returns:**

- `n::Integer`: number of nodes in wavefunction.
"""
function nodes(f)
    index = findlast(abs.(f) .> maximum(abs.(f))/20)
        
    n = 0
    for i = 2:index#(Int(round(length(f)*0.9)))
        if (sign(f[i]) != sign(f[i-1]))
            n += 1
        end
    end
    return n
end

"""
    dy = ddr(x,y)
Central difference numeric derivative.

**Arguments:**

- `x::Vector`: independent variable vector.

- `y::Vector`: depednent variable vector.

**Returns:**

- `dy::Vector`: derivative of `y`. 
"""
function ddr(x,y)

    dy = zeros(length(y))
    
    dy[1] = y[2] - y[1]
    dy[end] = y[end] - y[end-1]

    for i in 2:(length(y)-1)
        sum = (y[i+1] - y[i])/(x[i+1] - x[i]) + (y[i] - y[i-1])/(x[i] - x[i-1])
        dy[i] = sum/2
    end

    return dy
end

"""
    val = trapz(x,y)
Trapezoidal rule integration of y(x).

**Arguments:**

- `x::Vector`: independent variable vector.

- `y::Vector`: depednent variable vector.

**Returns:**

- `val::Float`: value of integral. 
"""
function trapz(x,y)
    val = 0
    val += y[1]*(x[2]-x[1])/2
    val += y[end]*(x[end]-x[end-1])/2
    for i = 2:(length(y)-1)
        x1 = (x[i-1] + x[i])/2
        x2 = (x[i+1] + x[i])/2
        val += y[i]*(x2-x1)
    end
    return val
end

"""
    val = trapz(x,y)
Simpson's rule integration y(x). Odd length of vectors allowed.

**Arguments:**

- `x::Vector`: independent variable vector.

- `y::Vector`: depednent variable vector.

**Returns:**

- `val::Float`: value of integral. 
"""
function simpson(x,y)
    if length(x) != length(y)
        error("Tried to integrate arrays of different length in simpson().")
    end

    h = diff(x)
    N = length(h)
    if isodd(N)
        # for odd N start with correction of last interval
        total = (x[N+1]-x[N])*(y[N+1] + y[N])/2
        N = N-1 # continue computing up to the second last interval
    else
        # else define an empty variable as normal
        total = 0
    end

    for i = 0:Integer(N/2-1)
        h0 = x[2*i+2] - x[2*i+1]
        h1 = x[2*i+3] - x[2*i+2]

        f0 = y[2*i+1]
        f1 = y[2*i+2]
        f2 = y[2*i+3]

        total += (h0 + h1)/6 * ((2 - h1/h0)*f0 + (h0+h1)^2/(h0*h1) * f1 + (2-h0/h1)*f2)
    end

    return total
end

"""
    c = bisect(f, a, b; tol=1e-10, Nmax=100, printsteps=false)
Root finding with bisection method. Finds roots of function `f(x)` with initial interval `[a,b]`.

**Arguments:**

- `f::Function`: function to find roots of.

- `a::Float, b::Float`: lower and upper bound of initial interval.

- `tol::Float`: error tolerence for finishing condition.

- `Nmax::Integer`: Maximum number of steps.

- `printsteps::Bool`: flag to print bisection progress to terminal. Useful for debugging.

**Returns:**

- `c::Float`: value of root. 
"""
function bisect(f::Function,a,b;tol=1e-10,Nmax=100,printsteps=false)

    c = (a+b)/2
    fa = f(a)
    fb = f(b)
    fc = f(c)
    if sign(fa) == sign(fb)
        println("a = ",a,", f(a) = ",fa)
        println("b = ",b,", f(b) = ",fb)
        error("Error in bisection method: sign(f(a)) == sign(f(b)). Either no or multiple roots in given range [a,b].")
    end

    N = 0
    print("carrying out bisection.")
    while abs(fc) > tol

        c = (a+b)/2
        fc = f(c)

        if printsteps
            println("a = ",a,", f(a) = ",fa)
            println("c = ",c,", f(c) = ",fc)
            println("b = ",b,", f(b) = ",fb)
            println(". . .")
        else
            print(".")
        end

        if sign(fc) == sign(fa)
            a = c
            fa = fc
        else
            b = c
            fb = fc
        end

        N += 1
        if N >= Nmax
            error("Bisection method did not converge")
            break
        end

    end

    println("end")
    return c
end

"""
    (W, P, Q, dP, dQ) = solveDirac(t, r_array, drdt_array, Z, n, l, j, W_i, ε, V_array, m, k_order=3)
Solves the Dirac equation for a given potential to find the upper and lower components of the radial wavefunction of a lepton bound to a nucleus.

**Arguments**

- `t::StepRangeLen`: array containing uniform grid.

- `r_array::Array, drdtr_array::Array`: arrays containing the physical radial distance corresponding to the grid points `t` and its derivative respectively.

- `Z::Integer`: atomic number of nucleus.

- `n::Integer`: principal quantum number of orbital.

- `l::Integer`:  quantum number of the orbitals's orbital angular momentum.

- `j::Integer`:  quantum number of the orbitals's total angular momentum.

- `W_i::Float`: initial guess for the energy of the orbital in atomic units (Eh).

- `ε::float`: tolerance for matching inner and outer integration steps.

- `V_array:Array`: Array of values of the potential along the grid `r(t)`.

- `m:Float`: mass of orbital in atomic units (electron mass = 1).

- `k-order`: order parameter of AM method to be used.

**Returns:**

- `W:Float`: energy of orbital state.

- `P:Array, Q:Array`: upper and lower radial components of the wave funtion over the grid `r(t)`.

- `dP:Array, dQ:Array`: first derivatives of upper and lower radial components of the wave funtion over the grid `r(t)`. Useful for verifying physicality of result.

"""
function solveDirac(t, r_array, drdt_array, Z, n, l, j, W_i, ε, V_array, m, k_order=3)

    Ngrid = length(r_array)

    P = zeros(Ngrid)
    Q = zeros(Ngrid)

    # calculate relativistic quantum number
    κ = (-1)^(j+l+1/2)*(j+1/2)

    # Parameters for G(t) matrix, defined from the Dirac equation.
    A = -κ .* drdt_array ./ r_array
    B(W) = -α .* drdt_array .* (W .- V_array .+ 2*m*c^2)
    C(W) = α .* drdt_array .* (W .- V_array)
    D = -A
    G(W) = [A, B(W), C(W), D]
    G_rev(W) = reverse.(G(W))

    # 1. Guess the relativistic energy.
    W = W_i
    
    # inital loop to get number of nodes correct
    nr = n - l - 1 # expected number of nodes
    while true
        # 2. Integrate outward to classical turning point a_c.
        tp_index = findlast((V_array .+ l*(l+1)./(2*r_array.^2)) .<= W)
        print("tp_index = ", tp_index, " ")
        # tp_index may be right at the end if Ngrid is too small
        if tp_index == Ngrid
            error("tp_index is at the end of the grid. Make grid larger or energy lower.")
        end

        # outwards integration
        γ = sqrt(1 - (α*Z)^2)
        P[1] = r_array[1]^γ
        Q[1] = ((κ>=0)*(-κ-γ)/(α*Z) + (κ<0)*α*Z/(γ-κ)) * r_array[1]^γ

        (P,Q) = adams_init(t, P, Q, G(W), k_order) # No need to chane adams functions
        (P,Q) = adams(t[1:tp_index], P, Q, G(W), k_order)
        tp_val_out = big(P[tp_index])
        Q⁻ = big(Q[tp_index])

        # 3. Integrate inwards to turning point a_c.
        PR = reverse(P)
        QR = reverse(Q)
        PR[1] = (-1)^nr*exp(-r_array[end])
        QR[1] = (-1)^nr*exp(-r_array[end])


        (PR,QR) = adams_init(t, PR, QR, -G_rev(W), k_order)
        (PR,QR) = adams(t[1:(Ngrid-tp_index+2)], PR, QR, -G_rev(W), k_order)
        P = reverse(PR)
        Q = reverse(QR)
        tp_val_in = big(P[tp_index])
        Q⁺ = big(Q[tp_index])
        
        # 4. multiplicative factor to scale inwards to meet outwards.
        P[1:(tp_index-2)] = P[1:(tp_index-2)]./tp_val_out
        Q[1:(tp_index-2)] = Q[1:(tp_index-2)]./tp_val_out

        P[(tp_index-1):length(P)] = P[(tp_index-1):length(P)]./tp_val_in
        Q[(tp_index-1):length(Q)] = Q[(tp_index-1):length(Q)]./tp_val_in
        
        Q⁻ = Q⁻/tp_val_out
        Q⁺ = Q⁺/tp_val_in
        
        # test condition for solution
        n_nodes = nodes(P)
        diff = Q⁺ - Q⁻
        if n_nodes > nr
            W = 1.1*W
            println("# of nodes = ", n_nodes, " != ", nr)
            println("new W = ", W)
        elseif n_nodes < nr
            W = 0.95*W
            println("# of nodes = ", n_nodes, " != ", nr)
            println("new W = ", W)
        else
            if abs(2*(diff)/(Q⁺+Q⁻)) <= ε
                println("about to break")
                println("tol = ", diff)
                break
            else
                norm = simpson(r_array, P.^2 .+ Q.^2)
                print("Wi = ", W)
                W = W + c*diff*P[tp_index]/norm # Johnson equation 2.190
                println(" Wf = ", W)
                println("rel tol = ", abs(2*(diff)/(Q⁺+Q⁻)))
            end
        end
    end

    dP = ddr(r_array,P)
    dQ = ddr(r_array,Q)

    ## Normalisation
    N = simpson(r_array, (P.^2 .+ Q.^2))^(-1/2)

    P_final = N*P
    Q_final = N*Q
    dP_final = N*dP
    dQ_final = N*dQ

    return (W,P_final,Q_final,dP_final,dQ_final)
end