"""
Codes for numerical computation of nuclear orbital wavefunctions and various hyperfine interactions.
Particular focus on the fitting of a nuclear radius in one system (muonic) for calculation of BW effect in another (electronic).

The core method for solving the Dirac equation is described in:
    " W. R. Johnson, Atomic Structure Theory. Springer Berlin Heidelberg, 2007. doi: 10.1007/978-3-540-68013-0 ".

As the focus of my thesis was on the physics of the hyperfine interaction and not the development of this code,
one should view this code as sufficient and functional. It is not necessarily optimised although I have tried to maintain
best practice as much as I know how.

The module requires the following files:
    - "DiracAM.jl": contains foundational methods for solving the Dirac equation using Adams-Moulton method.
        - "PhysicalConstants.jl": file containing gloabl physical constants,
                                  such as the fine structure constant α (required by "DiracAM.jl")

For assistance in using this code beyond what is given in the internal documentation and/or the Johnson lectures, I refer the reader to my honours thesis,
or feel free to contact me with questions. Information is provided below.

Author: James Vandeleur
Contact: jamesvandeleur@proton.me
Physics Honours Project 2023
University of Queensland (UQ), Australia
"""

using DelimitedFiles, Plots, QuadGK
include("DiracAM.jl")


struct Nucleus
    # Inputs
    Z::Int
    L::Int
    I::Float16
    gₗ::Int
    ρ::Function
    rrms::Float64
    μ::Float16
    # Internal
    V::Function # nuclear potential
    U::Function # Uehling potential
    F::Function # magnetic distribution

    """
    Nucleus(Z, L, I, gₗ, ρ_bare, rrms, μ)
    Creates a nucleus struct which contains all the nuclear information required in hyperfine calculations.
    In creating the `Nucleus` object also calculates the nuclear potential function (including Uehling potential) and the form of the magnetisation distribution F(r).

    **Arguments:**
    - `Z::Integer`: atomic number.
        
    - `L::Integer, I::Integer`: odd nucleon orbitial (`L`) and total (`I`) angular momentum quantum numbers.

    - `gₗ::Integer`: odd nucleon orbital momentum g-factor. `gₗ = 0` for neutron and `gₗ = 1` for proton.

    - `ρ_bare::Function`: charge distribution function `ρ₀(r,rrms)` where `r` is the radial coordinate and `rrms` is the root-mean-square charge radius.
    This form allows for breaking integrals up over the step function at `rrms`.
    The constructor will normalise this and reduce it to simply `ρ(r)`.
        
    - `rrms::Float`: root-mean-square nuclear charge radius in atomic units (a.u.).

    - `μ::Float`: nuclear magnetic moment in units of the Bohr magneton (μₙ).
    """
    function Nucleus(Z, L, I, gₗ, ρ_bare, rrms, μ)
        ## Define Potential ##
        # normalise Charge
        ρ_0 = quadgk(r -> 4*π*r^2*ρ_bare(r,rrms), 0,Inf)[1]^(-1) # 4π∫dr r^2 ρ(r) = 1
        ρ(r) = ρ_0*ρ_bare(r,rrms)
        # make potential
        I1(r) = (4*π/r)*quadgk(k -> k^2*ρ(k),0,r)[1]
        I2(r) = 4*π*quadgk(k -> k*ρ(k),r,Inf)[1]
        V(r) = -Z*(I1(r) + I2(r))

        # Uehling (with step charge distribution)
        U1(r,rₙ) = (Z*α)/(π*r) * quadgk(t -> sqrt(t^2-1)*(1/t^2 + 1/(2*t^4))*(2/(2*t*rₙ/α)^3) * ((r/rₙ)*2*t*rₙ/α - exp(-2*t*rₙ/α)*(1 + 2*t*rₙ/α)*sinh(2*t*r/α)), 1, 10000)[1]
        U2(r,rₙ) = (Z*α)/(π*r) * quadgk(t -> sqrt(t^2-1)*(1/t^2 + 1/(2*t^4))*(2/(2*t*rₙ/α)^3)*exp(-2*t*r/α)*((2*t*rₙ/α)*cosh(2*t*rₙ/α) - sinh(2*t*rₙ/α)), 1, 10000)[1]   
        function U(r)
            rₙ = sqrt(5/3)*rrms
            if r <= rₙ
                return U1(r,rₙ)
            elseif r > rₙ
                return U2(r,rₙ)
            end
        end  

        ## Define S.P. Mag distribution ##
        if I == (L + 1/2)
            gₛ = 2*μ - (2*I - 1)*gₗ
            A = gₛ/2 + (I - 1/2)*gₗ
            B = -gₛ*(2*I-1)/(8*(I+1)) + (I - 1/2)*gₗ
        elseif I == (L - 1/2)
            gₛ = -2*(I+1)*μ/I + (2*I+3)*gₗ
            A = -gₛ*I/(2*(I+1)) + gₗ*I*(2*I + 3)/(2*(I+1))
            B = gₛ*(2*I+3)/(8*(I+1)) + gₗ*I*(2*I + 3)/(2*(I+1))
        end
        F(r,u::Function,Rₘ) = (1/μ) * (A*quadgk(k -> k^2*u(k,Rₘ)^2, 0, (Rₘ<r)*Rₘ, r)[1] + B*quadgk(k -> k^2*u(k,Rₘ)^2*(r/k)^3, r, r + (r<Rₘ)*(Rₘ - r), Inf)[1]) 

        return new(Z, L, I, gₗ, ρ, rrms, μ, V, U, F)
    end
end

struct Orbital
    # Inputs
    nucleus::Nucleus
    m::Float16
    n::Int
    l::Int
    j::Float16
    # Internal
    r::Array
    f::Array
    g::Array
    df::Array
    dg::Array
    energy::Float64

    f_point::Array
    g_point::Array
    df_point::Array
    dg_point::Array
    energy_point::Float64

    f_Ul::Array
    g_Ul::Array
    df_Ul::Array
    dg_Ul::Array
    energy_Ul::Float64

    """
    Orbital(nuc, m, n, l, j; W_i=-3740, Ngrid=1000, h=0.02, r_func=(t->5e-9*(exp.(t) .- 1)), drdt_func=(t->5e-9*exp.(t)), AM_order=10, AM_tol=1e-16, use_save=true)
    Creates an orbital struct that solves the dirac equation to find the radial wavefunctions of the specified state.
    Can save the calculated results to disk such that if the code is run at a later data with exactly matching parameters, the solver does not need to run again.
    This speeds up the process of repeated calculations.

    The solver can be sensitive to solver parameters (W_i, Ngrid and h). The default values work well for most muonic atoms with a single muon in the ground state.

    **Arguments:**
    - `nuc::Nucleus`: the nucleus of the system to study.

    - `m::Float`: mass of orbital in atomic mass units. Set to `m=1` for an electron, or `m=mmu` using "PhysicalConstants.jl" for a muon.
        
    - `n::Integer, l::Integer, j::Integer`: quantum numbers specifying state of orbital.

    - `W_i::Float`: Initial guess for the energy of the orbital.

    - `Ngrid::Integer`: Size of grid to solve over.

    - `h::Float`: Set size of grid to solve over.

    - `r_func::Function`: map from uniform grid to radial distance `r(t)`. This allows greater precision in the nuclear region.

    - `drdt_func::Function`: map from uniform grid to derivative of radial distance `dr(t)/dt`.

    - `AM_order::Integer`: order for AM solution to Dirach equation.

    - `AM_tol::Float`: tolerance for matching inward and outward integrations in AM method.

    - `use_save::Bool`: flag to force a new calculation of the solution when `use_save=false`.
    """
    function Orbital(nuc::Nucleus,m,n,l,j;W_i=-3740,Ngrid=1000,h=0.02,r_func=(t->5e-9*(exp.(t) .- 1)),drdt_func=(t->5e-9*exp.(t)),AM_order=10,AM_tol=1e-16,use_save=true)

        ## Setup position grid.
        t = (1:Ngrid)*h
        r_array = r_func.(t)
        drdt_array = drdt_func.(t)

        # Solve for wavefunctions
        nuc_info = "nucleus(Z"*string(nuc.Z)*"_L"*string(nuc.L)*"_I"*string(nuc.I)*"_g"*string(nuc.gₗ)*"_rrms"*string(nuc.rrms)*"_mu"*string(nuc.μ)*"_charge_samples"*string(nuc.ρ(0.9*nuc.rrms))*":"*string(nuc.ρ(1.1*nuc.rrms))*")"
        orb_info = "orbital(m"*string(m)*"_n"*string(n)*"_l"*string(l)*"_j"*string(j)*")"
        file_path = "wavefunction_data/"*orb_info*nuc_info*".csv"
        if isfile(file_path) && use_save # file exists and use_save=true is selected
            dat = readdlm(file_path,',')
            (W,f,g,df,dg) = (dat[1],dat[2,:],dat[3,:],dat[4,:],dat[5,:])
            (W_point,f_point,g_point,df_point,dg_point) = (dat[6],dat[7,:],dat[8,:],dat[9,:],dat[10,:])
            (W_Ul,f_Ul,g_Ul,df_Ul,dg_Ul) = (dat[11],dat[12,:],dat[13,:],dat[14,:],dat[15,:])

            println("Accquired wavefunctions with energy ", W.*Eh_to_eV,"eV.")
        else
            (W,f,g,df,dg) = solveDirac(t, r_array, drdt_array, nuc.Z, n, l, j, W_i, AM_tol, nuc.V.(r_array), m, AM_order)
            V_point(r) = -nuc.Z/r
            (W_point,f_point,g_point,df_point,dg_point) = solveDirac(t, r_array, drdt_array, nuc.Z, n, l, j, W_i, 1e-15, V_point.(r_array), m, AM_order)
            (W_Ul,f_Ul,g_Ul,df_Ul,dg_Ul) = solveDirac(t, r_array, drdt_array, nuc.Z, n, l, j, W, AM_tol, nuc.V.(r_array) .+ nuc.U.(r_array), m, AM_order)

            open(file_path, "w") do file
                writedlm(file, [W, f, g, df, dg, W_point, f_point, g_point, df_point, dg_point, W_Ul,f_Ul,g_Ul,df_Ul,dg_Ul], ',')
            end
            println("New wavefunctions with energy ", W.*Eh_to_eV,"eV written.")
        end

        return new(nuc,m,n,l,j,r_array,f,g,df,dg,W, f_point,g_point,df_point,dg_point,W_point,f_Ul,g_Ul,df_Ul,dg_Ul,W_Ul)
    end
end

"""
    plotOrbital(orb::Orbital)
Plots the upper and lower radial components of the wavefunction of the given orbital.
"""
function plotOrbital(orb::Orbital)
    norm = simpson(orb.r, (orb.f.^2 .+ orb.g.^2))

    graph = plot(orb.r,orb.f,label="f(r)",title="Wavefunctions")
    plot!(orb.r,orb.g,label="g(r)")
    xlabel!("r (au)")
    ylabel!("∫ f(r)^2 + g(r)^2 dr = "*string(norm))
    xmax_index = findlast(orb.g.^2 .> maximum(orb.g.^2)*0.0001)
    xlims!(0, orb.r[xmax_index])
    return graph
end

"""
    (δ_BR, A_point, A_0) = BR(orb)
Calculates the Breit-Rosenthal effect for the given orbital
using the point and finite-charge nuclear wavefunctions solved in the creation of the orbital object.

**Arguments:**

- `ord::Orbital`: The orbital to calculate the BR for.

**Returns:**

- `δ_BR:Float`: relative BR effect as a decimal value.

- `A_point::Float`: hyperfine constant due to point nucleus.

- `A_0::Float`: hyperfine constant due to finite-charge nucleus.
"""
function BR(orb::Orbital)
    nuc = orb.nucleus

    R = orb.r
    f_point = orb.f_point
    g_point = orb.g_point
    f = orb.f
    g = orb.g

    # Calculation of BR correction
    factor = (4*α/(3*mp))*(nuc.μ/nuc.I)
    A_point = factor*simpson(R, f_point.*g_point./R.^2)
    A_0 = factor*simpson(R, f.*g./R.^2) 
    
    δ_BR = (A_0 - A_point)/A_point
    return (δ_BR, A_point, A_0)
end

"""
    (A_ELVP, A_MLVP) = QED(μHg199, dist="point")
Returns the electric and magnetic loop corrections to the hyperfine energy of the given orbital.
The magnetisation distribution can be chosen as `"point"` (default), `"ball"` or `"sp"` (single particle).

Results are returned in atomic energy units (Eh).
"""
function QED(orb::Orbital; dist::String="point")
    if dist == "point"
        point(r) = 1
        return QED(orb, point)
    elseif dist == "ball"
        Rₙ = sqrt(5/3)*orb.nucleus.rrms
        ball(r) = (r/Rₙ)^3*(r<Rₙ) + 1*(r>=Rₙ)
        return QED(orb, ball)
    elseif dist == "sp"
        Rₙ = sqrt(5/3)*orb.nucleus.rrms
        u(r,Rₙ) = sqrt(3/Rₙ^3)*(r<=Rₙ)
        sp(r) = orb.nucleus.F(r,u,Rₙ)
        return QED(orb, sp)
    end    
end

"""
    (A_ELVP, A_MLVP) = QED(μHg199, (r->1))
Returns the electric and magnetic loop corrections to the hyperfine energy of the given orbital.
The given distribution is used as the magnetisation distribution, `F(r)`, in the integral `A_0 = factor*simpson(R, F.(R).*f.*g./R.^2)`.

Results are returned in atomic energy units (Eh).
"""
function QED(orb::Orbital, dist::Function=(r -> 1))
    nuc = orb.nucleus

    R = orb.r
    f = orb.f
    g = orb.g
    f_Ul = orb.f_Ul
    g_Ul = orb.g_Ul

    # Calculation of Uehling (EL) correction
    factor = (4*α/(3*mp))*(nuc.μ/nuc.I)
    A_Ul = factor*simpson(R, dist.(R).*f_Ul.*g_Ul./R.^2)
    A_0 = factor*simpson(R, dist.(R).*f.*g./R.^2) 
    
    δ_ELVP = -(A_Ul - A_0)

    ## Calculation of ML correction
    Rₙ = sqrt(5/3)*nuc.rrms
    η(r) = max(r,Rₙ)/α
    χ(r) = min(r,Rₙ)/α
    point(t,r) = (α/(3*π))*sqrt(t^2-1)/t^4*(2*t^2+1)*(2*t*η(r)+1)*exp(-2*t*η(r))
    integrand(t,r) = point(t,r)*(3/(8*χ(r)^3*t^3)*(2*t*χ(r)*cosh(2*t*χ(r)) - sinh(2*t*χ(r))))

    Q_VP(r) = quadgk(t -> integrand(t,r),1,2,5,100,1000)[1]
    δMLVP = factor*simpson(R, Q_VP.(R).*dist.(R).*f.*g./R.^2)

    return (δ_ELVP, δMLVP)
end

"""
    (ϵ_BW, A_BW, A_0) = BW(orb, model="sp"; Rₘ=sqrt(5/3)*orb.nucleus.rrms, nucleon=((r,rₘ) -> sqrt(3/(sqrt(5/3)*orb.nucleus.rrms)^3)*(r<=rₘ)))
Returns the Bohr-Weisskopf effect in the given orbital with either a ball or single-particle magnetisation distribution.
The sp distribution requires a nucleon wavefunction and magnetic radius (see optional params).

**Arguments:**

- `orb::Orbital`: the orbital of interest.

- `model::String`: either `"sp"` or `"ball"` to specify the single particle or ball model.

- `Rₘ::Float`: nuclear magnetic radius.

- `nucleon(r,rₘ)::Float`: wavefunction for the odd nucleon in the sp model.

**Returns:**

- `ϵ_BW`: relative BW effect as a decimal value.

- `A_BW`: hyperfine constant due to the BW effect in units of atomic energy (Eh).

- `A_0`: hyperine constant due to a point magnetisation distribution in units of atomic energy (Eh).
"""
function BW(orb::Orbital, model::String="sp"; Rₘ=sqrt(5/3)*orb.nucleus.rrms, nucleon=((r,rₘ) -> sqrt(3/(sqrt(5/3)*orb.nucleus.rrms)^3)*(r<=rₘ)))
    if model == "ball"
        ball(r,rₘ) = (r/rₘ)^3*(r<rₘ) + 1*(r>=rₘ)
        return BW(orb, ball; Rₘ=Rₘ)
    elseif model == "sp"
        sp(r,rₘ) = orb.nucleus.F(r,nucleon,rₘ)
        return BW(orb, sp; Rₘ=Rₘ)
    end   
end

"""
    (ϵ_BW, A_BW, A_0) = BW(orb, F; Rₘ=sqrt(5/3)*orb.nucleus.rrms)
Returns the Bohr-Weisskopf effect in the given orbital for a general magnetisation distribution F(r).

**Arguments:**

- `orb::Orbital`: the orbital of interest.

- `F(r,rₘ)::Function`: Function for the magnetisation distribution.

- `Rₘ::Float`: nuclear magnetic radius.

**Returns:**

- `ϵ_BW`: relative BW effect as a decimal value.

- `A_BW`: hyperfine constant due to the BW effect in units of atomic energy (Eh).

- `A_0`: hyperine constant due to a point magnetisation distribution in units of atomic energy (Eh).
"""
function BW(orb::Orbital, model::Function; Rₘ=sqrt(5/3)*orb.nucleus.rrms)
    nuc = orb.nucleus

    f = orb.f
    g = orb.g
    R = orb.r

    factor = (4*α/(3*mp))*(nuc.μ/nuc.I)
    A_0 = factor*trapz(R, f.*g./R.^2) 
    A_BW = factor*trapz(R, (model.(R,Rₘ) .- 1).*f.*g./R.^2)
    ϵ_BW = A_BW/A_0

    return (ϵ_BW, A_BW, A_0)
end

"""
    ϵ_BW = diffBW(orb, Δ, ϵ2, δ2; BRscr=1)
Returns the relative BW effect (as a decimal value) in the specified orbital given a value for the differential BW effect and BW and BR effects in another isotope.

**Arguments:**

- `orb::Orbital`: orbital to calculate new ϵ_BW in.

- `Δ::Float`: differential anomaly between the isotope of this orbital and the other.

- `ϵ2::Float, δ2::Float`: differential BW and BR effect in second isotope.

- `BRscr::FLoat`: optional parameter to screen total BR effect if considering higher orbitals. Unscreened BR effect is calculted in the orbtial provided.

**Returns:**

- `ϵ_BW::Float`: relative BW effect (as a decimal value) in the isotope specified by `orb`.
"""
function diffBW(orb::Orbital, Δ, ϵ2, δ2; BRscr=1)
    δ1 = BR(orb)[1]
    δ12 = (δ1 - δ2)*BRscr # screening
    ϵ_BW = Δ + ϵ2 - δ12
    return ϵ_BW
end

"""
    (ϵ2_BW, A2_BW, A2_0, Rₘ) = BW_transfer(orb1, orb2, data, nucleon; tol_bisect=5e-14, printsteps=false)
Fits a magnetic radius `Rₘ`, for a given odd-nucleon wavefunction `nucleon(r,rₘ)`, to reproduce the BW effect in `orb1`.
Then calculates the BW effect in `orb2` using this value.

**Arguments:**

- `orb1::Orbital`: first oribital, used to fit `Rₘ`.

- `orb2::Orbital`: second oribital, the one which the BW effect is to be calculated in.

- `data::Tuple`: a tuple of data points in the form `data=(A1_exp::Float, A1_QED::Float)`,
where `A1_exp` is the total experimental value of the hyperfine constant and `A1_QED` is the total QED contribution to the hyperfine constant, both in the first orbital. 
Contributions are to be in atomic units of energy (Eh).

- `nucleon(r,rₘ)::Function`: odd-nucleon wavefunction to be used for fitting.

- 'tol_bisect::Float': tolerance to be used for bisection rootfinding when fitting `Rₘ`.

- `printsteps::Bool`: flag to print progress to the terminal. Usefal for debugging.
"""
function BW_transfer(orb1::Orbital, orb2::Orbital, data::Tuple, nucleon::Function; tol_bisect=5e-14, printsteps=false)
    if orb1.nucleus != orb2.nucleus
        println("Nucleui in BW_transfer are different. Result may be unphysical.")
    end

    nuc = orb1.nucleus
    (A1_exp, A1_QED) = data # in atomic units

    # Calc BW value for orb1
    A1_0 = BR(orb1)[3]
    A1_BW_exp = A1_exp - A1_QED - A1_0

    Rₘ = bisect(r -> (BW(orb1,Rₘ=r,nucleon=nucleon)[2] - A1_BW_exp), 1e-8 , 1, tol=tol_bisect, printsteps=printsteps)
    
    (ϵ2_BW, A2_BW, A2_0) = BW(orb2,Rₘ=Rₘ, nucleon=nucleon)

    return (ϵ2_BW, A2_BW, A2_0, Rₘ)
end