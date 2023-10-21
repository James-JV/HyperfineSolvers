"""
An example code that calculates the BW effect in H-like Hg from data for the hyperfine constant in muonic Hg.

"""

include("HyperfineSolvers.jl")

function calc_BW(orb1, orb2, data)
      
    u(r,Rₘ,n) = sqrt((2*n+3)/Rₘ^3)*(r/Rₘ)^n*(r<=Rₘ)
    v(r,Rₘ,n) = sqrt((2*n+1)*(2*n+2)*(2*n+3)/(2*Rₘ^3))*((Rₘ-r)/Rₘ)^n*(r<=Rₘ)

    nmodels = 5
    temp_vals = Array{Float64, 2}(undef, nmodels,4)
    for i in 1:nmodels
        if i <= 3
            n_power = i-1
            temp_vals[i,:] .= BW_transfer(orb1, orb2, data, (r,Rₘ) -> u(r,Rₘ,n_power),printsteps=true)
        elseif i >=4
            n_power = i-3
            temp_vals[i,:] .= BW_transfer(orb1, orb2, data, (r,Rₘ) -> v(r,Rₘ,n_power),printsteps=true)
        end
    end
    display(temp_vals)
    return temp_vals
    #return sum(x -> x/5, temp_vals, dims=1)
end

a = fm_to_au * 2.3/(4*log(3))
rc(rrms) = sqrt((5/3)*rrms^2-(7/3)*(π*a)^2)
ρ(r,rrms) = (1+exp((r-rc(rrms))/a))^(-1)

# Nucleus
Hg199 = Nucleus(80,1,1/2,0,ρ,5.4474*fm_to_au,0.5039)

# Orbital
μHg199 = Orbital(Hg199,mmu,1,0,1/2,W_i=-374000,Ngrid=1200,h=0.01,use_save=true,AM_tol=2e-16)
eHg199 = Orbital(Hg199,1,1,0,1/2,Ngrid=2000,h=0.01,use_save=true,AM_tol=5e-16)
#plotOrbital(μHg199)

## Calculate QED correction
Rₙ = sqrt(5/3)*Hg199.rrms
ball(r) = (r/Rₙ)^3*(r<Rₙ) + 1*(r>=Rₙ)
(ELVP, MLVP) = QED(μHg199, dist="ball") 
#data = (0.47*1000/Eh_to_eV, ELVP+MLVP) # (A1_exp, A1_QED) atomic units
data = ((0.47-0.12)*1000/Eh_to_eV, 0.11775967354861194 + 0.14573070706339641) # for ball

## Calculate BW in μ from e
results = calc_BW(μHg199, eHg199, data)
println("ABW: ",results[:,2])
println("Rm: ",results[:,4])

#####
# average table across nucleon distributiuons
(ϵ, A_BW, A_0, Rₘ) = sum(x -> x/5, results, dims=1)

# Print averages
println("ϵ = ",ϵ*100," %")
println("A_BW = ",A_BW*Eh_to_eV/1000," keV")
println("A_0 = ",A_0*Eh_to_eV/1000," keV")
println("Rₘ = ",A_0/fm_to_au," fm")