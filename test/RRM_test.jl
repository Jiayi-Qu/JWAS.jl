using Revise
using JWAS
using DataFrames, CSV, LinearAlgebra, Kronecker, Statistics;
using Random
using GSL;
using Distributions;
data_path = "/Users/apple/Box/Jiayi_Computer/UCD_PhD/RRM/simple_simulation/simulation_testCode/"
geno_file = data_path * "Genotype2_50.csv"
RR_pheno = data_path * "RR_BLUE_missing2_PSA.csv"
pheno = CSV.read(RR_pheno, DataFrame)
ncoef = 3
tp = length(unique(pheno[!, :time]));
### Generate Lengendre Polynomial Covariates
function generatefullPhi(timevec, ncoeff = 3)
    # timevec: a vector of time points for all observations
    # ncoeff : the number of polynomial coefficients to be estimated
    times = sort(unique(timevec)) # vector of whole timepoints sorted
    tmin = minimum(times)
    tmax = maximum(times)
    # standardized time points
    qi = 2 * (times .- tmin) ./ (tmax .- tmin) .- 1
    Φ = Matrix{Float64}(undef, length(times), ncoeff) # Phi of size ntimes x ncoeff
    for i = 1:ncoeff
        n = i - 1
        ## Given the normalized function from L. R. Schaeffer
        Φ[:, i] = sqrt((2 * n + 1) / 2) * sf_legendre_Pl.(n, qi) # construct the matrix
    end
    return Φ
end

myPhi = generatefullPhi(pheno[!, :time], ncoef);


# add corresponding Phi1 and Phi2 columns to the pheno_training data frame
nrec = size(pheno,1)
wholePhi = Array{Float64}(undef, nrec, ncoef);
r = 1
for timei in pheno[!, :time]
    wholePhi[r,:] = myPhi[timei,:]
    global r += 1
end
for i in 1:ncoef
    pheno[!,Symbol("Phi$i")] = wholePhi[:,i]
end


cd(data_path * "RRM_new")

hetero_residual = join("+ Day" .* string.(collect(1:10)))
model_equation = "PSA = geno + Phi1 + Phi2 + Phi3 + Phi1*ID + Phi2*ID + Phi3*ID"  * hetero_residual
geno = get_genotypes(geno_file, separator = ',', method = "BayesC", estimatePi = true);
model = build_model(model_equation);
for i in 1:ncoef
    set_covariate(model, "Phi$i")
end
for i in 1:3
    set_random(model, "Phi$i*ID")
end

for d in 1:10
    set_random(model, "Day$d")
end


# results 14
#@time output_old=runMCMC(model,pheno,
#    chain_length=5000,RRM=myPhi, output_samples_for_all_parameters = true,seed = 123);


#results 15
#
@time output_new=runMCMC(model,pheno,
    chain_length=100,RRM=myPhi, output_samples_for_all_parameters = true,seed = 123);








EBV_ids = readdlm("/Users/apple/Box/Jiayi_Computer/UCD_PhD/RRM/simple_simulation/simulation_testCode/RRM_new/results17/MCMC_samples_marker_effects_1.txt", header = true, ',')[1]
EBV_ids[:,3]
#u1 = CSV.read("/Users/apple/Box/Jiayi_Computer/UCD_PhD/RRM/simple_simulation/simulation_testCode/RRM_new/results1/EBV_1.txt", DataFrame) # estimated first random regression coefficients for individuals
#u2 = CSV.read("/Users/apple/Box/Jiayi_Computer/UCD_PhD/RRM/simple_simulation/simulation_testCode/RRM_new/results/EBV_2.txt", DataFrame) # estimated first random regression coefficients for individuals
#EBVs = DataFrame(ID = u1[!, :ID],u1 = u1[!, :EBV],u2 = u2[!, :EBV])



    #### Validation (sample codes, users need to do the training/testing split)
u1 = CSV.read("/Users/apple/Box/Jiayi_Computer/UCD_PhD/RRM/simple_simulation/simulation_testCode/RRM_new/results1/EBV_1.txt", DataFrame) # estimated first random regression coefficients for individuals
u2 = CSV.read("/Users/apple/Box/Jiayi_Computer/UCD_PhD/RRM/simple_simulation/simulation_testCode/RRM_new/results/EBV_2.txt", DataFrame) # estimated first random regression coefficients for individuals
EBVs = DataFrame(ID = u1[!, :ID],u1 = u1[!, :EBV],u2 = u2[!, :EBV])

# Calculate the BV for individuals
n = size(EBVs, 1)
Idt = Matrix{Float64}(I, n, n)
Phi = generatefullPhi(collect(1:tp), ncoef)
Z = Idt ⊗ Phi
us = [[] for i = 1:n]
for i = 1:n
    us[i] = Matrix(EBVs)[i, 2:end]
end
uhat = vcat(us...)
gBLUP = Z * uhat
gBLUP = convert(Array{Float64,1}, gBLUP)
correctEBVs = DataFrame(
    ID = repeat(EBVs[!, :ID], inner = tp),
    time = repeat(collect(1:tp), outer = n),
    EBV = gBLUP)
testingEBVs = filter(row -> row.ID in testingInd, correctEBVs)
accuracy = []
for timei = 1:tp
    println("Day$timei.")
    pheno_testingi = filter(row -> row.time in [timei], pheno_testing)
    select!(pheno_testingi,Not(:time))
    EBV_testingi = filter(row -> row.time in [timei], testingEBVs)
    select!(EBV_testingi,Not(:time))
    final_dfi = innerjoin(pheno_testingi, EBV_testingi, on = :ID)
    res = cor(final_dfi[!,:EBV], final_dfi[!,:PSA])
    push!(accuracy, res)
end
accuracy

mean(accuracy)
