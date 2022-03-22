using Revise;
using JWAS,DataFrames,CSV,Statistics,JWAS.Datasets
# Step 2: Read data
phenofile  = dataset("phenotypes.csv")
genofile   = dataset("genotypes.csv")
phenotypes = CSV.read(phenofile,DataFrame,delim = ',',header=true,missingstrings=["NA"])
first(phenotypes,5)
IDs = phenotypes[!,:ID][1:50]

genotypes  = get_genotypes(genofile,separator=',',method="GBLUP", name4core = IDs);
#genotypes  = get_genotypes(genofile,separator=',',method="GBLUP");

# Step 3: Build Model Equations
model_equation  ="y1 = intercept + x1 + x2 + genotypes"
model = build_model(model_equation);
model.M[1].genetic_variance
model.M[1].G
model.M[1].π
model.M[1].obsID
model.obsID
model.M[1].APYinfo
model.M[1].genotypes

# Step 4: Set Factors or Covariates
set_covariate(model,"x1");
# Step 5: Set Random or Fixed Effects
set_random(model,"x2");

# Step 6: Run Analysis
out=runMCMC(model,phenotypes);

# Step 7: Check Accuruacy
results    = innerjoin(out["EBV_y1"], phenotypes, on = :ID)
accuruacy  = cor(results[!,:EBV],results[!,:bv1])
