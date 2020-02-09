#MTBayesC requires the support for prior for delta for d is the set
#of all 2^ntrait outcomes of dj:
function MTBayesABC!(xArray,xpx,
                     xRinvArray,xpRinvx, #Heterogeneous residuals
                     wArray,betaArray,
                     deltaArray,
                     alphaArray,
                     vare,varEffects,
                     BigPi)
    nMarkers = length(xArray)
    nTraits  = length(alphaArray)

    Rinv     = inv(vare) #Do Not Use inv.(): elementwise inversion
    Ginv     = inv.(varEffects)

    β        = zeros(typeof(betaArray[1][1]),nTraits)
    newα     = zeros(typeof(alphaArray[1][1]),nTraits)
    oldα     = zeros(typeof(alphaArray[1][1]),nTraits)
    δ        = zeros(typeof(deltaArray[1][1]),nTraits)
    w        = zeros(typeof(wArray[1][1]),nTraits) #for rhs

    for marker=1:nMarkers
        x, xRinv = xArray[j], xRinvArray[j]

        for trait = 1:nTraits
            β[trait]  = betaArray[trait][marker]
         oldα[trait]  = newα[trait] = alphaArray[trait][marker]
            δ[trait]  = deltaArray[trait][marker]
            w[trait]  = dot(xRinv,wArray[trait])+xpx[marker]*oldα[trait]
        end

        for k=1:nTraits
            Ginv11 = Ginv[marker][k,k]
            nok    = deleteat!(collect(1:nTraits),k)
            Ginv12 = Ginv[marker][k,nok]
            C11    = Ginv11+Rinv[k,k]*xpRinvx[marker]
            C12    = Ginv12+xpRinvx[marker]*Matrix(Diagonal(δ[nok]))*Rinv[k,nok]
            #C12    = Ginv12+xpx[marker]*Rinv[k,nok].*δ[nok]' #δ[:,nok] : row vector,

            invLhs0  = 1/Ginv11
            rhs0     = - Ginv12'β[nok]
            gHat0    = (rhs0*invLhs0)[1,1]
            invLhs1  = 1/C11
            rhs1     = w'*Rinv[:,k]-C12'β[nok] #w transpose
            gHat1    = (rhs1*invLhs1)[1,1]

            d0 = copy(δ)
            d1 = copy(δ)
            d0[k] = 0.0
            d1[k] = 1.0

            logDelta0  = -0.5*(log(Ginv11)- gHat0^2*Ginv11) + log(BigPi[d0]) #logPi
            logDelta1  = -0.5*(log(C11)-gHat1^2*C11) + log(BigPi[d1]) #logPiComp

            probDelta1 =  1.0/(1.0+exp(logDelta0-logDelta1))
            if(rand()<probDelta1)
                δ[k] = 1
                β[k] = newα[k] = gHat1 + randn()*sqrt(invLhs1)
                BLAS.axpy!(oldα[k]-newα[k],x,wArray[k])
            else
                β[k] = gHat0 + randn()*sqrt(invLhs0)
                δ[k] = 0
                newα[k] = 0
                if oldα[k] != 0
                    BLAS.axpy!(oldα[k],x,wArray[k])
                end
            end
        end
        for trait = 1:nTraits
            betaArray[trait][marker]       = β[trait]
            deltaArray[trait][marker]      = δ[trait]
            alphaArray[trait][marker]      = newα[trait]
        end
    end
end
