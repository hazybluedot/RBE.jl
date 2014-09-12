using Distributions
using Models

abstract RecursiveBaysianEstimator

type EKF <: RecursiveBaysianEstimator
    apriori::Distribution
end

function initialize!(fitler::RecursiveBaysianEstimator, apriori::Distribution)
    filter.apriori = apriori
end

function predict!{F <: RecursiveBaysianEstimator, M <: AbstractModel, T}(filt::F, sys::M, u::Array{T,1})
    x = mean(filt.apriori)
    P = cov(filt.apriori)
    Q = cov(sys_N(sys))

    x = sys_f(sys)(x,u)

    Fx = sys_∇x_f(sys)(x,u)
    G = sys_∇x_g(sys)(x)

    P = Fx*P*Fx' + G*Q*G'
    filt.apriori = gmvnormal(x,P)
    filt.apriori
end

function correct!{F <: RecursiveBaysianEstimator, M <: AbstractModel, T}(filt::F, mm::M, z::Array{T,1})
    x = mean(filt.aprioi)
    Σ = cov(filt.apriori)
    
    Z = mes_h(mm)(x)
    R = mes_σ(mm)(x)
    H = mes_∇x_h(mm)(x)
    
    K = (Σ*H'/((H*Σ).dot(H') + R))'
    KH = K'*H
    Σ = (eye(4)-KH)*Σ*(eye(4)-KH)' + K'*R*K
    filt.apriori = MultivariateNormal(x,Σ)
    filt.apriori
end

function run(filter::RecursiveBaysianEstimator, apriori::Distribution, Z, U)
    for (z,u) in zip(Z,U)
        post_pre = predict!(filter, sysmodel, u)
        post_plus = correct!(filter, mesmodel, z)
        log_callback(post_pre, post_plus)
    end
end
