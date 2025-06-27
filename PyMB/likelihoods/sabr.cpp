#include <TMB.hpp>

// Helper function for phi and rho
// Transform x from the real line to [-1,1]
template<class Type>
Type f(Type x){
    Type y = (exp(x) -Type(1))/(Type(1) + exp(x));
    return(y);
}

template<class Type>
Type local_var(Type x, Type h, Type beta){
    Type sigma = exp(2 * h) * exp(x * (beta - 2));
    return(sigma);
}

template<class Type> 
Type objective_function<Type>::operator() ()
{
    DATA_VECTOR(X);

    PARAMETER(mu);
    PARAMETER(log_beta);
    PARAMETER(log_sigma);
    PARAMETER(logit_rho);
    PARAMETER_VECTOR(h); // Latent process 

    Type beta = exp(log_beta);
    Type sigma = exp(log_sigma);
    Type rho = f(logit_rho);

    ADREPORT(mu);
    ADREPORT(beta);
    ADREPORT(sigma);
    ADREPORT(rho);

    // Negative log likelihood
    Type nll = 0;
    int N = X.size();

    // Contribution from latent process
    nll -= dnorm(h(0), Type(0), sigma, true);

    for(int i = 1; i < N; i++){
        nll -= dnorm(h(i), h(i - 1) - Type(0.5) * sigma * sigma, sigma, true);
    }

    // Contribution from observations
    for (int i = 0; i < N - 1; i++) {
        Type eta = (h(i + 1) - h(i) - Type(0.5) * sigma * sigma) / sigma;
        Type var = local_var(X(i), h(i), beta);
        nll -= dnorm(X(i + 1), X(i) + mu - Type(0.5) * var + sqrt(var) * rho * eta,
                        sqrt(var) * sqrt(Type(1) - rho * rho), true);
    }

    return nll;
}
