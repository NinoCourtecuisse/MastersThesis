#include <TMB.hpp>

// Helper function for phi and rho
// Transform x from the real line to [-1,1]
template<class Type>
vector<Type> f(vector<Type> x){
    vector<Type> y(x.size());
    for (int i = 0; i < x.size(); i++) {
        y(i) = (exp(x(i)) - Type(1)) / (Type(1) + exp(x(i)));
    }
    return y;
}

template<class Type>
Type objective_function<Type>::operator() ()
{
    DATA_SCALAR(dt);
    DATA_INTEGER(N);
    DATA_INTEGER(P);
    DATA_ARRAY(y);                 // (n_particles, n)

    PARAMETER_VECTOR(log_sigma_y);  // (n_particles,)
    PARAMETER_VECTOR(log_sigma_h);
    PARAMETER_VECTOR(logit_phi);
    PARAMETER_VECTOR(logit_rho);
    PARAMETER_VECTOR(mu);
    PARAMETER_ARRAY(h);            // Latent process: (n_particles, n)

    vector<Type> mu_ = mu * dt;
    vector<Type> sigma_y = exp(log_sigma_y) * sqrt(dt);
    vector<Type> sigma_h = exp(log_sigma_h) * sqrt(dt);
    vector<Type> phi = f(logit_phi);
    vector<Type> rho = f(logit_rho);

    ADREPORT(mu);
    ADREPORT(sigma_y);
    ADREPORT(sigma_h);
    ADREPORT(phi);
    ADREPORT(rho);

    // Sum of negative log likelihoods for each particle
    Type nll = 0;

    // Contribution from latent process
    vector<Type> tmp(P);
    for (int j = 0; j < P; j++) {
        tmp(j) = sigma_h(j) / sqrt(Type(1.0) - phi(j) * phi(j));
    }
    nll -= sum( dnorm(h.col(0).vec(), Type(0), tmp, true) );

    // std::cout << nll << std::endl;

    for(int i = 1; i < N; i++){
        vector<Type> tmp2(P);
        for (int j = 0; j < P; j++) {
            tmp2(j) = phi(j) * h(j, i-1);
        }
        nll -= sum( dnorm(h.col(i).vec(), tmp2, sigma_h, true) );
    }

    // Contribution from observations
    for (int i = 0; i < N - 1; i++) {
        vector<Type> eta(P);
        for (int j = 0; j < P; j++) {
            eta(j) = (h(j, i+1) - phi(j) * h(j, i)) / sigma_h(j);
        }
        nll -= sum( dnorm(y.col(i).vec(), mu + sigma_y * exp(h.col(i).vec() / Type(2)) * rho * eta,
                        sigma_y * exp(h.col(i).vec() / Type(2)) * sqrt(Type(1) - rho * rho), true) );
    }
    return nll;
}
