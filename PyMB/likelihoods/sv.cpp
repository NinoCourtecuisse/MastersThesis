#include <TMB.hpp>

// Helper function for phi and rho
// Transform x from the real line to [-1,1]
template<class Type>
Type f(Type x){
  Type y = (exp(x) -Type(1))/(Type(1) + exp(x));
  return(y);
}

template<class Type> 
Type objective_function<Type>::operator() ()
{
  DATA_SCALAR(dt);
  DATA_VECTOR(y);

  PARAMETER(mu);
  PARAMETER(log_sigma_y);
  PARAMETER(log_sigma_h);
  PARAMETER(logit_phi);
  PARAMETER(logit_rho);
  PARAMETER_VECTOR(h); // Latent process 

  Type mu_ = mu * dt;
  Type sigma_y = exp(log_sigma_y) * sqrt(dt);
  Type sigma_h = exp(log_sigma_h) * sqrt(dt);
  Type phi = f(logit_phi);
  Type rho = f(logit_rho);

  ADREPORT(mu_);
  ADREPORT(sigma_y);
  ADREPORT(sigma_h);
  ADREPORT(phi);
  ADREPORT(rho);

  // Negative log likelihood
  Type nll = 0;
  int N = y.size();

  // Contribution from latent process
  nll -= dnorm(h(0), Type(0), sigma_h / sqrt(1 - phi * phi), true);

  for(int i = 1; i < N; i++){
    nll -= dnorm(h(i), phi * h(i - 1), sigma_h, true); 
  }

  // Contribution from observations
  for (int i = 0; i < N - 1; i++) {
    Type eta = (h(i + 1) - phi * h(i)) / sigma_h; 
    nll -= dnorm(y(i), mu_ + sigma_y * exp(h(i) / Type(2)) * rho * eta,
                  sigma_y * exp(h(i) / Type(2)) * sqrt(Type(1) - rho * rho), true);
  }

  return nll;
}
