data {
    int M; // number of participants
    int N; // number of trials per participant
    int K; // number of predictors
    int<lower=0, upper=1> y[M,N]; // stay for each trial
    matrix[N,K] x[M]; // predictors for each trial
}
parameters {
    // Coefficients for each participants
    vector[K] coefs[M];
    // Distribution of coefficients
    cholesky_factor_corr[K] L_Omega;
    vector<lower=0>[K] tau;
    vector[K] mu;
}
transformed parameters {
    matrix[K, K] Sigma;
    Sigma = diag_pre_multiply(tau, L_Omega);
    Sigma *= Sigma';
}

model {
    tau ~ cauchy(0, 1);
    mu ~ normal(0, 5);
    L_Omega ~ lkj_corr_cholesky(2);
    coefs ~ multi_student_t(4, mu, Sigma);
    for (p in 1:M) {
        y[p] ~ bernoulli_logit(x[p] * coefs[p]);
    }
}

generated quantities {
    vector[K] grp;
    grp = coefs[1];
    for (p in 2:M)
        grp += coefs[p];
    grp /= 1.*M;
}
