functions {
    real[] model_based(int T, int num_trials, int[] a1, int[] s2, int[] a2,
        int[] reward, real alpha, real beta1, real beta2, real p) {

        real log_lik[T];
        real v[2, 2];

        // Initializing values
        for (i in 1:2)
            for (j in 1:2)
                v[i, j] = 0;

        for (i in 1:T)
            log_lik[i] = 0;

        for (t in 1:num_trials) {
            real x1;
            real x2;
            x1 = 0.4*(max(v[2]) - max(v[1]));
            // Perseveration
            if (t > 1) {
                if (a1[t - 1] == 2)
                    x1 += p;
                else
                    x1 -= p;
            }
            // Exploration
            x1 *= beta1;
            // First stage choice
            if (a1[t] == 2)
                log_lik[t] += log_inv_logit(x1);
            else
                log_lik[t] += log1m_inv_logit(x1);

            // Second stage choice
            x2 = beta2*(v[s2[t], 2] - v[s2[t], 1]);
            if (a2[t] == 2)
                log_lik[t] += log_inv_logit(x2);
            else
                log_lik[t] += log1m_inv_logit(x2);

            // Learning
            v[s2[t], a2[t]] += alpha*(reward[t] - v[s2[t], a2[t]]);
        }
        return log_lik;
    }
}
data {
    int<lower=0> N; // Number of participants
    int<lower=0> T; // Maximum number of trials
    int<lower=0, upper=T> num_trials[N];
    int<lower=1, upper=2> a1[N, T]; // First stage actions
    int<lower=1, upper=2> a2[N, T]; // Second stage actions
    int<lower=1, upper=4> s1[N, T]; // First stage states
    int<lower=1, upper=2> s2[N, T]; // Second stage states
    int<lower=0, upper=1> reward[N, T]; // Rewards
}
parameters {
    // Transformed model parameters
    vector[4] subjparams[N];
    cholesky_factor_corr[4] L_Omega;
    vector<lower=0>[4] tau;
    vector[4] mu;
}
transformed parameters {
    real alpha[N];
    real beta1[N];
    real beta2[N];
    real p[N];
    matrix[4, 4] Sigma;
    real log_lik[N, T];
    for (i in 1:N) {
        alpha[i] = inv_logit(subjparams[i][1]);
        beta1[i] = exp(subjparams[i][2]);
        beta2[i] = exp(subjparams[i][3]);
        p[i] = subjparams[i][4];
    }
    Sigma = diag_pre_multiply(tau, L_Omega);
    Sigma *= Sigma';
    for (i in 1:N)
        log_lik[i] = model_based(T, num_trials[i], a1[i], s2[i], a2[i], reward[i],
            alpha[i], beta1[i], beta2[i], p[i]);
}
model {
    tau ~ cauchy(0, 1);
    mu ~ normal(0, 5);
    L_Omega ~ lkj_corr_cholesky(2);
    subjparams ~ multi_normal(mu, Sigma);
    for (i in 1:N)
        target += sum(log_lik[i]);
}
