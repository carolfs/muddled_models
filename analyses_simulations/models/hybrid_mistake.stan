functions {
    real[] hybrid_mistake(int T, int num_trials, int[] a1, int[] s2, int[] a2,
        int[] reward, int[] stage1_config, real alpha1, real alpha2, real lmbd, real beta1, 
        real beta2, real w, real p, real prob_mistake) {

        real log_lik[T];
        real q[2];
        real v[2, 2];

        // Initializing values
        for (i in 1:2)
            q[i] = 0;
        for (i in 1:2)
            for (j in 1:2)
                v[i, j] = 0;

        for (i in 1:T)
            log_lik[i] = 0;

        for (t in 1:num_trials) {
            real x1;
            real x2;
            x1 = // Model-based value
                w*0.4*(max(v[2]) - max(v[1])) +
                // Model-free value
                (1 - w)*(q[2] - q[1]);
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
                x1 *= -1;
            // Consider possibility of mistake
            if (t > 1 && stage1_config[t - 1] != stage1_config[t]) {
                log_lik[t] = log_sum_exp(
                    log(1 - prob_mistake) + log_inv_logit(x1),
                    log(prob_mistake) + log_inv_logit(-x1));
            }
            else
                log_lik[t] = log_inv_logit(x1);

            // Second stage choice
            x2 = beta2*(v[s2[t], 2] - v[s2[t], 1]);
            if (a2[t] == 2)
                log_lik[t] += log_inv_logit(x2);
            else
                log_lik[t] += log1m_inv_logit(x2);

            // Learning
            q[a1[t]] += alpha1*(v[s2[t], a2[t]] - q[a1[t]]) +
                alpha1*lmbd*(reward[t] - v[s2[t], a2[t]]);
            v[s2[t], a2[t]] += alpha2*(reward[t] - v[s2[t], a2[t]]);
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
    int<lower=1, upper=2> s1[N, T]; // Second stage states
    int<lower=1, upper=2> s2[N, T]; // Second stage states
    int<lower=0, upper=1> reward[N, T]; // Rewards
    int<lower=1, upper=2> stage1_config [N, T]; // First-stage configuration regarding choices
}
parameters {
    // Transformed model parameters
    vector[8] subjparams[N];
    cholesky_factor_corr[8] L_Omega;
    vector<lower=0>[8] tau;
    vector[8] mu;
}
transformed parameters {
    real alpha1[N];
    real alpha2[N];
    real lmbd[N];
    real beta1[N];
    real beta2[N];
    real w[N];
    real p[N];
    real prob_mistake[N];
    matrix[8, 8] Sigma;
    real log_lik[N, T];
    for (i in 1:N) {
        alpha1[i] = inv_logit(subjparams[i][1]);
        alpha2[i] = inv_logit(subjparams[i][2]);
        lmbd[i] = inv_logit(subjparams[i][3]);
        beta1[i] = exp(subjparams[i][4]);
        beta2[i] = exp(subjparams[i][5]);
        w[i] = inv_logit(subjparams[i][6]);
        p[i] = subjparams[i][7];
        prob_mistake[i] = inv_logit(subjparams[i][8]);
    }
    Sigma = diag_pre_multiply(tau, L_Omega);
    Sigma *= Sigma';
    for (i in 1:N)
        log_lik[i] = hybrid_mistake(T, num_trials[i], a1[i], s2[i], a2[i], reward[i],
            stage1_config[i], alpha1[i], alpha2[i], lmbd[i], beta1[i], beta2[i],
            w[i], p[i], prob_mistake[i]);
}
model {
    tau ~ cauchy(0, 1);
    mu ~ normal(0, 5);
    L_Omega ~ lkj_corr_cholesky(2);
    subjparams ~ multi_normal(mu, Sigma);
    for (i in 1:N)
        target += sum(log_lik[i]);
}
