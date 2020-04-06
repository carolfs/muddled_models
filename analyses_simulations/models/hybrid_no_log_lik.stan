functions {
    real hybrid(int T, int[] a1, int[] s2, int[] a2,
        int[] reward, real alpha1, real alpha2, real lmbd, real beta1, 
        real beta2, real w, real p) {

        real log_lik;
        real q[2];
        real v[2, 2];

        // Initializing values
        for (i in 1:2)
            q[i] = 0;
        for (i in 1:2)
            for (j in 1:2)
                v[i, j] = 0;

        log_lik = 0;

        for (t in 1:T) {
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
                log_lik += log_inv_logit(x1);
            else
                log_lik += log1m_inv_logit(x1);

            // Second stage choice
            x2 = beta2*(v[s2[t], 2] - v[s2[t], 1]);
            if (a2[t] == 2)
                log_lik += log_inv_logit(x2);
            else
                log_lik += log1m_inv_logit(x2);

            // Learning
            q[a1[t]] += alpha1*(v[s2[t], a2[t]] - q[a1[t]]) +
                alpha1*lmbd*(reward[t] - v[s2[t], a2[t]]);
            v[s2[t], a2[t]] += alpha2*(reward[t] - v[s2[t], a2[t]]);
        }
        return log_lik;
    }
}
data {
    int<lower=0> T; // Number of trials
    int<lower=1, upper=2> a1[T]; // First stage actions
    int<lower=1, upper=2> a2[T]; // Second stage actions
    int<lower=1, upper=2> s2[T]; // Second stage states
    int<lower=0, upper=1> reward[T]; // Rewards
}
parameters {
    real<lower=0, upper=1> alpha1;
    real<lower=0, upper=1> alpha2;
    real<lower=0, upper=1> lmbd;
    real<lower=0> beta1;
    real<lower=0> beta2;
    real<lower=0, upper=1> w;
    real p;
}
model {
    beta1 ~ cauchy(0, 5);
    beta2 ~ cauchy(0, 5);
    p ~ cauchy(0, 1);
    target += hybrid(T, a1, s2, a2, reward,
        alpha1, alpha2, lmbd, beta1, beta2,
        w, p);
}
