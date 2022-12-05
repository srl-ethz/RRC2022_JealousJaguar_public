import numpy as np

HYPERPARAM_IN_PAPER = {
    "sigma": 3e-4,
    "alpha": 3e-4,
    "amplify_alpha": 0.6, # choose from [0.6, 0.8]
    "amplify_beta": 1.2, # choose from [1.2, 1.4]
    "epsilon": 1e-4,
    "mixup_alpha": 0.4
}

class DataAug(object):
    def __init__(self):
        pass

    def aug(self, obs_raw, acs, rews, method="gaussian"):
        self.obs = obs_raw

        aug_method = eval(f"self.{method}_aug")
        obs = aug_method()
        return obs, acs, rews

    def gaussian_aug(self):
        obs = self.obs
        sigma = HYPERPARAM_IN_PAPER["sigma"]
        noise = np.random.randn(obs.shape[0], obs.shape[1]) * sigma
        return obs + noise

    def uniform_aug(self):
        obs = self.obs
        alpha = HYPERPARAM_IN_PAPER["alpha"]
        noise = np.random.uniform(-alpha, alpha, size=obs.shape)
        return obs + noise

    def amplify_aug(self):
        obs = self.obs
        alpha = HYPERPARAM_IN_PAPER["amplify_alpha"]
        beta = HYPERPARAM_IN_PAPER["amplify_beta"]
        noise = np.random.uniform(alpha, beta, size=obs.shape)
        return obs * noise

    def dimdropout_aug(self):
        obs = self.obs
        n, p = 1, 0.5
        noise = np.random.binomial(n, p, size=obs.shape)
        return obs * noise

    def statemixup_aug(self):
        obs = self.obs
        alpha = HYPERPARAM_IN_PAPER["mixup_alpha"]

        lams = np.random.beta(alpha, alpha, size=obs.shape[0]-1)
        obs_t = obs[:-1, :]
        obs_tplus1 = obs[1:, :]
        tmp_obs = []
        for i in range(len(obs_t)):
            ob_t = obs_t[i]
            ob_tplus1 = obs_tplus1[i]
            lam = lams[i]
            tmp_ob = ob_t * lam + ob_tplus1 * (1-lam)
            tmp_obs.append(tmp_ob)
        tmp_obs.append(obs[-1, :])
        obs = np.vstack(tmp_obs)
        return obs

    def stateswitch_aug(self):
        pass

if __name__ == '__main__':
    obs = np.ones(shape=(1000, 10))
    acs = np.ones(shape=(1000, 5))
    rews = np.ones(shape=(1000, ))

    data_augmentor = DataAug()
    obs, acs, rews = data_augmentor.aug(obs, acs, rews, method="statemixup")