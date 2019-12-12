class EegStore:

    def __init__(self, ts, filtered, features_ts, theta, alpha_low, alpha_high, beta, gamma, plf_pairs, plf):
        self.ts = ts
        self.filtered = filtered
        self.features_ts = features_ts
        self.theta = theta
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.beta = beta
        self.gamma = gamma
        self.plf_pairs = plf_pairs
        self.plf = plf


class EmgStore:

    def __init__(self, *args):
        self.ts = args[0]
        self.filtered = args[1]
        self.onsets = args[2]