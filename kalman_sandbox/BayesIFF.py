class IFFClassifier:
    def __init__(self, prior, likelihood):
        self.prior = prior
        self.likelihood = likelihood
        self.belief = dict(prior)

    def update(self, reply):
        if reply is None:
            return

        for label in self.belief:
            self.belief[label] *= self.likelihood[label][reply]

        total = sum(self.belief.values())
        for label in self.belief:
            self.belief[label] /= total
