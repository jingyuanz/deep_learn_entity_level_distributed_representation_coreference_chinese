class Config:
    def __init__(self):
        self.data_path = './out.txt'
        self.result_path = './results.txt'
        self.embedding_size = 256
        self.repr_size = 5129
        self.learning_rate = 1e-3
        self.batch_size = 32
        self.I = 5129
        self.M1 = 1000
        self.M2 = 500
        self.D = 500
        self.M3 = 20
        self.a_fn = 0.7
        self.a_fa = 0.4
        self.a_wl = 1.0
        self.NA = '#'
        self.test_batch_size = 50


