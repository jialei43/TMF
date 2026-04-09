class Config:
    def __init__(self):
        self.train_datapath = r'./train.txt'
        self.test_datapath = r'./test.txt'
        self.dev_path = r'./dev.txt'
        self.classpath = r'./class.txt'

if __name__ == '__main__':
    config = Config()
    print(config.train_datapath)