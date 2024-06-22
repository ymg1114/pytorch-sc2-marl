class Trajectory2:
    def __init__(self, seq_len, time_gen):
        self.seq_len = seq_len
        self.time_gen = time_gen  # 트레적터리 최초 생성 시간
        self.data = list()

    @property
    def gen_time(self):
        return self.time_gen

    @property
    def len(self):
        return len(self.data)

    def put(self, data):
        self.data.append(data)
        return

    def get(self):
        return self.data.pop(0)
