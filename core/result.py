# core/result.py
class Result:
    def __init__(self):
        self.data = []

    def save(self, state):
        self.data.append(state.__dict__.copy())
