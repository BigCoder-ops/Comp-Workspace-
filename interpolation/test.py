class stack():
    def __init__(self):
        self.items = []

    def push(self,item):
        self.items.append(item)
    def pop(self):
        self.pop()
    def peek(self):
        return self.items[-1]

    def  is_empty(self):
        return len(self.items) == 0

    def len