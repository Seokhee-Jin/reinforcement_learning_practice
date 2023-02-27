class A():
    def __init__(self):
        self.a = 1

    def printA(self):
        print(self.a)

cls1 = A()
cls1.printA()
cls1.a = 3
A.printA(cls1)
