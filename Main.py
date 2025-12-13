class Main:
    def __init__(self):
        print("init")

    def run(self, mode):
        if (mode == 0):
            print(mode)
        else:
            print(mode)

if (__name__ == "__main__"):
    try:
        user_input = input("Mode: Training(0), Test(1): ")
        mode = int(user_input)
        if (mode not in [0, 1]):
            print("Invalid input, default mode: Training")
            mode = 0
    except:
        print("Invalid input, default mode: Training")
        mode = 0

    A2C = Main()
    A2C.run(mode = mode)