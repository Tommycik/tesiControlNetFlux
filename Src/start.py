import testControlNetHed
import testControlnet
import testControlnetReduced
import testLora
import trainingControlNetCanny
import trainingControlNetCannyReduced
import trainingControlNetHed


def main():
    action = 0
    while action != 3:
        print("1 : training\n"
              "2 : inference\n"
              "3 : stop\n")
        action = int(input("Please choose an action: "))
        if action == 1:

            print("1 : training ControlNetCanny\n"
                  "2 : training ControlNetCannyReduced\n"
                  "3 : training ControlLora\n"
                  "4 : training ControlNetHed\n"
                  "5 : training ControlNetHedReduced\n")
            peculiar = int(input("Please choose a training: "))
            if peculiar == 1:
                trainingControlNetCanny.main()
            elif peculiar == 2:
                trainingControlNetCannyReduced.main()
            elif peculiar == 3:
                trainingControlNetHed.main()
            elif peculiar == 4:
                trainingControlNetHed.main()
            elif peculiar == 5:
                print("da fare")
                #todo


        elif action == 2:

            print("1 : inference ControlNetCanny\n"
                  "2 : inference ControlNetCannyReduced\n"
                  "3 : inference ControlLora\n"
                  "4 : inference ControlNetHed\n"
                  "5 : inference ControlNetHedReduced\n")
            peculiar = int(input("Please choose a inference: "))

            if peculiar == 1:
                testControlnet.main()
            elif peculiar == 2:
                testControlnetReduced.main()
            elif peculiar == 3:
                testLora.main()
            elif peculiar == 4:
                testControlNetHed.main()
            elif peculiar == 5:
                print("da fare")
                #todo

        elif action == 3:
            break



if __name__ == '__main__':
    main()