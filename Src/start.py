def main():
    action = 0
    while action != 3:
        print("1 : inference\n"
              "2 : training\n"
              "3 : stop\n")
        action = int(input("Please choose an action: "))
        print("1 : training ControlNetCanny\n"
              "2 : training ControlNetCannyReduced\n"
              "3 : training ControlLora\n"
              "4 : training ControlNetHed\n"
              "5 : training ControlNetHedReduced\n")
        peculiar = int(input("Please choose a training: "))


if __name__ == '__main__':
    main()