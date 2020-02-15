def main():
    y = 20
    print(y)

    import code
    code.InteractiveConsole(locals=locals()).interact()
    quit()

    x = 10
    print(x)

if __name__ == "__main__":
    main()
