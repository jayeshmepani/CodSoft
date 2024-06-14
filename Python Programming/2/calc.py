def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y == 0:
        return "Error! Division by zero is not allowed."
    else:
        return x / y

print("Simple Calculator")
print("Operations:")
print("1. Addition (+)")
print("2. Subtraction (-)")
print("3. Multiplication (*)")
print("4. Division (/)")

while True:
    choice = input("Enter operation choice (1/2/3/4): ")

    if choice in ('1', '2', '3', '4'):
        n1 = float(input("Enter first number: "))
        n2 = float(input("Enter second number: "))

        if choice == '1':
            print("Result:", add(n1, n2))
        elif choice == '2':
            print("Result:", subtract(n1, n2))
        elif choice == '3':
            print("Result:", multiply(n1, n2))
        elif choice == '4':
            print("Result:", divide(n1, n2))
        
        break
    else:
        print("Invalid choice. Please enter a valid operation choice.")
