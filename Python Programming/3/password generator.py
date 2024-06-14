import random
import string

def generate_password(length):
    """
    Generates a random password based on user-specified criteria.

    Args:
        length: Desired length of the password (int).

    Returns:
        A randomly generated password (str).
    """
    if length <= 0:
        return "Password length must be a positive integer."

    if length <= 4:
        password = ""
        for i in range(length):
            position = i + 1
            print(f"\nSelect character set for position {position}:")
            character_set = get_character_set()
            password += random.choice(character_set)
    else:
        # Ask user for number of digits, uppercase, lowercase, and punctuation marks
        print(f"\nEnter the number of characters for each category to include in the password (total length: {length}):")
        no_of_digits = int(input("Number of digits: "))
        no_of_uppercase = int(input("Number of uppercase letters: "))
        no_of_lowercase = int(input("Number of lowercase letters: "))
        no_of_punctuation = int(input("Number of punctuation marks: "))

        total_characters = no_of_digits + no_of_uppercase + no_of_lowercase + no_of_punctuation
        if total_characters > length:
            return "Total characters cannot exceed password length."

        # Generate characters for each category
        password = ''.join(random.choices(string.digits, k=no_of_digits))
        password += ''.join(random.choices(string.ascii_uppercase, k=no_of_uppercase))
        password += ''.join(random.choices(string.ascii_lowercase, k=no_of_lowercase))
        password += ''.join(random.choices(string.punctuation, k=no_of_punctuation))

        remaining_length = length - total_characters
        if remaining_length > 0:
            password += ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation, k=remaining_length))

        # Shuffle the password characters
        password_list = list(password)
        random.shuffle(password_list)
        password = ''.join(password_list)

    return password

def get_character_set():
    print("1. Uppercase letters")
    print("2. Lowercase letters")
    print("3. Digits")
    print("4. Punctuation")
    choice = input("Enter your choice: ")
    while choice not in ['1', '2', '3', '4']:
        print("Invalid choice. Please enter a number between 1 and 4.")
        choice = input("Enter your choice: ")

    if choice == '1':
        return string.ascii_uppercase
    elif choice == '2':
        return string.ascii_lowercase
    elif choice == '3':
        return string.digits
    else:
        return string.punctuation

def main():
    print("Password Generator")

    # Get user input for password length
    while True:
        try:
            length = int(input("Enter desired password length: "))
            password = generate_password(length)
            print("\nYour generated password is:", password)
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    main()
