import random

def determine_winner(user_choice, computer_choice):
    """
    Determine the winner of the Rock-Paper-Scissors game.

    Args:
        user_choice: User's choice (str).
        computer_choice: Computer's choice (str).

    Returns:
        A string indicating the winner ('user', 'computer', or 'tie').
    """
    if user_choice == computer_choice:
        return 'tie'
    elif (user_choice == 'rock' and computer_choice == 'scissors') or \
         (user_choice == 'scissors' and computer_choice == 'paper') or \
         (user_choice == 'paper' and computer_choice == 'rock'):
        return 'user'
    else:
        return 'computer'

def play_game():
    """
    Play one round of the Rock-Paper-Scissors game.
    """
    user_choice = input("Enter your choice (rock, paper, scissors): ").lower()
    while user_choice not in ['rock', 'paper', 'scissors']:
        print("Invalid choice. Please enter 'rock', 'paper', or 'scissors'.")
        user_choice = input("Enter your choice: ").lower()

    computer_choice = random.choice(['rock', 'paper', 'scissors'])

    print(f"\nYour choice: {user_choice}")
    print(f"Computer's choice: {computer_choice}")

    winner = determine_winner(user_choice, computer_choice)
    if winner == 'tie':
        print("It's a tie!")
        return 'tie'
    elif winner == 'user':
        print("Congratulations! You win!")
        return 'user'
    else:
        print("Computer wins!")
        return 'computer'

def main():
    print("Welcome to Rock-Paper-Scissors Game")
    user_score = 0
    computer_score = 0

    play_again = 'y'
    while play_again.lower() == 'y':
        result = play_game()
        if result == 'user':
            user_score += 1
        elif result == 'computer':
            computer_score += 1

        print(f"Your score: {user_score}, Computer score: {computer_score}")

        play_again = input("\nDo you want to play again? (y/n): ")
        while play_again.lower() not in ['y', 'n']:
            print("Invalid choice. Please enter 'y' or 'n'.")
            play_again = input("Do you want to play again? (y/n): ")

    print("Thanks for playing!")

if __name__ == "__main__":
    main()
