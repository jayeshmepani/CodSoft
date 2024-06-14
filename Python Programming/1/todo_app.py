import os
from datetime import datetime

class Task:
    def __init__(self, title, description):
        self.title = title
        self.description = description
        self.status = "Incomplete"
        self.created_at = datetime.now()
        self.completed_at = None

    def mark_as_completed(self):
        self.status = "Completed"
        self.completed_at = datetime.now()

class ToDoList:
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)

    def view_tasks(self):
        if not self.tasks:
            print("Your To-Do List is empty.")
        else:
            print("Your To-Do List:")
            for index, task in enumerate(self.tasks, start=1):
                print(f"{index}. Title: {task.title}")
                print(f"   Description: {task.description}")
                print(f"   Status: {task.status}")
                print(f"   Created: {task.created_at}")
                if task.completed_at:
                    print(f"   Completed: {task.completed_at}")
                print()

    def mark_task_as_completed(self, task_number):
        if 0 < task_number <= len(self.tasks):
            self.tasks[task_number - 1].mark_as_completed()
            print("Task marked as done.")
        else:
            print("Invalid task number.")

    def delete_task(self, task_number):
        if 0 < task_number <= len(self.tasks):
            del self.tasks[task_number - 1]
            print("Task deleted successfully.")
        else:
            print("Invalid task number.")

    def save_to_file(self, filename):
        with open(filename, "w") as file:
            for task in self.tasks:
                file.write(f"{task.title}|{task.description}|{task.status}|{task.created_at}|{task.completed_at}\n")

    def load_from_file(self, filename):
        if os.path.exists(filename):
            self.tasks = []
            with open(filename, "r") as file:
                for line in file:
                    title, description, status, created_at, completed_at = line.strip().split("|")
                    task = Task(title, description)
                    task.status = status
                    task.created_at = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S.%f")
                    if completed_at != 'None':
                        task.completed_at = datetime.strptime(completed_at, "%Y-%m-%d %H:%M:%S.%f")
                    self.tasks.append(task)
            print("To-Do List loaded from file.")
        else:
            print("File not found. Starting with an empty To-Do List.")

def display_menu():
    print("Welcome to To-Do List App")
    print("1. View To-Do List")
    print("2. Add Task")
    print("3. Mark Task as Done")
    print("4. Delete Task")
    print("5. Save To File")
    print("6. Load From File")
    print("7. Exit")

def main():
    filename = "todo_list.txt"
    todo_list = ToDoList()

    while True:
        display_menu()
        choice = input("Enter your choice: ")

        if choice == '1':
            todo_list.view_tasks()
        elif choice == '2':
            title = input("Enter the task title: ")
            description = input("Enter the task description: ")
            task = Task(title, description)
            todo_list.add_task(task)
            print("Task added successfully.")
        elif choice == '3':
            task_number = int(input("Enter the index of the task to mark as done: "))
            todo_list.mark_task_as_completed(task_number)
        elif choice == '4':
            task_number = int(input("Enter the index of the task to delete: "))
            todo_list.delete_task(task_number)
        elif choice == '5':
            todo_list.save_to_file(filename)
            print("To-Do List saved to file.")
        elif choice == '6':
            todo_list.load_from_file(filename)
        elif choice == '7':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
