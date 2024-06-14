import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
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

class ToDoListGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("To-Do List App")

        self.todo_list = []

        self.frame = tk.Frame(self.master)
        self.frame.pack()

        self.label = tk.Label(self.frame, text="To-Do List:")
        self.label.grid(row=0, column=0, columnspan=2)

        self.listbox = tk.Listbox(self.frame, width=50)
        self.listbox.grid(row=1, column=0, columnspan=2)


        self.add_button = tk.Button(self.frame, text="Add", command=self.add_task)
        self.add_button.grid(row=2, column=0)

        self.mark_button = tk.Button(self.frame, text="Mark as Done", command=self.mark_task_as_completed)
        self.mark_button.grid(row=4, column=0, columnspan=2)  # Spanning both columns


        self.delete_button = tk.Button(self.frame, text="Delete", command=self.delete_task)
        self.delete_button.grid(row=2, column=1)

        self.save_button = tk.Button(self.frame, text="Save", command=self.save_to_file)
        self.save_button.grid(row=3, column=0)

        self.load_button = tk.Button(self.frame, text="Load", command=self.load_from_file)
        self.load_button.grid(row=3, column=1)

    def view_tasks(self):
        self.listbox.delete(0, tk.END)
        for task in self.todo_list:
            self.listbox.insert(tk.END, f"{task.title}: {task.status}")

    def add_task(self):
        title = simpledialog.askstring("Title", "Enter the task title:")
        description = simpledialog.askstring("Description", "Enter the task description:")
        if title and description:
            task = Task(title, description)
            self.todo_list.append(task)
            messagebox.showinfo("Success", "Task added successfully.")
            self.view_tasks()

    def mark_task_as_completed(self):
        index = self.listbox.curselection()
        if index:
            index = int(index[0])
            self.todo_list[index].mark_as_completed()
            messagebox.showinfo("Success", "Task marked as done.")
            self.view_tasks()
        else:
            messagebox.showwarning("Warning", "No task selected.")

    def delete_task(self):
        index = self.listbox.curselection()
        if index:
            index = int(index[0])
            del self.todo_list[index]
            messagebox.showinfo("Success", "Task deleted successfully.")
            self.view_tasks()
        else:
            messagebox.showwarning("Warning", "No task selected.")

    def save_to_file(self):
        filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if filename:
            with open(filename, "w") as file:
                for task in self.todo_list:
                    file.write(f"{task.title}|{task.description}|{task.status}|{task.created_at}|{task.completed_at}\n")
            messagebox.showinfo("Success", "To-Do List saved to file.")

    def load_from_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if filename:
            self.todo_list = []
            with open(filename, "r") as file:
                for line in file:
                    title, description, status, created_at, completed_at = line.strip().split("|")
                    task = Task(title, description)
                    task.status = status
                    task.created_at = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S.%f")
                    if completed_at != 'None':
                        task.completed_at = datetime.strptime(completed_at, "%Y-%m-%d %H:%M:%S.%f")
                    self.todo_list.append(task)
            messagebox.showinfo("Success", "To-Do List loaded from file.")
            self.view_tasks()

def main():
    root = tk.Tk()
    app = ToDoListGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
