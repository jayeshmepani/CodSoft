import json
import os

class Contact:
    def __init__(self, name, phone, email, address):
        self.name = name
        self.phone = phone
        self.email = email
        self.address = address

class ContactBook:
    def __init__(self):
        self.contacts = []

    def add_contact(self, contact):
        self.contacts.append(contact)
        print("Contact added successfully.")

    def view_contacts(self):
        if not self.contacts:
            print("Contact list is empty.")
        else:
            print("Contact List:")
            for contact in self.contacts:
                print(f"Name: {contact.name}, Phone: {contact.phone}")

    def search_contact(self, keyword):
        found = False
        for contact in self.contacts:
            if keyword.lower() in contact.name.lower() or keyword in contact.phone:
                print(f"Name: {contact.name}, Phone: {contact.phone}, Email: {contact.email}, Address: {contact.address}")
                found = True
        if not found:
            print("No matching contacts found.")

    def update_contact(self, current_name, new_name=None, new_phone=None, new_email=None, new_address=None):
        updated = False
        for contact in self.contacts:
            if contact.name.lower() == current_name.lower():
                if new_name:
                    contact.name = new_name
                    updated = True
                if new_phone:
                    contact.phone = new_phone
                    updated = True
                if new_email:
                    contact.email = new_email
                    updated = True
                if new_address:
                    contact.address = new_address
                    updated = True
                if updated:
                    print("Contact updated successfully.")
                    return
        print("Contact not found.")

    def delete_contact(self, name):
        for contact in self.contacts:
            if contact.name.lower() == name.lower():
                self.contacts.remove(contact)
                print("Contact deleted successfully.")
                return
        print("Contact not found.")
    
    def save_contacts(self, filename):
        with open(filename, 'w') as file:
            json.dump([vars(contact) for contact in self.contacts], file)
        print("Contacts saved successfully.")

    def load_contacts(self, filename):
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
                self.contacts = [Contact(**contact_data) for contact_data in data]
            print("Contacts loaded successfully.")
        except FileNotFoundError:
            print("No previous contacts found.")

    def delete_all_contacts(self):
        self.contacts = []
        print("All contacts deleted successfully.")

def main():
    contact_book = ContactBook()
    filename = "contacts.json"

    # Check if previous contacts exist
    if os.path.isfile(filename):
        choice = input("Previous contacts found. Do you want to load them? (y/n): ").lower()
        if choice == 'y':
            contact_book.load_contacts(filename)
        else:
            contact_book.delete_all_contacts()

    while True:
        print("\nContact Book Menu:")
        print("1. Add Contact")
        print("2. View Contacts")
        print("3. Search Contact")
        print("4. Update Contact")
        print("5. Delete Contact")
        print("6. Save Contacts")
        print("7. Delete All Contacts")
        print("8. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            name = input("Enter name: ")
            phone = input("Enter phone number: ")
            email = input("Enter email: ")
            address = input("Enter address: ")
            new_contact = Contact(name, phone, email, address)
            contact_book.add_contact(new_contact)
        elif choice == '2':
            contact_book.view_contacts()
        elif choice == '3':
            keyword = input("Enter name or phone number to search: ")
            contact_book.search_contact(keyword)
        elif choice == '4':
            current_name = input("Enter current name of contact to update: ")
            new_name = input("Enter new name (press Enter to skip): ")
            new_phone = input("Enter new phone number (press Enter to skip): ")
            new_email = input("Enter new email (press Enter to skip): ")
            new_address = input("Enter new address (press Enter to skip): ")
            contact_book.update_contact(current_name, new_name, new_phone, new_email, new_address)
        elif choice == '5':
            name = input("Enter name of contact to delete: ")
            contact_book.delete_contact(name)
        elif choice == '6':
            contact_book.save_contacts(filename)
        elif choice == '7':
            contact_book.delete_all_contacts()
        elif choice == '8':
            print("Exiting the Contact Book.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
