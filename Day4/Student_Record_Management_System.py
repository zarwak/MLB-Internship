#Student Record Management System (Persistent Version)

import json
import os

FILE_NAME = "students.json"
Students = []

def load_students():
    if not os.path.exists(FILE_NAME):
        return []
    try:
        with open(FILE_NAME, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        print("Warning: could not read data file. Starting with an empty list.")
        return []

def save_students():
    try:
        with open(FILE_NAME, "w") as f:
            json.dump(Students, f, indent=4)
    except IOError:
        print("Error: could not save records to file.")

def get_age(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Please enter a valid number for age.")

def add_student():
    name = input("Enter student name: ")
    while True:
        roll_number = input("Enter student roll number: ")
        if roll_number in [student['roll_number'] for student in Students]:
            print("Roll number already exists! Please enter a unique roll number.")
        else:
            break
    age = get_age("Enter student age: ")
    course = input("Enter student course: ")
    student = {"name": name, "roll_number": roll_number, "age": age, "course": course}
    Students.append(student)
    save_students()
    print("Student added successfully!")

def view_students():
    if not Students:
        print("No students found.")
        return
    for i, student in enumerate(Students):
        print(f"{i+1}. Name: {student['name']}, Roll Number: {student['roll_number']}, Age: {student['age']}, Course: {student['course']}")

def search_student():
    roll_number = input("Enter student roll number to search: ")
    for student in Students:
        if student['roll_number'] == roll_number:
            print(f"Name: {student['name']}, Roll Number: {student['roll_number']}, Age: {student['age']}, Course: {student['course']}")
            return
    print("Student not found.")

def delete_student():
    roll_number = input("Enter student roll number to delete: ")
    for i, student in enumerate(Students):
        if student['roll_number'] == roll_number:
            del Students[i]
            save_students()
            print("Student deleted successfully!")
            return
    print("Student not found.")

def update_student():
    roll_number = input("Enter student roll number to update: ")
    for student in Students:
        if student['roll_number'] == roll_number:
            name = input("Enter new name (leave blank to keep current): ")
            age = input("Enter new age (leave blank to keep current): ")
            course = input("Enter new course (leave blank to keep current): ")
            if name:
                student['name'] = name
            if age:
                try:
                    student['age'] = int(age)
                except ValueError:
                    print("Invalid age. Keeping the current age.")
            if course:
                student['course'] = course
            save_students()
            print("Student updated successfully!")
            return
    print("Student not found.")

def main():
    global Students
    Students = load_students()
    while True:
        print("\nStudent Record Management System")
        print("1. Add Student")
        print("2. View Students")
        print("3. Search Student")
        print("4. Delete Student")
        print("5. Update Student")
        print("6. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            add_student()
        elif choice == '2':
            view_students()
        elif choice == '3':
            search_student()
        elif choice == '4':
            delete_student()
        elif choice == '5':
            update_student()
        elif choice == '6':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()
