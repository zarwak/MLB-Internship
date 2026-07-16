#grades

def grades(num:int) -> str:
    if num >= 85:
        return "A"
    elif num >= 80:
        return "A-"
    elif num >= 75:
        return "B+"
    elif num >= 70:
        return "B-"
    elif num >= 65:
        return "C+"
    elif num >= 60:
        return "C-"
    elif num >= 50:
        return "D"
    else:
        return "F"
    

def main():
    print("\t\t\t\tWelcome to the Student Grading System!\t\t\t\t")
    name = input("Enter your name: ")
    class_name = input("Enter your class name: ")
    total_marks = 0
    count = 0
    while True:
        subject = input("Enter the subject name: ")
        try:
            marks = int(input("Enter your marks: "))
            if marks < 0 or marks > 100:
                print("Marks should be between 0 and 100. Please try again.")
                continue
            total_marks += marks
            count += 1
        except ValueError:
            print("Invalid input. Please enter a valid integer for marks.")
            continue
        more_data = input("Do you want to enter more data? (y/n): ").strip().lower()
        if more_data != 'y':
            break
    if count > 0:
        average_marks = total_marks / count
        grade = grades(average_marks)
        print(f"\n{name}, your average marks are: {average_marks:.2f}")
        print(f"Your grade is: {grade}")

if __name__ == "__main__":
    main()
            
