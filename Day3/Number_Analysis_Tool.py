def input_number(): 
    try:
        num = int(input("Enter a number: "))
        return num
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
        return input_number()
    
def is_even(num): 
    if num % 2 == 0: return True
    else: return False

def is_prime(num): 
    if num <= 1:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def is_palindrome(num): 
    def reverse_num(num):
        rev = 0
        while num > 0:
            rev = rev * 10 + num % 10
            num //= 10
        return rev
    return num == reverse_num(num)

def reverse_num(num):
    rev = 0
    while num > 0:
        rev = rev * 10 + num % 10
        num //= 10
    return rev

def digit_count(num): 
    count = 0
    while num > 0:
        num //= 10
        count += 1
    return count

def main():
    input_num = input_number()
    print(f"Is even: {is_even(input_num)}")
    print(f"Is prime: {is_prime(input_num)}")
    print(f"Is palindrome: {is_palindrome(input_num)}")
    print(f"Reverse: {reverse_num(input_num)}")
    print(f"Digit count: {digit_count(input_num)}")
main()