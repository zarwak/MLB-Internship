# Number Analysis Tool

A console-based Python program that takes a number from the user and reports several things about it: whether it's even or odd, whether it's prime, whether it's a palindrome, its reverse, and its digit count.

## Overview
This project was built as a beginner-friendly exercise to practice:
- Conditional statements
- Loops (`while`, `for`)
- Functions and recursion
- Basic input validation with `try`/`except`
- Number logic (even/odd, primality, palindrome, digit reversal)

## Features
- Prompts for a number and re-prompts automatically on invalid input
- Checks if the number is even or odd
- Checks if the number is prime
- Checks if the number is a palindrome (reads the same forwards and backwards)
- Reverses the digits of the number
- Counts the number of digits

## How to Run
1. Open the project folder in your terminal.
2. Run the following command:
   ```bash
   python Number_Analysis_Tool.py
   ```
3. Enter a number when prompted and view the results.

## Project Files
- Number_Analysis_Tool.py - Main program combining even/odd, prime, palindrome, reverse, and digit count checks into one tool.
- coding.ipynb - Notebook of smaller practice problems covering conditional statements, loops, and logic-building exercises that led up to the main tool.

## Notebook Practice Notes
`coding.ipynb` covers the building blocks that fed into the final tool:
- **Conditional statements**: positive/negative/zero check, even/odd check, grade calculator, largest of three numbers, leap year check
- **Loops**: printing ranges, filtering even numbers, summing 1 to N, multiplication tables, digit counting
- **Logic-building problems**: reversing a number, palindrome check, Fibonacci sequence, prime check, listing primes between 1 and 100

Each problem was solved with test cases first, then wrapped in a function and a `main()` for structure.

## Notes
`Number_Analysis_Tool.py` pulls together several of the standalone logic problems from the notebook (even/odd, prime, palindrome, reverse) into a single reusable tool, using recursion for input validation instead of a loop.
