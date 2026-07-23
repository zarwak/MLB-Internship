# 📚 Library Management System

A console-based Library Management System built in Python using **Object-Oriented Programming (OOP)**.
You can add books, view them, search, borrow, and return — and all data is saved to a JSON file so it stays even after you close the program.

This README also explains the OOP concepts I learned while building it, using examples from my own code.

---

## 🚀 How to run

```bash
python Library_Management_System.py
```

Then follow the menu (type a number and press Enter):

```
1. Add a new book
2. View all books
3. Search for a book
4. Borrow a book
5. Return a book
6. Exit
```

---

## 🧠 What is OOP?

**Object-Oriented Programming** is a way of writing code by grouping related **data** and **actions** together into "objects" — instead of scattering them around in loose variables and functions.

Think of a real library. A book has *data* (title, author, ID) and things you can *do* with it (borrow it, return it). OOP lets me bundle both into one unit called a **`Book`**. This makes the code easier to read, reuse, and grow.

The four ideas OOP is famous for:
- **Class** – a blueprint
- **Object** – a real thing built from the blueprint
- **Inheritance** – making a new class based on an existing one
- **Encapsulation** – keeping data and the code that uses it together (and protected)

---

## 🏗️ Classes and Objects

A **class** is a *blueprint*. An **object** is an actual *thing* built from that blueprint.

Blueprint (class) — this is the plan for what every book looks like:

```python
class Book:
    def __init__(self, title, author, id):
        self.title = title
        self.author = author
        self.id = id
        self.is_borrowed = False
```

Object — a real book made from the blueprint:

```python
new_book = Book("The Hobbit", "Tolkien", "101")
```

> 🔑 One class → as many objects as you want. The `Book` class is written once, but I can create hundreds of book objects from it.

---

## 🔧 Attributes and Methods

- **Attributes** = the *data* an object holds (its nouns).
- **Methods** = the *actions* an object can do (its verbs — they're just functions that live inside a class).

In my `Book` class:

| Attributes (data) | Methods (actions) |
|-------------------|-------------------|
| `title`, `author`, `id`, `is_borrowed` | `to_dict()`, `describe()` |

Example of a method:

```python
def describe(self):
    return f"ID: {self.id}, Title: {self.title}, Author: {self.author}"
```

Calling it:

```python
print(new_book.describe())
# ID: 101, Title: The Hobbit, Author: Tolkien
```

---

## 🛠️ Constructors (`__init__`)

The **constructor** is a special method named `__init__`. Python runs it **automatically** the moment you create an object. Its job is to set up the starting data.

```python
def __init__(self, title, author, id):
    self.title = title          # store the title on this object
    self.author = author
    self.id = id
    self.is_borrowed = False     # every new book starts as "not borrowed"
```

So when I write `Book("The Hobbit", "Tolkien", "101")`, Python quietly calls `__init__` and fills in those values for me. I never call `__init__` by hand.

---

## 👥 Creating multiple objects

Because a class is just a blueprint, I can stamp out as many objects as I like — each one independent:

```python
book1 = Book("The Hobbit", "Tolkien", "101")
book2 = Book("Dune", "Herbert", "102")
book3 = Book("1984", "Orwell", "103")
```

If `book1` gets borrowed, `book2` and `book3` are unaffected — each object keeps its **own copy** of the attributes. In my project, the `Library` class holds them all in a list:

```python
self.books = []   # a list that will hold many Book objects
```

---

## 🔑 The `self` keyword

`self` means **"this particular object."** It's how a method knows *which* object's data it's working with.

```python
def borrow_book(self, id):
    for book in self.books:      # self.books = THIS library's book list
        if book.id == id:
            book.is_borrowed = True
```

When you call `book1.describe()`, Python secretly passes `book1` in as `self`. That's why every method's first parameter is `self` — it's the object itself, handed in automatically. Without `self`, a method wouldn't know whose `title` to return.

---

## 🧬 Inheritance & Encapsulation

### Why inheritance is used

Inheritance lets a new class **reuse** everything from an existing class, then add or change just the bits that are different — so I don't copy-paste the same code twice.

In a library, an **e-book** is still a book: it has a title, author, ID, and can be borrowed. The *only* extra thing is a file size. Rewriting all the book logic again would be wasteful — so `EBook` **inherits** from `Book`.

### Parent and Child classes

- **Parent (base) class** = the general one → `Book`
- **Child (derived) class** = the specialised one → `EBook`

```python
class EBook(Book):                       # EBook inherits from Book
    def __init__(self, title, author, id, file_size):
        super().__init__(title, author, id)   # reuse Book's setup
        self.file_size = file_size            # add the new detail
```

`super()` means *"run the parent's version first."* So an `EBook` gets `title`, `author`, `id`, and `is_borrowed` for free from `Book`, plus its own `file_size`. **An `EBook` *is a* `Book`.**

### Method overriding

**Overriding** = a child class replaces a method it inherited, to behave a little differently.

`Book` has a `describe()` method. `EBook` overrides it to add the e-book label:

```python
# In Book (parent)
def describe(self):
    return f"ID: {self.id}, Title: {self.title}, Author: {self.author}"

# In EBook (child) — same name, extended behaviour
def describe(self):
    return super().describe() + f" [E-Book, {self.file_size}]"
```

Now `describe()` does the right thing depending on the object type — a plain book prints normally, an e-book adds `[E-Book, ...]`.

### Public vs Private attributes

- **Public** = accessible from anywhere. Most of my attributes are public (e.g. `book.title`).
- **Private** = meant to be used only inside the class. In Python you signal this by starting the name with an underscore, e.g. `_is_borrowed` (single underscore = "please don't touch from outside"; double underscore `__` makes Python actively hide it).

Python doesn't *force* privacy like some languages — it's more of a polite agreement between programmers.

### Encapsulation and its benefits

**Encapsulation** = keeping an object's data and the methods that work on that data **bundled together**, and controlling how the outside world changes it.

In my project, the *right* way to borrow a book is through the method:

```python
self.library.borrow_book("101")   # ✅ goes through the proper method
```

instead of poking at the data directly like `book.is_borrowed = True` from random places. Benefits:
- 🛡️ **Safety** – the method can add checks (e.g. "is it already borrowed?") before changing anything.
- 🧹 **Cleaner code** – the rest of the program doesn't need to know *how* borrowing works internally.
- 🔧 **Easy to change** – if I later add a "due date," I only edit `borrow_book()`, not the whole program.

---

## 📍 Where inheritance was used in this project

- **`Book`** is the parent class.
- **`EBook`** is the child class — it inherits everything from `Book` using `class EBook(Book)`, calls `super().__init__(...)` to reuse the parent's constructor, and **overrides** `describe()` and `to_dict()` to include its extra `file_size` detail.

This is inheritance applied *where it makes sense*: an e-book genuinely **is a** book, so it should reuse book logic rather than duplicate it.

---

## 🧗 Challenges I faced and how I solved them

**1. My data crashed every time I loaded it from JSON.**
JSON can only store plain data (dictionaries, lists, text) — not Python objects. When I loaded the file, I got a list of **dictionaries**, but my code was treating them like **`Book` objects** (`book.title`), which caused `AttributeError` crashes.
✅ *Solution:* I added a `to_dict()` method to turn objects into dictionaries before saving, and a `from_dict()` method to rebuild real `Book` objects after loading. So the program always works with proper objects in memory.

**2. The program crashed on the very first run.**
There was no `books.json` file yet, so opening it threw a `FileNotFoundError`. An empty file also broke `json.load`.
✅ *Solution:* I wrapped the file reading in `try / except (FileNotFoundError, json.JSONDecodeError)` — if there's no valid file, the program just starts with an empty library instead of crashing.

**3. I was repeating the same save/load code in every menu option.**
✅ *Solution:* I moved it into two reusable methods, `load_books()` and `save_books()`, so each menu option just calls them. Less repetition, fewer mistakes.

**4. Understanding `self` at first.**
It felt confusing why every method needed `self`.
✅ *Solution:* Once I realised `self` just means *"this specific object,"* it clicked — it's how a method knows whose data to use.

---

## 📂 Project structure

```
Day5/
├── Library_Management_System.py   # the program
├── books.json                     # auto-created; stores book data
└── README.md                      # this file
```

## ✅ Requirements met

- [x] Classes for books and the library
- [x] Inheritance (`EBook` extends `Book`)
- [x] Data saved to a JSON file (persists between runs)
- [x] Graceful error handling for invalid input / missing file
