CREATE TABLE authors (
    id INT PRIMARY KEY,
    name VARCHAR(120) NOT NULL
);

CREATE TABLE books (
    id INT PRIMARY KEY,
    author_id INT REFERENCES authors(id),
    title VARCHAR(255) NOT NULL
);
