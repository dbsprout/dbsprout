CREATE TABLE parent (
  id INTEGER PRIMARY KEY,
  name VARCHAR(50) NOT NULL UNIQUE,
  parent_id INTEGER REFERENCES parent(id)
);
CREATE TABLE child (
  parent_id INTEGER NOT NULL REFERENCES parent(id),
  seq INTEGER NOT NULL,
  status VARCHAR(10) NOT NULL CHECK (status IN ('a','b','c')),
  PRIMARY KEY (parent_id, seq)
);
