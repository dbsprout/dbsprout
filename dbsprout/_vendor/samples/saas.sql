CREATE TABLE tenants (
  id INTEGER PRIMARY KEY,
  slug VARCHAR(40) NOT NULL UNIQUE,
  plan VARCHAR(20) NOT NULL DEFAULT 'free'
    CHECK (plan IN ('free','pro','enterprise'))
);
CREATE TABLE users (
  id INTEGER PRIMARY KEY,
  tenant_id INTEGER NOT NULL REFERENCES tenants(id),
  email VARCHAR(255) NOT NULL UNIQUE,
  role VARCHAR(20) NOT NULL DEFAULT 'member'
    CHECK (role IN ('owner','admin','member'))
);
CREATE TABLE teams (
  id INTEGER PRIMARY KEY,
  tenant_id INTEGER NOT NULL REFERENCES tenants(id),
  name VARCHAR(120) NOT NULL
);
CREATE TABLE team_members (
  team_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  added_by INTEGER REFERENCES users(id),
  PRIMARY KEY (team_id, user_id),
  FOREIGN KEY (team_id) REFERENCES teams(id),
  FOREIGN KEY (user_id) REFERENCES users(id)
);
CREATE TABLE projects (
  id INTEGER PRIMARY KEY,
  tenant_id INTEGER NOT NULL REFERENCES tenants(id),
  team_id INTEGER NOT NULL REFERENCES teams(id),
  name VARCHAR(150) NOT NULL
);
CREATE TABLE tasks (
  id INTEGER PRIMARY KEY,
  tenant_id INTEGER NOT NULL REFERENCES tenants(id),
  project_id INTEGER NOT NULL REFERENCES projects(id),
  assignee_id INTEGER REFERENCES users(id),
  status VARCHAR(20) NOT NULL DEFAULT 'todo'
    CHECK (status IN ('todo','doing','done','blocked')),
  priority INTEGER NOT NULL CHECK (priority BETWEEN 1 AND 5)
);
CREATE TABLE comments (
  id INTEGER PRIMARY KEY,
  task_id INTEGER NOT NULL REFERENCES tasks(id),
  author_id INTEGER NOT NULL REFERENCES users(id),
  body TEXT NOT NULL
);
CREATE TABLE labels (
  id INTEGER PRIMARY KEY,
  tenant_id INTEGER NOT NULL REFERENCES tenants(id),
  name VARCHAR(40) NOT NULL
);
CREATE TABLE task_labels (
  task_id INTEGER NOT NULL,
  label_id INTEGER NOT NULL,
  PRIMARY KEY (task_id, label_id),
  FOREIGN KEY (task_id) REFERENCES tasks(id),
  FOREIGN KEY (label_id) REFERENCES labels(id)
);
CREATE TABLE invoices (
  id INTEGER PRIMARY KEY,
  tenant_id INTEGER NOT NULL REFERENCES tenants(id),
  amount DECIMAL(12,2) NOT NULL CHECK (amount >= 0),
  status VARCHAR(20) NOT NULL DEFAULT 'open'
    CHECK (status IN ('open','paid','void'))
);
CREATE TABLE api_keys (
  id INTEGER PRIMARY KEY,
  tenant_id INTEGER NOT NULL REFERENCES tenants(id),
  token VARCHAR(64) NOT NULL UNIQUE
);
