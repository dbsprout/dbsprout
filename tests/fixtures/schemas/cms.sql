CREATE TABLE sites (
  id INTEGER PRIMARY KEY,
  domain VARCHAR(120) NOT NULL UNIQUE,
  name VARCHAR(150) NOT NULL
);
CREATE TABLE users (
  id INTEGER PRIMARY KEY,
  email VARCHAR(255) NOT NULL UNIQUE,
  display_name VARCHAR(120) NOT NULL
);
CREATE TABLE roles (
  id INTEGER PRIMARY KEY,
  name VARCHAR(40) NOT NULL UNIQUE
);
CREATE TABLE permissions (
  id INTEGER PRIMARY KEY,
  code VARCHAR(60) NOT NULL UNIQUE
);
CREATE TABLE role_permissions (
  role_id INTEGER NOT NULL,
  permission_id INTEGER NOT NULL,
  PRIMARY KEY (role_id, permission_id),
  FOREIGN KEY (role_id) REFERENCES roles(id),
  FOREIGN KEY (permission_id) REFERENCES permissions(id)
);
CREATE TABLE user_roles (
  user_id INTEGER NOT NULL,
  role_id INTEGER NOT NULL,
  site_id INTEGER NOT NULL REFERENCES sites(id),
  PRIMARY KEY (user_id, role_id),
  FOREIGN KEY (user_id) REFERENCES users(id),
  FOREIGN KEY (role_id) REFERENCES roles(id)
);
CREATE TABLE pages (
  id INTEGER PRIMARY KEY,
  site_id INTEGER NOT NULL REFERENCES sites(id),
  parent_id INTEGER REFERENCES pages(id),
  author_id INTEGER NOT NULL REFERENCES users(id),
  slug VARCHAR(120) NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'draft'
    CHECK (status IN ('draft','published','archived'))
);
CREATE TABLE blocks (
  id INTEGER PRIMARY KEY,
  page_id INTEGER NOT NULL REFERENCES pages(id),
  kind VARCHAR(20) NOT NULL
    CHECK (kind IN ('text','image','embed','gallery')),
  position INTEGER NOT NULL CHECK (position >= 0)
);
CREATE TABLE media (
  id INTEGER PRIMARY KEY,
  site_id INTEGER NOT NULL REFERENCES sites(id),
  uploaded_by INTEGER NOT NULL REFERENCES users(id),
  url VARCHAR(255) NOT NULL,
  mime VARCHAR(60) NOT NULL
);
CREATE TABLE revisions (
  id INTEGER PRIMARY KEY,
  page_id INTEGER NOT NULL REFERENCES pages(id),
  editor_id INTEGER NOT NULL REFERENCES users(id),
  prev_revision_id INTEGER REFERENCES revisions(id),
  summary VARCHAR(200) NOT NULL
);
CREATE TABLE tags (
  id INTEGER PRIMARY KEY,
  site_id INTEGER NOT NULL REFERENCES sites(id),
  label VARCHAR(50) NOT NULL
);
CREATE TABLE page_tags (
  page_id INTEGER NOT NULL,
  tag_id INTEGER NOT NULL,
  PRIMARY KEY (page_id, tag_id),
  FOREIGN KEY (page_id) REFERENCES pages(id),
  FOREIGN KEY (tag_id) REFERENCES tags(id)
);
