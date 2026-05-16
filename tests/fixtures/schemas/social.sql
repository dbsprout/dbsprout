CREATE TABLE users (
  id INTEGER PRIMARY KEY,
  handle VARCHAR(40) NOT NULL UNIQUE,
  email VARCHAR(255) NOT NULL UNIQUE,
  status VARCHAR(20) NOT NULL DEFAULT 'active'
    CHECK (status IN ('active','deactivated','banned'))
);
CREATE TABLE profiles (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id),
  display_name VARCHAR(120) NOT NULL,
  bio TEXT
);
CREATE TABLE follows (
  follower_id INTEGER NOT NULL,
  followee_id INTEGER NOT NULL,
  PRIMARY KEY (follower_id, followee_id),
  FOREIGN KEY (follower_id) REFERENCES users(id),
  FOREIGN KEY (followee_id) REFERENCES users(id)
);
CREATE TABLE posts (
  id INTEGER PRIMARY KEY,
  author_id INTEGER NOT NULL REFERENCES users(id),
  visibility VARCHAR(20) NOT NULL DEFAULT 'public'
    CHECK (visibility IN ('public','followers','private')),
  body TEXT NOT NULL
);
CREATE TABLE media (
  id INTEGER PRIMARY KEY,
  post_id INTEGER NOT NULL REFERENCES posts(id),
  kind VARCHAR(10) NOT NULL CHECK (kind IN ('image','video','gif')),
  url VARCHAR(255) NOT NULL
);
CREATE TABLE comments (
  id INTEGER PRIMARY KEY,
  post_id INTEGER NOT NULL REFERENCES posts(id),
  author_id INTEGER NOT NULL REFERENCES users(id),
  parent_comment_id INTEGER REFERENCES comments(id),
  body TEXT NOT NULL
);
CREATE TABLE likes (
  user_id INTEGER NOT NULL,
  post_id INTEGER NOT NULL,
  PRIMARY KEY (user_id, post_id),
  FOREIGN KEY (user_id) REFERENCES users(id),
  FOREIGN KEY (post_id) REFERENCES posts(id)
);
CREATE TABLE messages (
  id INTEGER PRIMARY KEY,
  sender_id INTEGER NOT NULL REFERENCES users(id),
  recipient_id INTEGER NOT NULL REFERENCES users(id),
  body TEXT NOT NULL
);
CREATE TABLE hashtags (
  id INTEGER PRIMARY KEY,
  tag VARCHAR(50) NOT NULL UNIQUE
);
CREATE TABLE post_hashtags (
  post_id INTEGER NOT NULL,
  hashtag_id INTEGER NOT NULL,
  PRIMARY KEY (post_id, hashtag_id),
  FOREIGN KEY (post_id) REFERENCES posts(id),
  FOREIGN KEY (hashtag_id) REFERENCES hashtags(id)
);
CREATE TABLE notifications (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id),
  actor_id INTEGER REFERENCES users(id),
  kind VARCHAR(20) NOT NULL
    CHECK (kind IN ('like','comment','follow','mention'))
);
