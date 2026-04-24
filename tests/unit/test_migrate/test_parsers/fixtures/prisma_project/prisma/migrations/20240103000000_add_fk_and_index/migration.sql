ALTER TABLE posts
  ADD CONSTRAINT fk_posts_author FOREIGN KEY (author_id) REFERENCES users (id);

CREATE INDEX posts_author_id_ix ON posts (author_id);
