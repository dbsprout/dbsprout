CREATE TABLE currencies (
  code VARCHAR(3) PRIMARY KEY,
  name VARCHAR(50) NOT NULL,
  symbol VARCHAR(5) NOT NULL
);
CREATE TABLE exchange_rates (
  id INTEGER PRIMARY KEY,
  base_code VARCHAR(3) NOT NULL REFERENCES currencies(code),
  quote_code VARCHAR(3) NOT NULL REFERENCES currencies(code),
  rate DECIMAL(18,8) NOT NULL CHECK (rate > 0)
);
CREATE TABLE customers (
  id INTEGER PRIMARY KEY,
  email VARCHAR(255) NOT NULL UNIQUE,
  kyc_status VARCHAR(20) NOT NULL DEFAULT 'pending'
    CHECK (kyc_status IN ('pending','verified','rejected'))
);
CREATE TABLE accounts (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL REFERENCES customers(id),
  currency_code VARCHAR(3) NOT NULL REFERENCES currencies(code),
  kind VARCHAR(20) NOT NULL
    CHECK (kind IN ('checking','savings','credit')),
  balance DECIMAL(18,2) NOT NULL DEFAULT 0
);
CREATE TABLE transactions (
  id INTEGER PRIMARY KEY,
  account_id INTEGER NOT NULL REFERENCES accounts(id),
  amount DECIMAL(18,2) NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'posted'
    CHECK (status IN ('pending','posted','reversed')),
  currency_code VARCHAR(3) NOT NULL REFERENCES currencies(code)
);
CREATE TABLE ledger_entries (
  id INTEGER PRIMARY KEY,
  transaction_id INTEGER NOT NULL REFERENCES transactions(id),
  account_id INTEGER NOT NULL REFERENCES accounts(id),
  direction VARCHAR(6) NOT NULL CHECK (direction IN ('debit','credit')),
  amount DECIMAL(18,2) NOT NULL CHECK (amount >= 0)
);
CREATE TABLE cards (
  id INTEGER PRIMARY KEY,
  account_id INTEGER NOT NULL REFERENCES accounts(id),
  last_four VARCHAR(4) NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'active'
    CHECK (status IN ('active','blocked','expired'))
);
CREATE TABLE merchants (
  id INTEGER PRIMARY KEY,
  name VARCHAR(150) NOT NULL,
  category VARCHAR(40) NOT NULL
);
CREATE TABLE card_transactions (
  id INTEGER PRIMARY KEY,
  card_id INTEGER NOT NULL REFERENCES cards(id),
  merchant_id INTEGER NOT NULL REFERENCES merchants(id),
  amount DECIMAL(18,2) NOT NULL CHECK (amount >= 0)
);
CREATE TABLE statements (
  id INTEGER PRIMARY KEY,
  account_id INTEGER NOT NULL REFERENCES accounts(id),
  opening_balance DECIMAL(18,2) NOT NULL,
  closing_balance DECIMAL(18,2) NOT NULL
);
CREATE TABLE fees (
  id INTEGER PRIMARY KEY,
  transaction_id INTEGER REFERENCES transactions(id),
  amount DECIMAL(18,2) NOT NULL CHECK (amount >= 0),
  kind VARCHAR(20) NOT NULL CHECK (kind IN ('fx','wire','late'))
);
