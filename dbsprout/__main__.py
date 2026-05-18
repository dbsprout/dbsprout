"""Enable ``python -m dbsprout`` as an entry point alias for the CLI."""

from dbsprout.cli.app import app

if __name__ == "__main__":
    app()
