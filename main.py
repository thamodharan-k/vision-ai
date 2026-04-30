"""
main.py
-------
Entry point for the Real-Time Vision App.
Run with:  python main.py
"""

import sys
import os

# Ensure we can import sibling modules regardless of CWD
sys.path.insert(0, os.path.dirname(__file__))

from gui import App


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
