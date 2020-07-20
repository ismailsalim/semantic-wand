from demo.app import App

import tkinter as tk
import argparse


def main():
    root = tk.Tk()
    # root.minsize(960, 480)
    app = App(root)
    root.deiconify()
    app.mainloop()

if __name__ == '__main__':
    main()