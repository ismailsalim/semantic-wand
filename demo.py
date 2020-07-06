import tkinter as tk

from demo.app import App

def main():
    root = tk.Tk()
    # root.minsize(960, 480)
    app = App(root)
    root.deiconify()
    app.mainloop()

if __name__ == '__main__':
    main()