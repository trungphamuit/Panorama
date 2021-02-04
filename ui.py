from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os, sys, subprocess
import doancv

def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener ="open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])

def demo():
	doancv.demo()
	open_file('demo.png')	

def browse():
	global folder_dir
	folder_dir = filedialog.askdirectory()
	folderLabel.configure(text=folder_dir)

def sumit():
	doancv.main(folder_dir)
	open_file('pano.png')


folder_dir = ''
root = Tk()

# icon and title of program
root.title('Do an CV')

# define components
# logo and title
logo = Image.open('ui_img/uit.png').resize((180, 180), Image.ANTIALIAS)
logo = ImageTk.PhotoImage(logo)
logoLabel = Label(image=logo)
myLabel = Label(root, text='Panorama stitching')

# frame (main manu)
frame = LabelFrame(root, text='Main menu')

# put buttons and labels (widgets) inside menu frame
demoButton = Button(frame, text='Demo', command=demo)
browseButton = Button(frame, text='Browse', command=browse)
folderLabel = Label(frame)
submitButton = Button(frame, text='Stitch', command=sumit)
exitButton = Button(frame, text='Exit', command=root.quit)

WIDGETS = [
	demoButton,
	browseButton,
	submitButton,
	exitButton
]

# throw everything on screen
logoLabel.grid(row=0, column=0)
myLabel.grid(row=0, column=1)

frame.grid(columnspan=2)

demoButton.grid(row=1)
browseButton.grid(row=2)
folderLabel.grid(row=3)
submitButton.grid(row=4)
exitButton.grid(row=5)

# padding all buttons
for widget in WIDGETS:
	widget.configure(padx=50, pady=20)

root.mainloop()
