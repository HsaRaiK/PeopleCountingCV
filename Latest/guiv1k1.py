import tkinter as tk
from tkinter import ttk
from tkinter import * 
from tkinter.filedialog import askopenfilename, askopenfilenames

source = ''
root = tk.Tk()

# This is the section of code which creates the main window
root.geometry('400x600')
root.configure(background='#C0C0C0')
root.title('opServator')

# Start window in the middle of the screen
windowWidth = root.winfo_reqwidth()
windowHeight = root.winfo_reqheight()
positionRight = int(root.winfo_screenwidth()/2 - windowWidth/2)
positionDown = int(root.winfo_screenheight()/2 - windowHeight/2)
root.geometry("+{}+{}".format(positionRight, positionDown))

# Maximize the window
root.attributes('-zoomed', True)

# This is the section of code which creates a label
lable1 = tk.Label(root, text="", bg='#F0FFFF', font=('arial', 12, 'normal'))
lable1.pack(pady=(30, 10), fill=tk.X)

def pathdef(l):
    
    VIDEO_PATH = askopenfilename()
    global source
    source = VIDEO_PATH
    l.config(text=source)
    return source

def set_source_camera(l):
    global source
    source = 0
    l.config(text="Camera was selected as input")
    return source

# This is the section of code which creates buttons
path_btn = Button(root, text='Select Input video', bg='#E0EEEE', font=('arial', 12, 'normal'), command=lambda: pathdef(lable1))
path_btn.pack(pady=10, fill=tk.X)
alt_path_def = Button(root, text='Select Camera as Input', bg='#E0EEEE', font=('arial', 12, 'normal'), command=lambda:set_source_camera(lable1))
alt_path_def.pack(pady=10, fill=tk.X)

# This is the section of code which creates an image
main_page_canvas = tk.Canvas(root)
main_page_image = PhotoImage(file='sd.gif')
#ain_page_image = main_page_image.zoom(1.5)
main_page_image_width = main_page_image.width()
main_page_canvas.create_image(positionRight-(main_page_image_width/3), positionDown/3, anchor=NW, image=main_page_image)
main_page_canvas.pack(fill=BOTH, expand=YES)

def give_source():
    return source

def closewin():
    root.quit()

start_btn = Button(root, text='Start', bg='#ceffcf', font=('arial', 12, 'normal'), command=closewin)
start_btn.pack(pady=(10, 30), side=tk.BOTTOM, fill=tk.X)

root.mainloop()
