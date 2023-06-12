import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter.filedialog import askopenfilename, askopenfilenames

source = ''
frame_counter = 1

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

def set_frame_counter(value):
    global frame_counter
    frame_counter = int(value)
    return frame_counter

def give_frame_counter():
    return frame_counter

# This is the section of code which creates buttons
path_btn = Button(root, text='Select Input video', bg='#E0EEEE', font=('arial', 12, 'normal'), command=lambda: pathdef(lable1))
path_btn.pack(pady=10, fill=tk.X)

alt_path_def = Button(root, text='Select Camera as Input', bg='#E0EEEE', font=('arial', 12, 'normal'), command=lambda:set_source_camera(lable1))
alt_path_def.pack(pady=10, fill=tk.X)

frame_counter_label = Label(root, text='Frame Counter:', bg='#C0C0C0', font=('arial', 12, 'normal'))
frame_counter_label.pack(pady=(10, 0), fill=tk.X)

frame_counter_value = tk.StringVar()
frame_counter_value.set("1")  # Set default value to 1

frame_counter_dropdown = ttk.OptionMenu(root, frame_counter_value, "1", "1", "2", "5", "10", command=set_frame_counter)
frame_counter_dropdown.pack(pady=5, fill=tk.X)

# This is the section of code which creates an image
main_page_canvas = tk.Canvas(root, width=windowWidth*0.5, height=windowHeight*0.5)
main_page_canvas.place(relx=0.5, rely=0.5, anchor=CENTER)

main_page_image = PhotoImage(file='sd.gif')
main_page_image_width = main_page_image.width()
main_page_image_height = main_page_image.height()
main_page_canvas.create_image(0, 0, anchor=NW, image=main_page_image)

def give_source():
    return source

def give_frame_counter():
    return frame_counter

def closewin():
    root.quit()

start_btn = Button(root, text='Start', bg='#ceffcf', font=('arial', 12, 'normal'), command=closewin)
start_btn.pack(pady=(10, 30), side=tk.BOTTOM, fill=tk.X)

root.mainloop()
