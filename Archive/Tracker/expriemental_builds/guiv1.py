import tkinter as tk
from tkinter import ttk
from tkinter import * 
from tkinter.filedialog import askopenfilename


source = ''
root = tk.Tk()

# This is the section of code which creates the main window
root.geometry('400x600')
root.configure(background='#fbf4d1')
root.title('opServator')


# This is the section of code which creates the a label
lable1 = tk.Label(root, text="", bg='#F0FFFF', font=('arial', 12, 'normal'))
lable1.place(y=50)
lable1.pack(padx=20, pady=10)

def pathdef(l):
    
    VIDEO_PATH = askopenfilename()
    global source
    source = VIDEO_PATH
    l.config(text = source)
    #give_source(source)
    #print(source)
    return source

def set_source_camera(l):
    global source
    source = 0
    l.config(text = "Camera was selected as input")
    return source



# This is the section of code which creates a button
path_btn = Button(root, text='Select Input video', bg='#E0EEEE', font=('arial', 12, 'normal'), command=lambda: pathdef(lable1))
path_btn.place(y=100)
path_btn.pack(padx=20, pady=10)
alt_path_def = Button(root, text='Select Camera as Input', bg='#E0EEEE', font=('arial', 12, 'normal'), command=lambda:set_source_camera(lable1))
alt_path_def.place(y=100)
alt_path_def.pack(padx=20, pady=10)

main_page_canvas = tk.Canvas(root, height=100, width=150)
main_page_image = PhotoImage(file='main3.gif')
main_page_canvas.create_image(75, 50, image=main_page_image)
main_page_canvas.pack(padx=20, pady=60)

def give_source():
    return source

def closewin():
    root.quit()
start_btn = Button(root, text= 'Start',bg='#ceffcf', font=('arial', 12, 'normal'),command=closewin)
start_btn.pack(padx=5, pady=80,side=tk.BOTTOM)

root.mainloop()



