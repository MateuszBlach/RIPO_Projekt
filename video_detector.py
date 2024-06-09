import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os

def process_video(file_name, choices, conf_value):
    input_file = "videos/" + file_name + ".mp4"
    output_file = "videos/" + file_name + "_out.mp4"

    model = YOLO('best.pt')

    cap = cv2.VideoCapture(input_file)

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    ret = True

    while ret:
        ret, frame = cap.read()
        if ret:
            # conf - minimum confidence threshold for detections
            # persist - adding a new ID for every newly found and tracked object
            results = model.track(frame, conf=conf_value, persist=True)

            frame = results[0].plot()

            # Write the frame into the output video file
            out.write(frame)
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()

    # Update the GUI after processing
    start_button.config(state=tk.NORMAL)
    status_label.config(text="Processing complete")

    # Show a message box and play the video
    messagebox.showinfo("Processing Complete", "The video processing is complete.")
    os.system(f'start {output_file}')  # This will open the output file using the default video player on Windows

def start_processing():
    file_name = file_name_entry.get()
    choices = []
    if pedestrian_var.get():
        choices.append(13)
    if stop_var.get():
        choices.append(14)
    if yield_var.get():
        choices.append(15)

    conf_value = conf_slider.get()

    # Disable the start button and update status
    start_button.config(state=tk.DISABLED)
    status_label.config(text="Processing...")

    # Start the video processing in a new thread
    threading.Thread(target=process_video, args=(file_name, choices, conf_value)).start()

def update_conf_label(value):
    conf_label.config(text=f"Confidence Threshold: {float(value):.1f}")

# Create the main window
root = tk.Tk()
root.title("YOLO Video Processing")

# Create a frame for the file name input
file_frame = ttk.Frame(root, padding="10")
file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

ttk.Label(file_frame, text="Enter video name (without .mp4):").grid(row=0, column=0, sticky=tk.W)
file_name_entry = ttk.Entry(file_frame, width=30)
file_name_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))

# Create a frame for the checkboxes
checkbox_frame = ttk.Frame(root, padding="10")
checkbox_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))

pedestrian_var = tk.BooleanVar()
stop_var = tk.BooleanVar()
yield_var = tk.BooleanVar()

ttk.Label(checkbox_frame, text="Wybierz znaki, które mają być wykrywane:").grid(row=0, column=0, columnspan=2, sticky=tk.W)
ttk.Checkbutton(checkbox_frame, text="Przejście dla pieszych", variable=pedestrian_var).grid(row=1, column=0, sticky=tk.W)
ttk.Checkbutton(checkbox_frame, text="Stop", variable=stop_var).grid(row=2, column=0, sticky=tk.W)
ttk.Checkbutton(checkbox_frame, text="Ustąp pierwszeństwa", variable=yield_var).grid(row=3, column=0, sticky=tk.W)

# Create a frame for the confidence slider
slider_frame = ttk.Frame(root, padding="10")
slider_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))

ttk.Label(slider_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky=tk.W)
conf_slider = ttk.Scale(slider_frame, from_=0, to=1.0, orient=tk.HORIZONTAL, command=update_conf_label)
conf_slider.set(0.2)  # Default value
conf_slider.grid(row=0, column=1, sticky=(tk.W, tk.E))

conf_label = ttk.Label(slider_frame, text="Confidence Threshold: 0.2")
conf_label.grid(row=1, column=0, columnspan=2, sticky=tk.W)

# Create the start button and status label
start_button = ttk.Button(root, text="Start", command=start_processing)
start_button.grid(row=3, column=0, pady=10)

status_label = ttk.Label(root, text="")
status_label.grid(row=4, column=0, pady=10)

# Run the application
root.mainloop()
