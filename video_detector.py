import cv2
import pygame
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import time

pygame.init()
pygame.mixer.init()
pedestrian_sound = pygame.mixer.Sound("assets/sounds/przejscie.mp3")
stop_sound = pygame.mixer.Sound("assets/sounds/stop.mp3")
yield_sound = pygame.mixer.Sound("assets/sounds/ustap.mp3")
ped_time = 0
stop_time = 0
yield_time = 0
global_time = 0


def play_pedestrian():
    global ped_time
    global global_time
    current_time = time.time()
    if current_time - ped_time >= 5 and current_time - global_time >= 1:
        pygame.mixer.Sound.play(pedestrian_sound)
        ped_time = current_time
        global_time = current_time


def play_stop():
    global stop_time
    global global_time
    current_time = time.time()
    if current_time - stop_time >= 5 and current_time - global_time >= 1:
        pygame.mixer.Sound.play(stop_sound)
        stop_time = current_time
        global_time = current_time


def play_yield():
    global yield_time
    global global_time
    current_time = time.time()
    if current_time - yield_time >= 5 and current_time - global_time >= 1:
        pygame.mixer.Sound.play(yield_sound)
        yield_time = current_time
        global_time = current_time


def process_video(file_name, choices, conf_value, live):
    input_file = file_name
    if not live:
        output_file = "output/" + os.path.splitext(os.path.basename(file_name))[0] + "_out.mp4"

    model = YOLO('best.pt')
    model.predict(classes=choices)

    if live:
        cv2.namedWindow('Wyjście', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Wyjście', 1280, 960)

    cap = cv2.VideoCapture(input_file)

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not live:
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
            if live and sound_var.get():
                for r in results:
                    if len(r.boxes.cls) > 0:
                        for box in r.boxes.cls:
                            dclass = box.item()
                            if dclass == 13.0:
                                play_pedestrian()
                            elif dclass == 14.0:
                                play_stop()
                            if dclass == 15.0:
                                play_yield()

            frame = results[0].plot()
            if live:
                cv2.imshow('Wyjście', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Write the frame into the output video file
            if not live:
                out.write(frame)
        else:
            break

    # Release everything if job is finished
    cap.release()
    if not live:
        out.release()
    if live:
        cv2.destroyAllWindows()

    # Update the GUI after processing
    start_button.config(state=tk.NORMAL)
    status_label.config(text="Przetwarzanie zakończone")

    # Show a message box and play the video
    messagebox.showinfo("Przetwarzanie zakończone", "Przetwarzanie filmu zostało zakończone.")
    if not live:
        os.system(f'start {output_file}')  # This will open the output file using the default video player on Windows


def start_processing(live):
    file_name = file_name_entry
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
    status_label.config(text="Przetwarzanie...")

    # Start the video processing in a new thread
    threading.Thread(target=process_video, args=(file_name, choices, conf_value, live)).start()


def update_conf_label(value):
    conf_label.config(text=f"Aktualna wartość: {float(value):.1f}")


def select_file():
    global file_name_entry
    filename = filedialog.askopenfilename(title='Wybierz plik',
                                          filetypes=(('Pliki wideo', '*.mp4'), ('Wszystkie pliki', '*.*')))
    file_name_entry = filename
    file_label.config(text=os.path.basename(filename))


# Create the main window
root = tk.Tk()
root.title("Wykrywanie znaków")

# Create a frame for the file name input
file_frame = ttk.Frame(root, padding="10")
file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

ttk.Button(file_frame, command=select_file, text='Wybierz plik').grid(row=0, column=0, sticky=(tk.W, tk.E))
file_name_entry = ""
file_label = ttk.Label(file_frame, text=file_name_entry)
file_label.grid(row=0, column=1, sticky=tk.W)

# Create a frame for the checkboxes
checkbox_frame = ttk.Frame(root, padding="10")
checkbox_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))

pedestrian_var = tk.BooleanVar()
stop_var = tk.BooleanVar()
yield_var = tk.BooleanVar()

ttk.Label(checkbox_frame, text="Wybierz znaki, które mają być wykrywane:").grid(row=0, column=0, columnspan=2,
                                                                                sticky=tk.W)
ttk.Checkbutton(checkbox_frame, text="Przejście dla pieszych", variable=pedestrian_var).grid(row=1, column=0,
                                                                                             sticky=tk.W)
ttk.Checkbutton(checkbox_frame, text="Stop", variable=stop_var).grid(row=2, column=0, sticky=tk.W)
ttk.Checkbutton(checkbox_frame, text="Ustąp pierwszeństwa", variable=yield_var).grid(row=3, column=0, sticky=tk.W)

# Create a frame for the confidence slider
slider_frame = ttk.Frame(root, padding="10")
slider_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))

ttk.Label(slider_frame, text="Dokładność wykrywania:").grid(row=0, column=0, sticky=tk.W)
conf_slider = ttk.Scale(slider_frame, from_=0, to=1.0, orient=tk.HORIZONTAL, command=update_conf_label)
conf_slider.set(0.2)  # Default value
conf_slider.grid(row=0, column=1, sticky=(tk.W, tk.E))

conf_label = ttk.Label(slider_frame, text="Aktualna wartość: 0.2")
conf_label.grid(row=1, column=0, columnspan=2, sticky=tk.W)

sound_var = tk.BooleanVar()
ttk.Checkbutton(root, text="Powiadomienia dźwiękowe (tylko dla trybu na żywo)", variable=sound_var).grid(row=3,
                                                                                                         column=0,
                                                                                                         sticky=tk.W,
                                                                                                         pady=10)

# Create the start button and status label
start_button = ttk.Button(root, text="Zapisz do pliku", command=lambda: start_processing(False))
start_button.grid(row=4, column=0, pady=10)
start_button2 = ttk.Button(root, text="Wykrywanie na żywo", command=lambda: start_processing(True))
start_button2.grid(row=5, column=0, pady=10)

status_label = ttk.Label(root, text="")
status_label.grid(row=6, column=0, pady=10)

# Run the application
root.mainloop()
