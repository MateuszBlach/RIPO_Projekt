import cv2
from ultralytics import YOLO

file_name = input("Enter video name (without .mp4): ")
input_file = "videos/" + file_name + ".mp4"
output_file = "videos/" + file_name + "_out.mp4"

model = YOLO('best.pt')
model.predict(classes=[13])

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
        results = model.track(frame, conf=0.7, persist=True)

        frame = results[0].plot()

        # Write the frame into the output video file
        out.write(frame)

        # Uncomment this if you want to display the processed frame
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(25) & 0xFF == ord('q'):  # press q to quit
        #     break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()

# Close all windows if you uncommented imshow
# cv2.destroyAllWindows()
