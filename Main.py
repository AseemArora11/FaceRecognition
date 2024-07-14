import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, Canvas
from PIL import Image, ImageTk

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")
        
        # Create GUI elements
        self.load_image_button = tk.Button(self.root, text="Load Image", command=self.load_image)
        self.load_image_button.pack(pady=10)
        
        self.camera_button = tk.Button(self.root, text="Open Camera", command=self.open_camera)
        self.camera_button.pack(pady=10)
        
        self.canvas = Canvas(self.root, width=640, height=480)
        self.canvas.pack(pady=10)
        
        self.status_label = tk.Label(self.root, text="", fg="blue")
        self.status_label.pack(pady=10)
        
        self.image = None  # To store loaded image or camera frame
        
        # Initialize camera capture
        self.capture = None
    
    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.capture = None  # Release camera if loaded from image
            self.image = cv2.imread(file_path)
            self.display_image()
            self.detect_and_display_faces()
    
    def open_camera(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            messagebox.showerror("Error", "Failed to open camera.")
            return
        self.image = None  # Clear image if switching to camera
        self.display_camera()
    
    def display_image(self):
        if self.image is not None:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image = Image.fromarray(self.image)
            self.image = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
    
    def display_camera(self):
        if self.capture.isOpened():
            _, frame = self.capture.read()
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                self.image = ImageTk.PhotoImage(frame)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
                self.detect_and_display_faces()
                self.root.after(10, self.display_camera)
            else:
                messagebox.showwarning("Warning", "Failed to capture frame from camera.")
        else:
            messagebox.showerror("Error", "Failed to open camera.")
    
    def detect_and_display_faces(self):
        if self.image is not None:
            # Load the Haar Cascade classifier for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert image to grayscale
            gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            
            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(self.image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Update displayed image with detected faces
            self.display_image()
            
            # Update status label
            self.status_label.config(text=f"{len(faces)} face(s) detected")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
