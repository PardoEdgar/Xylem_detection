import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import os
from pathlib import Path
BG = "#0f1117"
PANEL = "#1a1d27"
CARD = "#22263a"
ACCENT = "#00d4aa"
TEXT = "#e8eaf0"
MUTED = "#6b7280"

class ROIselector:

    def __init__(self, root):
        self.root = root
        self.root.title("ROI Selector")
        container = tk.Frame(root, bg=PANEL, height=50)
        container.pack(fill="x", expand=True)
        tk.Label(container, text="ROI Analyzer",
            bg=PANEL, fg=ACCENT,
            font=("Segoe UI", 14, "bold")).pack(padx=10, pady=10, anchor="w")

        main = tk.Frame(self.root, bg=BG, height=50)
        main.pack(fill="both", expand=True)
        # Canvas izquierda (imagen)
        self.canvas = tk.Canvas(container, cursor="cross", bg="black")
        self.canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Canvas i (ROI)
        self.roi_canvas = tk.Canvas(container, bg="gray")
        self.roi_canvas.pack(side="right", fill="both", expand=True, padx=5, pady=5)


        self.btn = tk.Button(root, text="Load Image",bg=ACCENT,fg="black",font=("Segoe UI", 10, "bold" ), relief="flat",
                                padx=40, pady=5,cursor="hand2", command=self.load_image)
        self.btn.pack(side="left",pady=10, padx=40, fill="x")

        self.btn_2 = tk.Button(root, text="save Image",bg=ACCENT,fg="black",font=("Segoe UI", 10, "bold" ), relief="flat",
                                padx=40, pady=5,cursor="hand2", command=self.save_ROI)
        self.btn_2.pack(side="right",pady=10, padx=40, fill="x")


        self.image = None
        self.tk_image = None

        self.start_x = None
        self.start_y = None
        self.rect = None

        self.current_roi= None

        # Eventos
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.image = img
        self.display_image(img)
        self.image_path = path

    def display_image(self, img):
        self_canvas_w = self.canvas.winfo_width()
        self_canvas_h = self.canvas.winfo_height()
        
        h, w = img.shape[:2]

        scale = min(self_canvas_w / w, self_canvas_h / h, 1)

        new_w  = int(w * scale)
        new_h  = int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self.display_scale = scale

        self.img_offset_x = (self_canvas_w - new_w) // 2
        self.img_offset_y = (self_canvas_h - new_h) // 2

        pil_img = Image.fromarray(resized)
        self.tk_image = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")   # limpia antes de redibujar
        self.canvas.create_image(self.img_offset_x, self.img_offset_y,
                                  anchor="nw", image=self.tk_image)


    def on_click(self, event):
        self.start_x = event.x
        self.start_y = event.y

        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y,
            self.start_x, self.start_y,
            outline="red"
        )

    def on_drag(self, event):
        self.canvas.coords(
            self.rect,
            self.start_x, self.start_y,
            event.x, event.y
        )

    def on_release(self, event):
        end_x, end_y = event.x, event.y
        x1 = int((min(self.start_x, end_x) - self.img_offset_x) / self.display_scale)
        y1 = int((min(self.start_y, end_y) - self.img_offset_y) / self.display_scale)
        x2 = int((max(self.start_x, end_x) - self.img_offset_x) / self.display_scale)
        y2 = int((max(self.start_y, end_y) - self.img_offset_y) / self.display_scale)

        
        h, w = self.image.shape[:2]
        
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        roi = self.image[y1:y2, x1:x2]
        self.current_roi = roi
        self.show_roi(roi)

    def show_roi(self, roi):
        roi_canvas_w = self.roi_canvas.winfo_width()
        roi_canvas_h = self.roi_canvas.winfo_height()
        
        h, w = roi.shape[:2]

        roi_scale = min(roi_canvas_w / w, roi_canvas_h / h, 1)
        new_w = int(w * roi_scale)
        new_h = int(h * roi_scale)
        
        resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_img = Image.fromarray(resized)
        self.tk_roi = ImageTk.PhotoImage(pil_img)
        self.roi_canvas.delete("all")

        cx = roi_canvas_w // 2
        cy = roi_canvas_h // 2
        self.roi_canvas.create_image(cx, cy, anchor="center", image=self.tk_roi)

    def save_ROI(self):
        name = Path(self.image_path).stem
        directory = r"C:\Users\jandr\OneDrive - Universidad del rosario\Gui_xylem\ROIs"
        os.makedirs(directory, exist_ok=True)
        filepath = f"{name}.tiff"
        filepath = os.path.join(directory, filepath)
        cv2.imwrite(filepath, self.current_roi)
        print(f"image{name} saved")
        
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x660")
    app = ROIselector(root)
    root.mainloop()
