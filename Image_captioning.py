import tkinter as tk
from PIL import ImageTk, Image
from transformers import BlipProcessor, BlipForConditionalGeneration

window = tk.Tk()
window.title("Image Captioning")
window.geometry("600x550")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
image_path = tk.StringVar()
heading_label = tk.Label(window, text="Image Captioning", font=("Arial", 16, "bold"))
heading_label.pack(side="top", pady=20)
caption_label = tk.Label(window, text="Caption will be displayed here", font=("Arial", 12))
image_label = tk.Label(window)

def process_image():
    img_path = image_path.get()
    try:
        raw_image = Image.open(img_path).convert('RGB')
        resized_image = raw_image.resize((250, 250))
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)


        caption = processor.decode(out[0], skip_special_tokens=True)
        caption_label.config(text=caption)
        image = ImageTk.PhotoImage(resized_image)
        image_label.configure(image=image)
        image_label.image = image
    except:
        caption_label.config(text="Error processing the image")

def browse_image():
    from tkinter import filedialog
    file_path = filedialog.askopenfilename(filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
    image_path.set(file_path)

image_entry = tk.Entry(window, textvariable=image_path, width=40, font=("Arial", 12))
image_entry.pack(pady=20)
browse_button = tk.Button(window, text="Browse", command=browse_image)
browse_button.pack()
process_button = tk.Button(window, text="Generate Caption", command=process_image)
process_button.pack(pady=10)
image_label.pack()
caption_label.pack(pady=20)
window.mainloop()
