from django.shortcuts import render
import keras
from PIL import Image
import numpy as np 
import os 
from django.core.files.storage import FileSystemStorage


# Create your views here.
media='media'
model=keras.models.load_model('EfficientNetB0_model.h5')
def makepredictions(path):
    img = Image.open(path)
    img_d = img.resize((224, 224))  # Resize the image to match the expected input shape
    if len(np.array(img_d).shape) < 3:
        rgb_img = Image.new("RGB", img_d.size)
        rgb_img.paste(img_d)
    else:
        rgb_img = img_d

    rgb_img = np.array(rgb_img, dtype=np.float64)
    rgb_img = rgb_img.reshape(1, 224, 224, 3)  # Corrected reshape dimensions
    predictions = model.predict(rgb_img)
    
    # Convert predicted probabilities to class
    predicted_class = int(np.argmax(predictions))
    
    classes = [
        "Normal",
        "Diabetic Retinopathy",
        "Glaucoma",
        "Not an eye Image",
        "AMD",
        "Hypertension",
        "Myopia",
        "Others"
    ]
    
    if 0 <= predicted_class < len(classes):
        predicted_disease = f"Predicted Disease: {classes[predicted_class]}"
    else:
        predicted_disease = "Predicted Disease: Unknown"

    return predicted_disease


def index(request):
    if request.method == "POST" and request.FILES['upload']:
        if 'upload' not in request.FILES:
            err='No images selected'
            return render(request,'index.html',{'err':err})
        f = request.FILES['upload']
        if f == '':
            err='No files selected'
            return render(request,'index.html',{'err':err})
        upload=request.FILES['upload']
        fss= FileSystemStorage()
        file=fss.save(upload.name,upload)
        file_url=fss.url(file)
        predictions=makepredictions(os.path.join(media,file))
        return render(request,'index.html',{'pred':predictions, 'file_url':file_url})
    
    else:
        return render(request,'index.html')
