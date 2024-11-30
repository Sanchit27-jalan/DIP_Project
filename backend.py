import os
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import FastSAM
from scipy import stats
import matplotlib.pyplot as plt
import zxing
import numpy as np
from fastapi import FastAPI, File, UploadFile
from zbar import Scanner,zbar
from fastapi.middleware.cors import CORSMiddleware
import io
import uvicorn
import uuid
import shutil

def check_barcode(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ddepth = cv2.CV_32F 
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.blur(gradient, (9, 9))
    _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return 0
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return 1


def check_qr_code(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    potential_qr = []

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            potential_qr.append(approx)

    if len(potential_qr) >= 3:
        return True

    return False

def qr_code(img_path):
    scanner = Scanner()
    pil_image = Image.open(img_path)
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')

    image_np = np.array(pil_image)

    results = scanner.scan(image_np)
    qr_result=[]
    for result in results:
        qr_result.append(result.data.decode('ascii'))
    if qr_result==[]:
        return False
    return qr_result[0]

def extract_segments(image_path):
    output_dir = "./segments"
    os.makedirs(output_dir, exist_ok=True)
    model = FastSAM('./dependencies/FastSAM-s.pt')
    original_image = Image.open(image_path)
    image_np = np.array(original_image)
    image_area = original_image.size[0] * original_image.size[1]
    results = model(original_image, device='cuda' if torch.cuda.is_available() else 'cpu',
                    retina_masks=True, conf=0.4, iou=0.9)
    annotations = results[0].masks.data
    if annotations is None:
        print("No segments detected")
        return []
    segments = []
    for idx, annotation in enumerate(annotations):
        mask = annotation.cpu().numpy()
        mask = cv2.resize(mask, (original_image.size[0], original_image.size[1]))
        mask = mask > 0.5
        segment = np.zeros_like(image_np)
        
        segment[mask] = image_np[mask]
        
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            bbox_area = (x_max - x_min + 1) * (y_max - y_min + 1)
            if bbox_area < 0.05 * image_area or bbox_area > 0.9 * image_area:
                print(f"Skipping segment {idx} due to size constraints")
                continue
    
            cropped_segment = segment[y_min:y_max+1, x_min:x_max+1]
            if(check_barcode(cropped_segment)==0 and check_qr_code(cropped_segment)==False):
                print(f"Skipping segment {idx} due to segmenation problem")
                continue
            cv2.imwrite(f"{output_dir}/segment_{idx}.png", cv2.cvtColor(cropped_segment, cv2.COLOR_RGB2BGR))
            segments.append({'index': idx, 'mask': mask, 'cropped_segment': cropped_segment})
    
    return segments


def check_horizontal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 300)
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # plt.plot((x1, x2), (y1, y2), 'r')
        angles = [np.degrees(theta) for rho, theta in lines[:, 0]]
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        filtered_angles = [angle for angle in angles if abs(angle - mean_angle) < 2 * std_angle]
        angles = filtered_angles
        print("Mean of angles is: ", np.mean(angles))
        if(abs(np.mean(angles) - 90) < 5 or abs(np.mean(angles) - 270) < 5):
            return img
        else:
            angle = np.mean(angles) + 180  
            if(np.mean(angles) < 90):
                angle = angle + 180
        print(f"Detected angle: {angle}")

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(img, M, (w, h))
    else:
        print("No rotation detected")
        rotated_img = img  
    return rotated_img

def decode_qr_code(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Could not load image."
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pil_image = Image.fromarray(gray)
    scanner = zbar.Scanner()
    results = scanner.scan(pil_image)
    if not results:
        return False
    
    decoded_data = []
    for result in results:
        code_data = result.data.decode('utf-8')
        code_type = result.type
        decoded_data.append(code_data)
    return decoded_data

def decode(image_path):
    input_dir = "./segments"
    output_dir = "./rotated_segments"
    os.makedirs(output_dir, exist_ok=True)
    barcode_ans=[]
    qr_code_ans=[]
    reader = zxing.BarCodeReader()
    qrr=qr_code(image_path)
    if qrr:
        qr_code_ans.append(qrr)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_dir, filename)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rotated_img = check_horizontal(img)
            output_path = os.path.join(output_dir, f"rotated_{filename}")
            cv2.imwrite(output_path, cv2.cvtColor(rotated_img, cv2.COLOR_RGB2BGR))
            barcode = reader.decode(output_path)
            if barcode:
                barcode_ans.append(barcode.raw)
    return barcode_ans,qr_code_ans



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)
def handle_remove_error(func, path, exc_info):
    print(f"Error removing {path}: {exc_info}")
    os.chmod(path, 0o777) 
    func(path)

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        width, height = image.size
        image_format = image.format
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        save_path = os.path.join("./images", unique_filename)
        image.save(save_path)
        extract_segments(save_path)
        barcode_ans,qr_code=decode(save_path)
        answers=[]
        for i in barcode_ans:
            if i not in answers:
                answers.append(i)
        for i in qr_code:
            if i not in answers:
                answers.append(i)
        directory = "./segments"
        shutil.rmtree(directory, onerror=handle_remove_error)
        directory = "./rotated_segments"
        shutil.rmtree(directory, onerror=handle_remove_error)
        return {
            "decoded_results": answers,
        }
    
    except Exception as e:
        return {
            "status": "error", 
            "message": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)