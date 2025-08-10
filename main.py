import io, os, json, time, threading, hashlib
from typing import Optional, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pillow_heif
import numpy as np
import cv2
import piexif
from skimage.metrics import structural_similarity as ssim_fn

RETENTION_SECONDS = 2*60*60
TMP_DIR = "/tmp/wandtool"
os.makedirs(TMP_DIR, exist_ok=True)

app = FastAPI(title="WandTool API", version="1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

def _cleanup_loop():
    while True:
        try:
            now = time.time()
            for fn in os.listdir(TMP_DIR):
                p = os.path.join(TMP_DIR, fn)
                try:
                    if now - os.stat(p).st_mtime > RETENTION_SECONDS:
                        os.remove(p)
                except: pass
        except: pass
        time.sleep(300)
threading.Thread(target=_cleanup_loop, daemon=True).start()

def read_image_preserve_exif(b: bytes):
    # HEIC
    try:
        heif = pillow_heif.read_heif(b)
        img = Image.frombytes(heif.mode, heif.size, heif.data, "raw", heif.mode, heif.stride)
        meta = {"format": "HEIC", "exif": heif.metadata or {}}
        if heif.color_profile and heif.color_profile.get("icc_profile"):
            img.info["icc_profile"] = heif.color_profile["icc_profile"]
        return img, meta
    except Exception:
        pass
    # JPEG/PNG/…
    img = Image.open(io.BytesIO(b)); img.load()
    exif = img.info.get("exif", b"")
    icc  = img.info.get("icc_profile", None)
    if icc: img.info["icc_profile"] = icc
    return img, {"format": img.format or "JPEG", "exif": exif}

def strip_gps(meta: dict):
    fmt = (meta.get("format") or "").upper()
    if fmt in ["JPEG","JPG","JPE"] and isinstance(meta.get("exif"), (bytes, bytearray)):
        try:
            ex = piexif.load(meta["exif"]); ex["GPS"] = {}; meta["exif"] = piexif.dump(ex)
        except: meta["exif"] = b""
    elif fmt in ["HEIC","HEIF"] and isinstance(meta.get("exif"), dict):
        for k in list(meta["exif"].keys()):
            if "gps" in k.lower(): meta["exif"].pop(k, None)
    return meta

def pil2bgr(img: Image.Image): return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
def bgr2pil(arr: np.ndarray):  return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def feather_in(mask: np.ndarray, r=2):
    if r<=0: return (mask.astype(np.uint8)*255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r*2+1, r*2+1))
    inner = cv2.erode(mask.astype(np.uint8), k, 1)
    blur  = cv2.GaussianBlur(inner, (r*2+1, r*2+1), 0)
    return cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)

def wall_tex(h, w, base=(245,245,245), grain=2, drift=2, seed=42):
    rng = np.random.default_rng(seed)
    wall = np.ones((h,w,3), np.uint8); wall[:] = np.array(base, np.uint8)
    noise = rng.integers(-grain, grain+1, size=(h,w,1), dtype=np.int16)
    wall = np.clip(wall.astype(np.int16)+noise, 0,255)
    v = (np.linspace(-drift, drift, h).reshape(h,1,1)).astype(np.int16)
    wall = np.clip(wall+v, 0,255).astype(np.uint8)
    return wall

def compose(original_bgr, mask_bin, base_color, feather_px=2):
    h,w = original_bgr.shape[:2]
    wall = wall_tex(h,w, base=base_color, grain=2, drift=2, seed=42)
    m = (feather_in(mask_bin, feather_px).astype(np.float32))/255.0
    m3 = np.dstack([m]*3)
    out = original_bgr.astype(np.float32)*(1-m3) + wall.astype(np.float32)*m3
    return out.astype(np.uint8)

def sample_wall_color(bgr, mask_bin):
    h,w = bgr.shape[:2]
    top = bgr[:h//2]
    inv = (cv2.resize(mask_bin, (w,h), interpolation=cv2.INTER_NEAREST)[:h//2]==0)
    pix = top[inv]
    if pix.size==0: return (245,245,245)
    lab = cv2.cvtColor(pix.reshape(-1,1,3), cv2.COLOR_BGR2LAB).reshape(-1,3)
    L = int(np.clip(np.median(lab[:,0]), 230, 250))
    a = int(round((np.median(lab[:,1])-128)*0.2 + 128))
    b = int(round((np.median(lab[:,2])-128)*0.2 + 128))
    bgr = cv2.cvtColor(np.uint8([[[L,a,b]]]), cv2.COLOR_LAB2BGR)[0,0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

def detect_auto_mask(bgr):
    h,w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Tischkante
    lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=150,minLineLength=w//3,maxLineGap=10)
    desk_y = int(h*0.6)
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            if abs(y1-y2)<4 and y1>h*0.3: desk_y = min(desk_y,int((y1+y2)//2))
    wall_region = np.zeros((h,w),np.uint8); wall_region[:desk_y,:]=1
    # Kabel/Gerät
    cable = (edges>0).astype(np.uint8)
    device = cv2.morphologyEx((gray<40).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
    # Produktschutz (Textur)
    sobelx = cv2.Sobel(gray, cv2.CV_32F,1,0,3); sobely = cv2.Sobel(gray, cv2.CV_32F,0,1,3)
    mag = cv2.magnitude(sobelx, sobely)
    texture = (mag>60).astype(np.uint8)
    protect = cv2.dilate(texture, cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)),1)
    # 6 px Puffer
    dist = cv2.distanceTransform((protect==0).astype(np.uint8), cv2.DIST_L2,3)
    buffer6 = (dist<=6).astype(np.uint8)
    # Maske
    mask = (wall_region & ((cable>0)|(device>0))).astype(np.uint8)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)),1)
    mask[protect>0]=0; mask[buffer6>0]=0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)),1)
    return mask

def psnr(a,b):
    a=a.astype(np.float32); b=b.astype(np.float32)
    mse = np.mean((a-b)**2); 
    if mse==0: return 99.0
    return 20*np.log10(255.0/np.sqrt(mse))

def write_with_meta(img: Image.Image, meta: dict, prefer: Optional[str], quality=95):
    out = io.BytesIO(); fmt = (prefer or meta.get("format") or "JPEG").upper()
    if fmt in ["HEIC","HEIF"]:
        pillow_heif.write_heif(img, out, quality=quality, save_exif=meta.get("exif"),
                               color_profile=("icc", img.info.get("icc_profile")) if img.info.get("icc_profile") else None)
        return out.getvalue(), fmt
    if fmt in ["JPEG","JPG","JPE"]:
        img.save(out, format="JPEG", quality=quality, subsampling=0, optimize=True,
                 exif=meta.get("exif", b""), icc_profile=img.info.get("icc_profile"))
        return out.getvalue(), "JPEG"
    if fmt=="PNG":
        img.save(out, format="PNG", icc_profile=img.info.get("icc_profile")); return out.getvalue(), "PNG"
    img.save(out, format="JPEG", quality=quality, subsampling=0, optimize=True,
             exif=meta.get("exif", b""), icc_profile=img.info.get("icc_profile"))
    return out.getvalue(), "JPEG"

@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)):
    data = await image.read()
    pil, meta = read_image_preserve_exif(data)
    bgr = pil2bgr(pil)
    mask = detect_auto_mask(bgr)
    buf = io.BytesIO()
    Image.fromarray((mask*255).astype(np.uint8)).save(buf, format="PNG"); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/api/render")
async def render(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    quality: int = Form(95),
    feather_px: int = Form(2),
    remove_gps: bool = Form(True),
):
    img_bytes = await image.read()
    mask_bytes = await mask.read()
    pil, meta = read_image_preserve_exif(img_bytes)
    if remove_gps: meta = strip_gps(meta)
    bgr = pil2bgr(pil)
    m = (np.array(Image.open(io.BytesIO(mask_bytes)).convert("L").resize(pil.size, Image.NEAREST))>127).astype(np.uint8)
    base = sample_wall_color(bgr, m)
    out_bgr = compose(bgr, m, base, feather_px=feather_px)
    out_pil = bgr2pil(out_bgr)
    if "icc_profile" in pil.info: out_pil.info["icc_profile"] = pil.info["icc_profile"]
    out_bytes, out_fmt = write_with_meta(out_pil, meta, prefer=meta.get("format"), quality=quality)
    # Integritätscheck ausserhalb Maske
    ob = pil2bgr(Image.open(io.BytesIO(out_bytes)))
    inv = (m==0)
    ps = psnr(bgr[inv], ob[inv])
    if ps < 50.0:
        # PNG-Fallback
        buf = io.BytesIO(); out_pil.save(buf, format="PNG"); out_bytes = buf.getvalue(); out_fmt = "PNG"
    return StreamingResponse(io.BytesIO(out_bytes), media_type="application/octet-stream",
                             headers={"X-Wandtool-Report": json.dumps({
                                 "width": pil.size[0], "height": pil.size[1],
                                 "format_in": meta.get("format"), "format_out": out_fmt
                             })})
