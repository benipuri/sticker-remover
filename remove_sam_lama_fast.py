import os
import sys
import shutil
import cv2
import numpy as np
import torch

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from simple_lama_inpainting import SimpleLama

# ================== CONFIG / CONSTANTS ==================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "/app/best.pt")
SAM_CHECKPOINT  = os.getenv("SAM_CHECKPOINT", "/app/sam_vit_b_01ec64.pth")

IMGSZ = int(os.getenv("YOLO_IMGSZ", "1024"))
CONF  = float(os.getenv("YOLO_CONF", "0.30"))

SAM_MAX_SIDE = int(os.getenv("SAM_MAX_SIDE", "1024"))

ERODE_PX   = int(os.getenv("ERODE_PX", "1"))
DILATE_PX  = int(os.getenv("DILATE_PX", "3"))
FEATHER_PX = int(os.getenv("FEATHER_PX", "7"))

TELEA_RADIUS = int(os.getenv("TELEA_RADIUS", "3"))
NS_RADIUS    = int(os.getenv("NS_RADIUS", "3"))

MAX_ROI_PIXELS  = int(os.getenv("MAX_ROI_PIXELS", str(800*800)))
SMALL_MASK_AREA = int(os.getenv("SMALL_MASK_AREA", "64"))
ROI_PAD_PX      = int(os.getenv("ROI_PAD_PX", "10"))

INPAINT_METHOD = os.getenv("INPAINT_METHOD", "lama")

# ========= LAZY GLOBALS (initially None) =========
yolo = None
sam = None
predictor = None
lama_model = None

# ========= LAZY LOADER =========
def load_models_once():
    global yolo, sam, predictor, lama_model

    if yolo is not None:
        return  # Already loaded

    print("========== LOADING MODELS (first request) ==========")

    # YOLO
    print(f"Loading YOLO from {YOLO_MODEL_PATH}...")
    yolo_m = YOLO(YOLO_MODEL_PATH)
    yolo_m.fuse()
    yolo_m.to(DEVICE)
    yolo_m.model.eval()

    # SAM
    print(f"Loading SAM VIT-B from {SAM_CHECKPOINT}...")
    sam_m = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam_m.to(DEVICE)
    sam_m.eval()
    predictor_m = SamPredictor(sam_m)

    # LaMA
    print("Loading LaMA inpainter...")
    lama_m = SimpleLama(device=DEVICE)

    # Assign once
    yolo = yolo_m
    sam = sam_m
    predictor = predictor_m
    lama_model = lama_m

    print("========== MODELS LOADED SUCCESSFULLY ==========")


# ---------------------------------------------------------------------
#   ALL OTHER CODE BELOW THIS POINT IS IDENTICAL TO YOUR ORIGINAL VERSION
# ---------------------------------------------------------------------

def _tight_feather(mask: np.ndarray) -> np.ndarray:
    m = mask.copy()
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)
    if ERODE_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_PX*2+1, ERODE_PX*2+1))
        m = cv2.erode(m, k, 1)
    if DILATE_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_PX*2+1, DILATE_PX*2+1))
        m = cv2.dilate(m, k, 1)
    if FEATHER_PX > 0:
        m = cv2.GaussianBlur(m, (FEATHER_PX*2+1, FEATHER_PX*2+1), 0)
    return m


def telea_inpaint(img_bgr: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
    mask = _tight_feather(raw_mask)
    return cv2.inpaint(img_bgr, mask, TELEA_RADIUS, cv2.INPAINT_TELEA)


def ns_inpaint(img_bgr: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
    mask = _tight_feather(raw_mask)
    return cv2.inpaint(img_bgr, mask, NS_RADIUS, cv2.INPAINT_NS)


def lama_inpaint(img_bgr: np.ndarray, raw_mask: np.ndarray) -> np.ndarray:
    mask = _tight_feather(raw_mask)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = (mask > 0).astype(np.uint8) * 255
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with torch.inference_mode():
        out = lama_model(img_rgb, mask)

    if hasattr(out, "mode"):
        out = np.array(out)

    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def smart_inpaint(img: np.ndarray, raw_mask: np.ndarray, method: str = "lama") -> np.ndarray:
    if method == "lama":
        return lama_inpaint(img, raw_mask)
    if method == "telea":
        return telea_inpaint(img, raw_mask)
    if method == "ns":
        return ns_inpaint(img, raw_mask)
    return telea_inpaint(img, raw_mask)


def _resize_for_sam(img_bgr: np.ndarray, max_side: int = SAM_MAX_SIDE):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    scale = min(max_side / max(H, W), 1.0)
    if scale < 1.0:
        newW = int(W * scale)
        newH = int(H * scale)
        small = cv2.resize(img_rgb, (newW, newH), interpolation=cv2.INTER_AREA)
        return small, scale
    return img_rgb, 1.0


def _upsample_mask_bool(mask_small: np.ndarray, orig_shape_hw: tuple[int,int]):
    H, W = orig_shape_hw
    big = cv2.resize(mask_small.astype(np.uint8)*255, (W, H), interpolation=cv2.INTER_NEAREST)
    return (big > 0)


def sam_prepare_image_once(img_bgr: np.ndarray):
    small_rgb, sam_scale = _resize_for_sam(img_bgr, SAM_MAX_SIDE)
    predictor.set_image(small_rgb)
    return sam_scale, img_bgr.shape[:2]


def sam_mask_from_box_scaled(box_xyxy, sam_scale, orig_hw):
    b = np.array(box_xyxy, dtype=np.float32) * sam_scale
    H, W = orig_hw

    with torch.inference_mode():
        dev = predictor.device
        b_t = torch.as_tensor(b, device=dev).unsqueeze(0)
        b_trans = predictor.transform.apply_boxes_torch(b_t, (int(H*sam_scale), int(W*sam_scale)))

        masks_t, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=b_trans,
            multimask_output=False
        )
        mask_small = masks_t[0][0].to("cpu").numpy()

    mask_big = _upsample_mask_bool(mask_small, orig_hw)
    return (mask_big.astype(np.uint8))*255


def yolo_box_mask(h, w, x1, y1, x2, y2, pad=2):
    bx1 = max(0, x1-pad)
    by1 = max(0, y1-pad)
    bx2 = min(w-1, x2+pad)
    by2 = min(h-1, y2+pad)

    m = np.zeros((h,w), np.uint8)
    m[by1:by2, bx1:bx2] = 255
    return m


def combine_masks(masks, h, w):
    m = np.zeros((h,w), np.uint8)
    for mk in masks:
        if mk is None:
            continue
        m = cv2.bitwise_or(m, (mk>0).astype(np.uint8)*255)
    return m


def bbox_from_mask(mask, pad=ROI_PAD_PX, shape=None):
    nz = cv2.findNonZero((mask>0).astype(np.uint8))
    if nz is None:
        return None
    x,y,w,h = cv2.boundingRect(nz)
    H, W = shape if shape else mask.shape[:2]
    x0 = max(0, x-pad)
    y0 = max(0, y-pad)
    x1 = min(W, x+w+pad)
    y1 = min(H, y+h+pad)
    return x0,y0,x1,y1


def inpaint_on_roi(img, full_mask, method="lama", max_roi_pixels=MAX_ROI_PIXELS):
    if int(np.count_nonzero(full_mask)) <= SMALL_MASK_AREA:
        return telea_inpaint(img, full_mask)

    box = bbox_from_mask(full_mask, pad=ROI_PAD_PX, shape=img.shape[:2])
    if box is None:
        return img

    x0,y0,x1,y1 = box
    roi_img  = img[y0:y1, x0:x1].copy()
    roi_mask = full_mask[y0:y1, x0:x1].copy()

    H, W = roi_img.shape[:2]
    pixels = H * W

    if pixels > max_roi_pixels:
        scale = (max_roi_pixels / float(pixels))**0.5
        newW = max(64, int(W*scale))
        newH = max(64, int(H*scale))
        roi_img  = cv2.resize(roi_img, (newW,newH), interpolation=cv2.INTER_AREA)
        roi_mask = cv2.resize(roi_mask, (newW,newH), interpolation=cv2.INTER_NEAREST)

    cleaned = smart_inpaint(roi_img, roi_mask, method)

    target_h = y1-y0
    target_w = x1-x0

    if cleaned.shape[:2] != (target_h,target_w):
        cleaned = cv2.resize(cleaned, (target_w,target_h), interpolation=cv2.INTER_CUBIC)

    out = img.copy()
    out[y0:y1, x0:x1] = cleaned
    return out


# ============= MAIN FUNCTION (called by FastAPI) =============
def process_image(path_in, path_out, slno=1):

    load_models_once()  # ⭐⭐ LAZY LOAD MODELS HERE ONLY ⭐⭐

    img = cv2.imread(path_in)
    if img is None:
        print("Unreadable:", path_in)
        shutil.copy2(path_in, path_out)
        return

    with torch.inference_mode():
        res = yolo(img, conf=CONF, iou=0.4, imgsz=IMGSZ, augment=False, verbose=False, device=DEVICE)

    boxes = res[0].boxes.xyxy.detach().cpu().numpy()

    if boxes.shape[0] == 0:
        shutil.copy2(path_in, path_out)
        return

    h,w = img.shape[:2]
    sam_scale, orig_hw = sam_prepare_image_once(img)

    masks = []
    for (x1,y1,x2,y2) in boxes:
        boxmask = yolo_box_mask(h,w,int(x1),int(y1),int(x2),int(y2))
        sam_m   = sam_mask_from_box_scaled([x1,y1,x2,y2], sam_scale, orig_hw)
        final_m = cv2.bitwise_and(sam_m, boxmask)
        masks.append(final_m if np.count_nonzero(final_m)>=50 else boxmask)

    full_mask = combine_masks(masks, h, w)

    cleaned = inpaint_on_roi(img, full_mask, method=INPAINT_METHOD)
    cv2.imwrite(path_out, cleaned, [int(cv2.IMWRITE_WEBP_QUALITY),80])
