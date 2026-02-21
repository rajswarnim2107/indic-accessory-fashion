import runpod
from runpod.serverless.utils import rp_upload
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii # imported for Base64 error handling
import subprocess
import time

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _truncate(value, max_len=180):
    text = str(value)
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}...(+{len(text) - max_len} chars)"


def _sanitize_job_input(job_input):
    """Summarize input values for logging (mask base64/long prompt)"""
    sanitized = {}
    for key, value in job_input.items():
        if key.startswith("image_base64"):
            if isinstance(value, str):
                sanitized[key] = f"<base64:{len(value)} chars>"
            else:
                sanitized[key] = "<base64:non-string>"
            continue
        if key == "prompt" and isinstance(value, str):
            sanitized[key] = _truncate(value, 240)
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        else:
            sanitized[key] = _truncate(repr(value), 140)
    return sanitized

# CUDA check and setup
def check_cuda_availability():
    """Check if CUDA is available and set environment variables."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("✅ CUDA is available and working")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            return True
        else:
            logger.error("❌ CUDA is not available")
            raise RuntimeError("CUDA is required but not available")
    except Exception as e:
        logger.error(f"❌ CUDA check failed: {e}")
        raise RuntimeError(f"CUDA initialization failed: {e}")

# Run CUDA check
try:
    cuda_available = check_cuda_availability()
    if not cuda_available:
        raise RuntimeError("CUDA is not available")
except Exception as e:
    logger.error(f"Fatal error: {e}")
    logger.error("Exiting due to CUDA requirements not met")
    exit(1)

server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())
def save_data_if_base64(data_input, temp_dir, output_filename):
    """
    Check if input data is a Base64 string, save to file if so, return path.
    Otherwise, return as path string.
    """
    # If not a string, return unchanged
    if not isinstance(data_input, str):
        return data_input

    try:
        # If it's base64, decoding will succeed
        decoded_data = base64.b64decode(data_input)
        
        # Create directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        # If decoding succeeded, save as temp file
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f: # Write in binary mode
            f.write(decoded_data)
        
        # Return saved file path
        print(f"✅ Base64 input saved to file '{file_path}'.")
        return file_path

    except (binascii.Error, ValueError):
        # If decoding fails, treat as file path and return as-is
        print(f"➡️ '{data_input}' treated as a file path.")
        return data_input
    
def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    logger.info(f"Getting image from: {url}")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def get_images(ws, prompt, request_id="n/a"):
    queued = queue_prompt(prompt)
    prompt_id = queued['prompt_id']
    logger.info(f"[{request_id}] Prompt queued. prompt_id={prompt_id}")
    output_images = {}
    loop_count = 0
    while True:
        loop_count += 1
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    logger.info(f"[{request_id}] ComfyUI execution finished for prompt_id={prompt_id}")
                    break
            elif loop_count % 25 == 0:
                logger.info(f"[{request_id}] Waiting for completion... event={message.get('type')}")
        else:
            continue

    history_all = get_history(prompt_id)
    history = history_all.get(prompt_id)
    if history is None:
        raise RuntimeError(f"History not found for prompt_id={prompt_id}")

    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                # Convert bytes to base64 for JSON serialization
                if isinstance(image_data, bytes):
                    import base64
                    image_data = base64.b64encode(image_data).decode('utf-8')
                images_output.append(image_data)
        output_images[node_id] = images_output
    logger.info(f"[{request_id}] Output nodes collected: {len(output_images)}")

    return output_images

def load_workflow(workflow_path):
    with open(workflow_path, 'r') as file:
        return json.load(file)

# Workflow filenames by image count
_WORKFLOW_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflow")
_WORKFLOW_FILES = {
    1: "qwen_image_edit_1_1image.json",
    2: "qwen_image_edit_1_2image.json",
    3: "qwen_image_edit_1_3image.json",
}

# Node IDs per workflow (varies by image count)
# 1-image: LoadImage=78, KSampler(seed)=3, prompt=111
# 2-image: above + LoadImage2=117
# 3-image: above + LoadImage3=119
_NODE_IMAGE_1 = "78"
_NODE_IMAGE_2 = "117"
_NODE_IMAGE_3 = "119"
_NODE_SEED = "3"
_NODE_PROMPT = "111"
_NODE_WIDTH = "128"   # not present in current workflow (optional)
_NODE_HEIGHT = "129"  # not present in current workflow (optional)
_NODE_SAVE_IMAGE = "60"


def build_default_tryon_prompt(
    category: str,
    garment_name: str,
    garment_photo_type: str,
    extra_hint: str = "",
    num_images: int = 1,
    accessory_class: str = "",
    accessory_classes=None,
) -> str:
    if num_images >= 3:
        return _build_multi_reference_prompt(
            category=category,
            garment_name=garment_name,
            extra_hint=extra_hint,
            accessory_class=accessory_class,
            accessory_classes=accessory_classes,
        )
    return _build_single_reference_prompt(
        category=category,
        garment_name=garment_name,
        garment_photo_type=garment_photo_type,
        extra_hint=extra_hint,
        accessory_class=accessory_class,
    )


def _build_single_reference_prompt(
    category: str,
    garment_name: str,
    garment_photo_type: str,
    extra_hint: str = "",
    accessory_class: str = "",
) -> str:
    category_map = {
        "tops": "top",
        "bottoms": "bottom",
        "one-pieces": "one-piece",
        "accessories": "accessory",
    }
    target_region = category_map.get((category or "").strip().lower(), "garment region")
    garment_label = garment_name.strip() if isinstance(garment_name, str) and garment_name.strip() else "the reference garment"
    photo_type = garment_photo_type.strip() if isinstance(garment_photo_type, str) and garment_photo_type.strip() else "reference"

    accessory_line = ""
    if target_region == "accessory" and accessory_class:
        accessory_line = f"Accessory type is {accessory_class}; place it naturally on the correct body region. "

    core = (
        "Virtual try-on edit. "
        "CRITICAL: Image 1 is the source person — strictly preserve their face, skin tone, hair, "
        "body shape, exact pose, limb positions, camera angle, framing, and full background. "
        "NEVER replace the person with the model shown in any reference photo. "
        "Reference images (Image 2, Image 3) are ONLY for extracting garment/accessory appearance "
        "(fabric, pattern, color, cut, drape). Ignore the reference model's face, body, pose, and setting entirely. "
        f"Extract only the {target_region} design from the reference ({photo_type} photo) and apply it onto "
        f"the original person from Image 1. "
        f"Garment style to match: {garment_label}. "
        f"{accessory_line}"
        "Do not alter face, hair, hands, skin, or any region outside the target garment area."
    )
    if isinstance(extra_hint, str) and extra_hint.strip():
        return f"{core} Additional instruction: {extra_hint.strip()}"
    return core


def _build_multi_reference_prompt(
    category: str,
    garment_name: str,
    extra_hint: str = "",
    accessory_class: str = "",
    accessory_classes=None,
) -> str:
    item_names = [n.strip() for n in (garment_name or "").split(",") if n.strip()]
    item_line = ", ".join(item_names) if item_names else "selected reference items"

    acc_classes = []
    if isinstance(accessory_classes, (list, tuple)):
        acc_classes = [str(c).strip() for c in accessory_classes if str(c).strip()]
    elif accessory_class:
        acc_classes = [accessory_class.strip()]

    accessory_line = ""
    if acc_classes:
        accessory_line = (
            f"Accessories ({', '.join(acc_classes)}) are high priority: keep each visible, "
            "faithful in shape/material, and naturally placed on the correct body region. "
        )
    else:
        accessory_line = "Do not invent new accessories unless explicitly referenced. "

    category_value = (category or "").strip().lower()
    category_line = ""
    if category_value == "one-pieces":
        category_line = (
            "Primary target is one-piece garment from reference; use the exact garment class visible "
            "(bikini stays bikini, dress stays dress, saree stays saree). "
        )

    core = (
        "Virtual try-on edit. "
        "CRITICAL: Image 1 is the source person — strictly preserve their face, skin tone, hair, "
        "body shape, exact pose, limb positions, camera angle, framing, and full background. "
        "NEVER replace the person with any model shown in reference photos. "
        "Reference images (Image 2, Image 3) provide ONLY garment/accessory appearance "
        "(fabric, pattern, color, cut, silhouette). Completely ignore the reference model's face, body, pose, and setting. "
        f"{category_line}"
        f"Extract outfit/accessory design from references and dress the original person in: {item_line}. "
        f"{accessory_line}"
        "Do not alter face, hair, hands, skin, or any region outside the target garment/accessory area."
    )
    if isinstance(extra_hint, str) and extra_hint.strip():
        return f"{core} Additional instruction: {extra_hint.strip()}"
    return core

# ------------------------------
# Input processing utils (path/url/base64)
# ------------------------------
def process_input(input_data, temp_dir, output_filename, input_type):
    """Process input data and return a file path.
    - input_type: "path" | "url" | "base64"
    """
    if input_type == "path":
        logger.info(f"📁 Handling input path: {input_data}")
        return input_data
    elif input_type == "url":
        logger.info(f"🌐 Handling input URL: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        logger.info("🔢 Handling base64 input")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"Unsupported input type: {input_type}")

def download_file_from_url(url, output_path):
    """Download file from URL."""
    try:
        result = subprocess.run([
            'wget', '-O', output_path, '--no-verbose', url
        ], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Successfully downloaded file from URL: {url} -> {output_path}")
            return output_path
        else:
            logger.error(f"❌ wget download failed: {result.stderr}")
            raise Exception(f"URL download failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("❌ Download timeout")
        raise Exception("Download timeout")
    except Exception as e:
        logger.error(f"❌ Error while downloading: {e}")
        raise Exception(f"Error while downloading: {e}")

def save_base64_to_file(base64_data, temp_dir, output_filename):
    """Save base64 data to file."""
    try:
        decoded_data = base64.b64decode(base64_data)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        logger.info(f"✅ Base64 input saved to file '{file_path}'.")
        return file_path
    except (binascii.Error, ValueError) as e:
        logger.error(f"❌ Base64 decode error: {e}")
        raise Exception(f"Base64 decode error: {e}")

def handler(job):
    task_id = f"task_{uuid.uuid4()}"
    started_at = time.perf_counter()
    ws = None
    job_input = job.get("input", {})
    logger.info(f"[{task_id}] Received job input: {_sanitize_job_input(job_input)}")

    try:
        # ------------------------------
        # Collect image inputs (1/2/3 images supported)
        # Supported keys: image_path | image_url | image_base64
        #                 image_path_2 | image_url_2 | image_base64_2
        #                 image_path_3 | image_url_3 | image_base64_3
        # ------------------------------
        image_paths = []

        for i, suffix in enumerate([ "", "_2", "_3" ], start=1):
            path_key = f"image_path{suffix}"
            url_key = f"image_url{suffix}"
            b64_key = f"image_base64{suffix}"
            fname = f"input_image_{i}.jpg"
            if path_key in job_input:
                logger.info(f"[{task_id}] Resolving image{i} from key={path_key}")
                image_paths.append(process_input(job_input[path_key], task_id, fname, "path"))
            elif url_key in job_input:
                logger.info(f"[{task_id}] Resolving image{i} from key={url_key}")
                image_paths.append(process_input(job_input[url_key], task_id, fname, "url"))
            elif b64_key in job_input:
                logger.info(f"[{task_id}] Resolving image{i} from key={b64_key}")
                image_paths.append(process_input(job_input[b64_key], task_id, fname, "base64"))
            else:
                break

        num_images = len(image_paths)
        logger.info(f"[{task_id}] Total input images resolved: {num_images}")
        if num_images == 0:
            return {"error": "At least 1 image input is required. Use one of: image_path / image_url / image_base64."}

        if num_images not in _WORKFLOW_FILES:
            return {"error": f"Supported image counts are 1, 2, or 3. Number of input images: {num_images}"}

        workflow_filename = _WORKFLOW_FILES[num_images]
        workflow_path = os.path.join(_WORKFLOW_BASE, workflow_filename)
        logger.info(f"[{task_id}] Selected workflow: {workflow_filename}")
        if not os.path.exists(workflow_path):
            return {"error": f"Workflow file not found: {workflow_path}"}

        prompt = load_workflow(workflow_path)
        logger.info(f"[{task_id}] Workflow loaded. node_count={len(prompt)}")

        # Node numbers are used as in each workflow JSON
        prompt[_NODE_IMAGE_1]["inputs"]["image"] = image_paths[0]
        logger.info(f"[{task_id}] Assigned image1 -> node {_NODE_IMAGE_1}")
        if num_images >= 2:
            prompt[_NODE_IMAGE_2]["inputs"]["image"] = image_paths[1]
            logger.info(f"[{task_id}] Assigned image2 -> node {_NODE_IMAGE_2}")
        if num_images >= 3:
            prompt[_NODE_IMAGE_3]["inputs"]["image"] = image_paths[2]
            logger.info(f"[{task_id}] Assigned image3 -> node {_NODE_IMAGE_3}")

        requested_prompt = job_input.get("prompt", "")
        prompt_source = "request"
        if not isinstance(requested_prompt, str) or not requested_prompt.strip():
            requested_prompt = build_default_tryon_prompt(
                category=job_input.get("category", "tops"),
                garment_name=job_input.get("garment_name", ""),
                garment_photo_type=job_input.get("garment_photo_type", "reference"),
                extra_hint=job_input.get("prompt_hint", ""),
                num_images=num_images,
                accessory_class=job_input.get("accessory_class", ""),
                accessory_classes=job_input.get("accessory_classes", []),
            )
            prompt_source = "default_tryon_prompt"
        prompt[_NODE_PROMPT]["inputs"]["prompt"] = requested_prompt
        logger.info(
            f"[{task_id}] Prompt applied from {prompt_source}. length={len(requested_prompt)} text={_truncate(requested_prompt, 220)}"
        )

        if _NODE_SEED in prompt and "seed" in job_input:
            prompt[_NODE_SEED]["inputs"]["seed"] = job_input["seed"]
            logger.info(f"[{task_id}] Seed override applied: {job_input['seed']}")
        if _NODE_WIDTH in prompt and "width" in job_input:
            prompt[_NODE_WIDTH]["inputs"]["value"] = job_input["width"]
            logger.info(f"[{task_id}] Width override applied: {job_input['width']}")
        if _NODE_HEIGHT in prompt and "height" in job_input:
            prompt[_NODE_HEIGHT]["inputs"]["value"] = job_input["height"]
            logger.info(f"[{task_id}] Height override applied: {job_input['height']}")

        ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
        logger.info(f"[{task_id}] Connecting to WebSocket: {ws_url}")
        
        # First, check if HTTP connection works
        http_url = f"http://{server_address}:8188/"
        logger.info(f"[{task_id}] Checking HTTP connection to: {http_url}")
        
        # HTTP connection check
        max_http_attempts = 180
        for http_attempt in range(max_http_attempts):
            try:
                import urllib.request
                response = urllib.request.urlopen(http_url, timeout=5)
                logger.info(f"[{task_id}] HTTP connection succeeded (attempt {http_attempt+1}) status={response.status}")
                break
            except Exception as e:
                if http_attempt < 3 or (http_attempt + 1) % 15 == 0:
                    logger.warning(f"[{task_id}] HTTP connection failed (attempt {http_attempt+1}/{max_http_attempts}): {e}")
                if http_attempt == max_http_attempts - 1:
                    raise Exception("Could not connect to the ComfyUI server. Please make sure the server is running.")
                time.sleep(1)
        
        ws = websocket.WebSocket()
        ws.settimeout(180)
        # WebSocket connection attempts
        max_attempts = int(180/5)  # 3 minutes (5 sec intervals)
        for attempt in range(max_attempts):
            try:
                ws.connect(ws_url)
                logger.info(f"[{task_id}] WebSocket connection succeeded (attempt {attempt+1})")
                break
            except Exception as e:
                if attempt < 3 or (attempt + 1) % 5 == 0:
                    logger.warning(f"[{task_id}] WebSocket connection failed (attempt {attempt+1}/{max_attempts}): {e}")
                if attempt == max_attempts - 1:
                    raise Exception("WebSocket connection timeout (3 minutes)")
                time.sleep(5)

        images = get_images(ws, prompt, request_id=task_id)
        node_image_counts = {node_id: len(node_images) for node_id, node_images in images.items()}
        logger.info(f"[{task_id}] Image outputs by node: {node_image_counts}")

        # If no images, handle gracefully
        if not images:
            return {"error": "Could not generate any images."}
        
        response_meta = {
            "prompt_used": requested_prompt,
            "prompt_source": prompt_source,
            "num_images": num_images,
            "category": job_input.get("category", ""),
            "accessory_classes": job_input.get("accessory_classes", []),
        }

        if _NODE_SAVE_IMAGE in images and images[_NODE_SAVE_IMAGE]:
            elapsed = time.perf_counter() - started_at
            logger.info(f"[{task_id}] Completed successfully via SaveImage node in {elapsed:.2f}s")
            return {"image": images[_NODE_SAVE_IMAGE][0], "meta": response_meta}

        for node_id in images:
            if images[node_id]:
                elapsed = time.perf_counter() - started_at
                logger.info(f"[{task_id}] Completed with fallback node={node_id} in {elapsed:.2f}s")
                return {"image": images[node_id][0], "meta": response_meta}
        
        return {"error": f"No images produced. workflow={workflow_filename}"}
    except Exception as e:
        elapsed = time.perf_counter() - started_at
        logger.exception(f"[{task_id}] Handler failed after {elapsed:.2f}s: {e}")
        return {"error": f"An error occurred during processing: {str(e)}"}
    finally:
        if ws is not None:
            try:
                ws.close()
                logger.info(f"[{task_id}] WebSocket closed")
            except Exception:
                logger.warning(f"[{task_id}] WebSocket close skipped (already closed)")

runpod.serverless.start({"handler": handler})