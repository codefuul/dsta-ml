# import os
# import io
# import logging
# from PIL import Image, ImageEnhance
# import tesserocr
# from tesserocr import PyTessBaseAPI
# from concurrent.futures import ThreadPoolExecutor

# logging.basicConfig(level=logging.WARNING)
# logger = logging.getLogger(__name__)

# class OCRManager:
#     _instance = None
#     _initialized = False

#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instance

#     def __init__(self):
#         if self._initialized:
#             return

#         try:
#             with PyTessBaseAPI(path="/usr/share/tesseract-ocr/4.00/tessdata") as api:
#                 logger.info(f"Tesseract version: {tesserocr.tesseract_version()}")
#         except Exception as e:
#             raise RuntimeError(f"Tesseract not available: {e}")

#         self.executor = ThreadPoolExecutor(max_workers=4)
#         self._initialized = True

#     def _preprocess(self, image: Image.Image) -> Image.Image:
#         image = image.convert("L")
#         max_dim = 1800
#         if max(image.size) > max_dim:
#             scale = max_dim / max(image.size)
#             image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
#         image = ImageEnhance.Contrast(image).enhance(1.5)
#         return image.point(lambda x: 0 if x < 140 else 255, mode='1')

#     def _process_single(self, image_bytes: bytes) -> str:
#         try:
#             image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#             preprocessed = self._preprocess(image)
#             with PyTessBaseAPI(lang='eng', psm=tesserocr.PSM.SINGLE_BLOCK, path="/usr/share/tesseract-ocr/4.00/tessdata") as api:
#                 api.SetImage(preprocessed)
#                 text = api.GetUTF8Text()
#                 return text.strip()
#         except Exception as e:
#             logger.warning(f"OCR failed for a sample: {e}")
#             return ""

#     def ocr(self, image_bytes: bytes) -> str:
#         return self._process_single(image_bytes)

#     def ocr_batch(self, batch_images: list[bytes]) -> list[str]:
#         return list(self.executor.map(self._process_single, batch_images))

# ocr_manager = OCRManager()
# import os
# import io
# import logging
# from PIL import Image, ImageEnhance
# import tesserocr
# from tesserocr import PyTessBaseAPI
# from concurrent.futures import ThreadPoolExecutor

# logging.basicConfig(level=logging.WARNING)
# logger = logging.getLogger(__name__)

# class OCRManager:
#     _instance = None
#     _initialized = False

#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instance

#     def __init__(self):
#         if self._initialized:
#             return

#         try:
#             with PyTessBaseAPI(path="/usr/share/tesseract-ocr/4.00/tessdata") as api:
#                 logger.info(f"Tesseract version: {tesserocr.tesseract_version()}")
#         except Exception as e:
#             raise RuntimeError(f"Tesseract not available: {e}")

#         self.executor = ThreadPoolExecutor(max_workers=4)  # You can increase if CPU allows
#         self._initialized = True

#     def _preprocess(self, image: Image.Image) -> Image.Image:
#         image = image.convert("L")

#         max_dim = 1800
#         if max(image.size) > max_dim:
#             scale = max_dim / max(image.size)
#             image = image.resize(
#                 (int(image.width * scale), int(image.height * scale)),
#                 Image.BILINEAR  # Slightly faster than LANCZOS, similar quality for OCR
#             )

#         image = ImageEnhance.Contrast(image).enhance(1.5)
#         return image.point(lambda x: 0 if x < 140 else 255, mode='1')  # Your original thresholding

#     def _process_single(self, image_bytes: bytes) -> str:
#         try:
#             image = Image.open(io.BytesIO(image_bytes))
#             preprocessed = self._preprocess(image)
#             with PyTessBaseAPI(lang='eng', psm=tesserocr.PSM.SINGLE_BLOCK, path="/usr/share/tesseract-ocr/4.00/tessdata") as api:
#                 api.SetImage(preprocessed)
#                 return api.GetUTF8Text().strip()
#         except Exception as e:
#             logger.warning(f"OCR failed: {e}")
#             return ""

#     def ocr(self, image_bytes: bytes) -> str:
#         return self._process_single(image_bytes)

#     def ocr_batch(self, batch_images: list[bytes]) -> list[str]:
#         return list(self.executor.map(self._process_single, batch_images))

# ocr_manager = OCRManager()
# import os
# import io
# import logging
# import threading
# from PIL import Image, ImageEnhance
# import tesserocr
# from tesserocr import PyTessBaseAPI, PSM, OEM 
# from concurrent.futures import ThreadPoolExecutor # <--- THIS LINE IS CRUCIAL AND MUST BE PRESENT

# # --- Configuration ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# TESSDATA_PATH = "/usr/share/tesseract-ocr/4.00/tessdata"
# DEFAULT_LANG = 'eng'
# DEFAULT_PSM = 3 # Integer value for PSM.AUTO_PAGE_SEG
# DEFAULT_OEM = OEM.LSTM_ONLY # Still using OEM.LSTM_ONLY, which is --oem 2

# MAX_IMAGE_DIM = 2000 
# NUM_OCR_WORKERS = os.cpu_count() or 4 

# _thread_local_api = threading.local()

# class OCRManager:
#     _instance = None
#     _initialized = False

#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instance

#     def __init__(self):
#         if self._initialized:
#             return

#         logger.info("Initializing OCRManager...")

#         if not os.path.exists(TESSDATA_PATH):
#             logger.error(f"Tessdata directory not found at: {TESSDATA_PATH}")
#             raise RuntimeError(f"Tessdata directory not found. OCR cannot proceed.")

#         try:
#             self.main_api = PyTessBaseAPI(lang=DEFAULT_LANG, psm=DEFAULT_PSM, oem=DEFAULT_OEM, path=TESSDATA_PATH)
#             logger.info(f"Main PyTessBaseAPI initialized successfully.")
#             logger.info(f"Tesseract version: {tesserocr.tesseract_version()}")
#             logger.info(f"Tesseract initialized with language: '{self.main_api.GetInitLanguagesAsString()}', PSM: {self.main_api.GetPageSegMode()}")

#         except Exception as e:
#             logger.error(f"Failed to initialize main PyTessBaseAPI: {e}", exc_info=True)
#             raise RuntimeError(f"Tesseract not available or failed to initialize: {e}")

#         self.executor = ThreadPoolExecutor(max_workers=NUM_OCR_WORKERS)
#         logger.info(f"ThreadPoolExecutor initialized with {NUM_OCR_WORKERS} workers.")
        
#         self._initialized = True
#         logger.info("OCRManager fully initialized.")

#     def _get_thread_api(self):
#         if not hasattr(_thread_local_api, "api"):
#             logger.debug(f"Initializing new PyTessBaseAPI for thread: {threading.current_thread().name}")
#             _thread_local_api.api = PyTessBaseAPI(lang=DEFAULT_LANG, psm=DEFAULT_PSM, oem=DEFAULT_OEM, path=TESSDATA_PATH)
#         return _thread_local_api.api

#     def _preprocess_image(self, image_input: Image.Image) -> Image.Image:
#         if image_input.mode not in ("RGB", "L"):
#             image = image_input.convert("RGB")
#         else:
#             image = image_input

#         original_width, original_height = image.size
#         if max(original_width, original_height) > MAX_IMAGE_DIM:
#             scale_factor = MAX_IMAGE_DIM / max(original_width, original_height)
#             new_width = int(original_width * scale_factor)
#             new_height = int(original_height * scale_factor)
#             image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
#             logger.debug(f"Image downscaled from {original_width}x{original_height} to {new_width}x{new_height}.")
#         else:
#             logger.debug("Image dimension within optimal range; no downscaling applied.")

#         image = image.convert("L")
#         image = ImageEnhance.Contrast(image).enhance(1.5)
#         image = image.point(lambda x: 0 if x < 140 else 255, mode='1')
#         logger.debug("Image preprocessed (grayscale, scaled, contrast enhanced, binarized).")
#         return image

#     def _process_image_with_api(self, image_bytes: bytes, api: PyTessBaseAPI) -> str:
#         try:
#             pil_image = Image.open(io.BytesIO(image_bytes))
#             preprocessed_image = self._preprocess_image(pil_image)
            
#             api.SetImage(preprocessed_image)
#             text = api.GetUTF8Text().strip()
#             api.Clear()
#             return text
#         except Exception as e:
#             logger.warning(f"OCR failed for an image: {e}", exc_info=True)
#             return ""

#     def ocr(self, image_bytes: bytes) -> str:
#         logger.info("Starting single image OCR.")
#         result = self._process_image_with_api(image_bytes, self.main_api)
#         logger.info("Single image OCR complete.")
#         return result

#     def ocr_batch(self, batch_images_bytes: list[bytes]) -> list[str]:
#         logger.info(f"Starting batch OCR for {len(batch_images_bytes)} images.")
        
#         def _batch_worker_task(image_bytes_for_worker: bytes) -> str:
#             thread_api = self._get_thread_api()
#             return self._process_image_with_api(image_bytes_for_worker, thread_api)

#         results = list(self.executor.map(_batch_worker_task, batch_images_bytes))
#         logger.info("Batch OCR complete.")
#         return results

# ocr_manager = OCRManager()

import os
import io
import logging
import threading
from PIL import Image
import tesserocr
from tesserocr import PyTessBaseAPI, PSM, OEM
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TESSDATA_PATH = "/usr/share/tesseract-ocr/4.00/tessdata"
DEFAULT_LANG = 'eng'

# Tesseract Page Segmentation Mode (PSM) for optimal accuracy and speed.
# PSM.AUTO_PAGE_SEG (3): Fully automatic page segmentation. This is generally the most robust
#                        and often the best balance of speed and accuracy for varied document layouts.
#
# For potentially faster processing on *very specific* input types:
# PSM.SINGLE_BLOCK (6): Assume a single uniform block of text. *Consider this ONLY if your input images
#                       are consistently single, well-defined blocks of text (like a cropped paragraph).*
#                       It skips complex layout analysis, which can save time, but will significantly
#                       degrade accuracy if the layout is more complex (multiple columns, mixed content).
#                       **TEST THOROUGHLY FOR ACCURACY if you switch to PSM=6.**
#
# PSM.SINGLE_LINE (7): Treat the image as a single text line. Fastest, but only suitable if you feed
#                      cropped single lines of text.
DEFAULT_PSM = 3 # Sticking to 3 for robustness, but highlighting 6 as a faster *option* for specific cases.

DEFAULT_OEM = OEM.LSTM_ONLY

MAX_IMAGE_DIM = 2000

# Number of worker threads for concurrent OCR processing in batches.
# Using `os.cpu_count()` is a good default. For heavily CPU-bound tasks like Tesseract,
# sometimes setting `max_workers` to `os.cpu_count() - 1` (to leave a core for other tasks)
# or even tuning based on physical vs. logical cores can yield minor improvements.
NUM_OCR_WORKERS = os.cpu_count() if os.cpu_count() is not None else 8 # Use actual CPU core count, or default

_thread_local_api = threading.local()

class OCRManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info("Initializing OCRManager...")

        if not os.path.exists(TESSDATA_PATH):
            logger.error(f"Tessdata directory not found at: {TESSDATA_PATH}")
            raise RuntimeError(f"Tessdata directory not found. OCR cannot proceed.")

        try:
            self.main_api = PyTessBaseAPI(lang=DEFAULT_LANG, psm=DEFAULT_PSM, oem=DEFAULT_OEM, path=TESSDATA_PATH)
            logger.info(f"Main PyTessBaseAPI initialized successfully.")
            logger.info(f"Tesseract version: {tesserocr.tesseract_version()}")
            logger.info(f"Tesseract initialized with language: '{self.main_api.GetInitLanguagesAsString()}', PSM: {self.main_api.GetPageSegMode()}")

        except Exception as e:
            logger.error(f"Failed to initialize main PyTessBaseAPI: {e}", exc_info=True)
            raise RuntimeError(f"Tesseract not available or failed to initialize: {e}")

        self.executor = ThreadPoolExecutor(max_workers=NUM_OCR_WORKERS)
        logger.info(f"ThreadPoolExecutor initialized with {NUM_OCR_WORKERS} workers.")
        
        self._initialized = True
        logger.info("OCRManager fully initialized.")

    def _get_thread_api(self):
        if not hasattr(_thread_local_api, "api"):
            logger.debug(f"Initializing new PyTessBaseAPI for thread: {threading.current_thread().name}")
            _thread_local_api.api = PyTessBaseAPI(lang=DEFAULT_LANG, psm=DEFAULT_PSM, oem=DEFAULT_OEM, path=TESSDATA_PATH)
        return _thread_local_api.api

    def _preprocess_image(self, image_input: Image.Image) -> Image.Image:
        if image_input.mode not in ("RGB", "L"):
            image = image_input.convert("RGB")
        else:
            image = image_input

        original_width, original_height = image.size
        if max(original_width, original_height) > MAX_IMAGE_DIM:
            scale_factor = MAX_IMAGE_DIM / max(original_width, original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.debug(f"Image downscaled from {original_width}x{original_height} to {new_width}x{new_height}.")
        else:
            logger.debug("Image dimension within optimal range; no downscaling applied.")

        image = image.convert("L")
        
        logger.debug("Image preprocessed (grayscale and scaled). Tesseract will handle binarization and contrast internally.")
        return image

    def _process_image_with_api(self, image_bytes: bytes, api: PyTessBaseAPI) -> str:
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            preprocessed_image = self._preprocess_image(pil_image)
            
            api.SetImage(preprocessed_image)
            text = api.GetUTF8Text().strip()
            api.Clear()
            return text
        except Exception as e:
            logger.warning(f"OCR failed for an image: {e}", exc_info=True)
            return ""

    def ocr(self, image_bytes: bytes) -> str:
        logger.info("Starting single image OCR.")
        result = self._process_image_with_api(image_bytes, self.main_api)
        logger.info("Single image OCR complete.")
        return result

    def ocr_batch(self, batch_images_bytes: list[bytes]) -> list[str]:
        logger.info(f"Starting batch OCR for {len(batch_images_bytes)} images.")
        
        def _batch_worker_task(image_bytes_for_worker: bytes) -> str:
            thread_api = self._get_thread_api()
            return self._process_image_with_api(image_bytes_for_worker, thread_api)

        results = list(self.executor.map(_batch_worker_task, batch_images_bytes))
        logger.info("Batch OCR complete.")
        return results

ocr_manager = OCRManager()