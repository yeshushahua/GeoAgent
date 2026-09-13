class SegmentationError(RuntimeError):
    code = "SEGMENTATION_ERROR"
    status_code = 500


class SegmentationCudaUnavailableError(SegmentationError):
    code = "SEGMENTATION_CUDA_UNAVAILABLE"
    status_code = 503


class SegmentationFilesMissingError(SegmentationError):
    code = "SEGMENTATION_FILES_MISSING"
    status_code = 503


class SegmentationBusyError(SegmentationError):
    code = "SEGMENTATION_BUSY"
    status_code = 409


class SegmentationLoadError(SegmentationError):
    code = "SEGMENTATION_LOAD_FAILED"


class SegmentationInferenceError(SegmentationError):
    code = "SEGMENTATION_INFERENCE_FAILED"


class InvalidSegmentationPromptError(SegmentationError):
    code = "INVALID_SEGMENTATION_PROMPT"
    status_code = 422
