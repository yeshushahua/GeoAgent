class VlmError(RuntimeError):
    code = "VLM_ERROR"
    status_code = 500


class CudaUnavailableError(VlmError):
    code = "CUDA_UNAVAILABLE"
    status_code = 503


class ModelFilesMissingError(VlmError):
    code = "MODEL_FILES_MISSING"
    status_code = 503


class ModelBusyError(VlmError):
    code = "MODEL_BUSY"
    status_code = 409


class ModelLoadError(VlmError):
    code = "MODEL_LOAD_FAILED"
    status_code = 500


class InvalidInputError(VlmError):
    code = "INVALID_INPUT"
    status_code = 422


class InvalidImageError(VlmError):
    code = "INVALID_IMAGE"
    status_code = 415


class InferenceFailedError(VlmError):
    code = "INFERENCE_FAILED"
    status_code = 500


class CudaOutOfMemoryError(VlmError):
    code = "CUDA_OUT_OF_MEMORY"
    status_code = 507
