class DetectorError(RuntimeError):
    code = "DETECTOR_ERROR"
    status_code = 500


class DetectorCudaUnavailableError(DetectorError):
    code = "DETECTOR_CUDA_UNAVAILABLE"
    status_code = 503


class DetectorFilesMissingError(DetectorError):
    code = "DETECTOR_FILES_MISSING"
    status_code = 503


class DetectorBusyError(DetectorError):
    code = "DETECTOR_BUSY"
    status_code = 409


class DetectorLoadError(DetectorError):
    code = "DETECTOR_LOAD_FAILED"


class DetectorInferenceError(DetectorError):
    code = "DETECTOR_INFERENCE_FAILED"


class UnsupportedDetectionClassError(DetectorError):
    code = "UNSUPPORTED_DETECTION_CLASS"
    status_code = 422
