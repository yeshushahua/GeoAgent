class OpenVocabularyError(RuntimeError):
    code = "OPEN_VOCAB_ERROR"
    status_code = 500


class OpenVocabularyCudaUnavailableError(OpenVocabularyError):
    code = "OPEN_VOCAB_CUDA_UNAVAILABLE"
    status_code = 503


class OpenVocabularyFilesMissingError(OpenVocabularyError):
    code = "OPEN_VOCAB_FILES_MISSING"
    status_code = 503


class OpenVocabularyBusyError(OpenVocabularyError):
    code = "OPEN_VOCAB_BUSY"
    status_code = 409


class OpenVocabularyLoadError(OpenVocabularyError):
    code = "OPEN_VOCAB_LOAD_FAILED"


class OpenVocabularyInferenceError(OpenVocabularyError):
    code = "OPEN_VOCAB_INFERENCE_FAILED"


class InvalidOpenVocabularyClassesError(OpenVocabularyError):
    code = "INVALID_OPEN_VOCAB_CLASSES"
    status_code = 422
