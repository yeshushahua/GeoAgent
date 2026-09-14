class RasterError(RuntimeError):
    code = "INVALID_RASTER"


class InvalidRasterError(RasterError):
    code = "INVALID_RASTER"


class RasterOpenFailedError(RasterError):
    code = "RASTER_OPEN_FAILED"


class BandNotFoundError(RasterError):
    code = "BAND_NOT_FOUND"


class InvalidWindowError(RasterError):
    code = "INVALID_WINDOW"


class EmptyWindowError(RasterError):
    code = "EMPTY_WINDOW"


class UnsupportedRasterDtypeError(RasterError):
    code = "UNSUPPORTED_RASTER_DTYPE"


class PreviewFailedError(RasterError):
    code = "PREVIEW_FAILED"
