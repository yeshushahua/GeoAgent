class GeoError(RuntimeError):
    code = "GEOSPATIAL_ERROR"


class CrsRequiredError(GeoError):
    code = "CRS_REQUIRED"


class InvalidPixelError(GeoError):
    code = "INVALID_PIXEL"


class InvalidGeometryError(GeoError):
    code = "INVALID_GEOMETRY"


class InvalidVectorError(GeoError):
    code = "INVALID_VECTOR"


class CrsTransformError(GeoError):
    code = "CRS_TRANSFORM_FAILED"


class AreaCalculationError(GeoError):
    code = "AREA_CALCULATION_FAILED"


class ZonalStatisticsError(GeoError):
    code = "ZONAL_STATISTICS_FAILED"