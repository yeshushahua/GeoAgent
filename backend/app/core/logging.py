import logging


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
    logging.getLogger("geoagent").setLevel(level)
