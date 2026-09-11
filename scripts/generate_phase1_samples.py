"""Create small original RGB benchmark scenes without external datasets."""
from pathlib import Path

from PIL import Image, ImageDraw

from backend.app.core.config import get_settings


def save_scene(name: str, size: tuple[int, int], painter) -> None:
    folder = get_settings().project_root / "sample_data" / "images"
    folder.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, "#dbe9f4")
    painter(ImageDraw.Draw(image), size)
    image.save(folder / name, optimize=True)


def street(draw: ImageDraw.ImageDraw, size):
    w, h = size
    draw.rectangle((0, h * 0.58, w, h), fill="#5f6770")
    draw.rectangle((0, 0, w, h * 0.58), fill="#95cbe8")
    draw.rectangle((40, 140, 340, 500), fill="#d97c56", outline="#733b2c", width=5)
    for x in (80, 180, 260):
        draw.rectangle((x, 200, x + 55, 285), fill="#c9efff", outline="#314b5b", width=3)
    draw.polygon([(520, 620), (760, 620), (715, 510), (565, 510)], fill="#2e608b")
    draw.ellipse((565, 590, 620, 645), fill="#222")
    draw.ellipse((665, 590, 720, 645), fill="#222")
    draw.line((480, h, 610, h * 0.58), fill="#f5e8a6", width=15)
    draw.line((820, h, 720, h * 0.58), fill="#f5e8a6", width=15)
    draw.text((30, 25), "ORIGINAL SYNTHETIC STREET SCENE", fill="#123")


def aerial(draw: ImageDraw.ImageDraw, size):
    w, h = size
    draw.rectangle((0, 0, w, h), fill="#79a95b")
    draw.polygon([(0, 350), (430, 250), (w, 390), (w, 520), (420, 390), (0, 480)], fill="#6d7379")
    draw.line((0, 415, w, 455), fill="#f3dd71", width=12)
    draw.polygon([(640, 0), (860, 0), (780, h), (570, h)], fill="#5aa3c2")
    for y in range(80, h, 190):
        for x in range(60, 520, 180):
            draw.rectangle((x, y, x + 95, y + 70), fill="#d9d0b8", outline="#4a483f", width=3)
    draw.text((30, 25), "ORIGINAL SYNTHETIC AERIAL RGB VIEW", fill="#102b18")


def objects(draw: ImageDraw.ImageDraw, size):
    w, h = size
    draw.rectangle((0, 0, w, h), fill="#f0e7d4")
    colors = ["#d84a4a", "#4b82d0", "#e4b53d", "#4fa267", "#9d62bd", "#e87931"]
    for index, color in enumerate(colors):
        x = 90 + (index % 3) * 290
        y = 110 + (index // 3) * 300
        draw.ellipse((x, y, x + 150, y + 150), fill=color, outline="#222", width=5)
    draw.rectangle((410, 300, 590, 510), fill="#f7f7f7", outline="#222", width=5)
    draw.text((30, 25), "SIX COLORED BALLS AND ONE WHITE BOX", fill="#222")


def spatial(draw: ImageDraw.ImageDraw, size):
    w, h = size
    draw.rectangle((0, 0, w, h), fill="#f7f4ec")
    draw.rectangle((70, 70, 300, 250), fill="#d75151", outline="#222", width=5)
    draw.ellipse((700, 80, 940, 320), fill="#4b86cf", outline="#222", width=5)
    draw.polygon([(430, 570), (600, 310), (770, 570)], fill="#58a66c", outline="#222")
    draw.text((90, 130), "RED", fill="white")
    draw.text((770, 170), "BLUE", fill="white")
    draw.text((485, 470), "GREEN", fill="white")
    draw.text((30, 25), "UPPER-LEFT / UPPER-RIGHT / LOWER-CENTER", fill="#222")


def high_resolution(draw: ImageDraw.ImageDraw, size):
    w, h = size
    draw.rectangle((0, 0, w, h), fill="#e8efe5")
    for x in range(0, w, 256):
        draw.line((x, 0, x, h), fill="#9db79a", width=8)
    for y in range(0, h, 256):
        draw.line((0, y, w, y), fill="#9db79a", width=8)
    draw.ellipse((w * 0.35, h * 0.25, w * 0.65, h * 0.75), fill="#4e93bc")
    draw.text((80, 80), "4096 x 3072 HIGH RESOLUTION INPUT CONTROL TEST", fill="#17391b")


def main():
    save_scene("phase1-street.png", (1024, 768), street)
    save_scene("phase1-aerial-rgb.png", (1024, 768), aerial)
    save_scene("phase1-objects.png", (1024, 768), objects)
    save_scene("phase1-spatial.png", (1024, 768), spatial)
    save_scene("phase1-high-resolution.png", (4096, 3072), high_resolution)
    print("Created five original Phase 1 RGB samples.")


if __name__ == "__main__":
    main()
