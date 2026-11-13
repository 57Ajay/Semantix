import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(sys.argv[1])
FONTS = "/usr/share/fonts/truetype"


def font(name, size):
    return ImageFont.truetype(f"{FONTS}/{name}", size)


SANS = "dejavu/DejaVuSans.ttf"
BOLD = "dejavu/DejaVuSans-Bold.ttf"
MONO = "dejavu/DejaVuSansMono.ttf"
MONO_BOLD = "dejavu/DejaVuSansMono-Bold.ttf"


def save(image, relative, **kwargs):
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, **kwargs)
    print(relative, image.size)


def terminal():
    image = Image.new("RGB", (1400, 820), "#1e1f29")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 1400, 40], fill="#2b2d3a")
    for index, color in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
        draw.ellipse([18 + index * 28, 12, 34 + index * 28, 28], fill=color)
    draw.text((560, 10), "ops@bastion: ~", font=font(SANS, 18), fill="#c0c0c0")
    lines = [
        ("#8be9fd", "ops@bastion:~$ ", "#f8f8f2", "curl -v https://api.example-shop.com/v1/health"),
        ("#f8f8f2", "*   Trying 203.0.113.24:443...", None, None),
        ("#f8f8f2", "* Connected to api.example-shop.com (203.0.113.24) port 443", None, None),
        ("#f8f8f2", "* ALPN: offers h2,http/1.1", None, None),
        ("#f8f8f2", "* TLSv1.3 (OUT), TLS handshake, Client hello (1):", None, None),
        ("#f8f8f2", "* TLSv1.3 (IN), TLS handshake, Server hello (2):", None, None),
        ("#ff5555", "* SSL certificate problem: certificate has expired", None, None),
        ("#ff5555", "* Closing connection", None, None),
        ("#ff5555", "curl: (60) SSL certificate problem: certificate has expired", None, None),
        ("#f8f8f2", "More details here: https://curl.se/docs/sslcerts.html", None, None),
        ("#f8f8f2", "", None, None),
        ("#8be9fd", "ops@bastion:~$ ", "#f8f8f2", "openssl s_client -connect api.example-shop.com:443 2>/dev/null | openssl x509 -noout -dates"),
        ("#f8f8f2", "notBefore=Feb  6 09:00:00 2024 GMT", None, None),
        ("#ffb86c", "notAfter=May  7 09:00:00 2024 GMT", None, None),
        ("#8be9fd", "ops@bastion:~$ ", "#f8f8f2", "date -u"),
        ("#f8f8f2", "Tue May  7 09:14:22 UTC 2024", None, None),
        ("#8be9fd", "ops@bastion:~$ ", "#f8f8f2", ""),
    ]
    y = 62
    mono = font(MONO, 21)
    for color, text, second_color, second in lines:
        draw.text((24, y), text, font=mono, fill=color)
        if second is not None:
            offset = draw.textlength(text, font=mono)
            draw.text((24 + offset, y), second, font=mono, fill=second_color)
        y += 40
    save(image, "screenshots/Screenshot 2024-05-07 at 09.14.22.png")


def chart():
    image = Image.new("RGB", (1200, 760), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 30), "Monthly active users, first half of 2024", font=font(BOLD, 32), fill="#222")
    draw.text((60, 78), "Mobile app beta launched in April", font=font(SANS, 20), fill="#666")
    values = [("Jan", 18200), ("Feb", 19100), ("Mar", 20400), ("Apr", 26900), ("May", 31800), ("Jun", 34500)]
    left, bottom, top, width = 140, 680, 160, 1000
    for step in range(0, 40001, 10000):
        y = bottom - (bottom - top) * step / 40000
        draw.line([left, y, left + width, y], fill="#e5e5e5", width=1)
        draw.text((40, y - 12), f"{step // 1000}k", font=font(SANS, 18), fill="#777")
    bar = 110
    for index, (month, value) in enumerate(values):
        x = left + 50 + index * 160
        y = bottom - (bottom - top) * value / 40000
        draw.rectangle([x, y, x + bar, bottom], fill="#4f6bed" if index < 3 else "#f28c28")
        draw.text((x + 22, bottom + 14), month, font=font(SANS, 22), fill="#333")
        draw.text((x + 8, y - 30), f"{value:,}", font=font(SANS, 18), fill="#333")
    draw.line([left, bottom, left + width, bottom], fill="#333", width=2)
    save(image, "screenshots/Screenshot 2024-07-03 at 16.40.05.png")


def receipt():
    paper = Image.new("RGB", (620, 1180), "#fbfaf5")
    draw = ImageDraw.Draw(paper)
    mono, bold = font(MONO, 24), font(MONO_BOLD, 30)
    y = 50

    def line(text, style=mono, center=False):
        nonlocal y
        width = draw.textlength(text, font=style)
        draw.text(((620 - width) / 2 if center else 50, y), text, font=style, fill="#2a2a2a")
        y += 40

    line("GREEN VALLEY MARKET", bold, True)
    line("412 Orchard Road", center=True)
    line("Tel 555-0142", center=True)
    line("14/09/2024  17:32   Till 3", center=True)
    y += 10
    items = [
        ("Bananas 1.2kg", "2.39"), ("Oat milk 1L x2", "4.58"), ("Sourdough loaf", "5.20"),
        ("Free range eggs 12", "4.99"), ("Red lentils 1kg", "3.49"), ("Coconut milk", "1.89"),
        ("Chicken thighs 700g", "7.95"), ("Spinach 250g", "2.10"), ("Coffee beans 500g", "11.50"),
        ("Basil plant", "3.25"),
    ]
    for name, price in items:
        draw.text((50, y), name, font=mono, fill="#2a2a2a")
        draw.text((470, y), price.rjust(6), font=mono, fill="#2a2a2a")
        y += 40
    y += 10
    draw.line([50, y, 570, y], fill="#444", width=2)
    y += 20
    draw.text((50, y), "TOTAL", font=bold, fill="#111")
    draw.text((430, y), "47.34", font=bold, fill="#111")
    y += 60
    line("CARD **** 4417    47.34")
    line("Items: 11")
    y += 20
    line("Thank you for shopping local!", center=True)
    line("Keep your receipt for returns", center=True)
    paper = paper.rotate(2.5, expand=True, fillcolor="#6d5c4b")
    background = Image.new("RGB", (900, 1400), "#6d5c4b")
    noise = Image.effect_noise((900, 1400), 18).convert("RGB")
    background = Image.blend(background, noise, 0.08)
    background.paste(paper, (130, 90))
    background = background.filter(ImageFilter.GaussianBlur(0.6))
    save(background, "scans/IMG_7731.jpg", quality=88)


def slide():
    image = Image.new("RGB", (1600, 900), "#0f2a44")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 1600, 12], fill="#f2b134")
    draw.text((100, 90), "Q3 Roadmap", font=font(BOLD, 72), fill="white")
    draw.text((100, 190), "Engineering priorities, July to September", font=font(SANS, 32), fill="#b9c7d6")
    bullets = [
        "Mobile app beta: open to all customers by August",
        "Finish moving billing to Postgres 16",
        "Hire two backend engineers and one designer",
        "Cut cloud spend by 15 percent",
        "Push notifications for order updates",
    ]
    y = 310
    for bullet in bullets:
        draw.ellipse([110, y + 16, 126, y + 32], fill="#f2b134")
        draw.text((150, y), bullet, font=font(SANS, 38), fill="white")
        y += 95
    draw.text((1380, 840), "3 / 12", font=font(SANS, 24), fill="#7f93a8")
    save(image, "slides/q3-planning-deck-slide3.png")


def diagram():
    image = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(image)
    label = font(BOLD, 24)
    small = font(SANS, 18)

    def box(x, y, text, color, sub=None):
        draw.rounded_rectangle([x, y, x + 240, y + 100], radius=16, fill=color, outline="#333", width=2)
        width = draw.textlength(text, font=label)
        draw.text((x + 120 - width / 2, y + (24 if sub else 36)), text, font=label, fill="#111")
        if sub:
            width = draw.textlength(sub, font=small)
            draw.text((x + 120 - width / 2, y + 60), sub, font=small, fill="#333")
        return (x, y)

    def arrow(start, end):
        draw.line([start, end], fill="#333", width=3)
        (x1, y1), (x2, y2) = start, end
        import math
        angle = math.atan2(y2 - y1, x2 - x1)
        for side in (2.6, -2.6):
            draw.line([end, (x2 + 18 * math.cos(angle + side), y2 + 18 * math.sin(angle + side))], fill="#333", width=3)

    draw.text((60, 30), "Shop platform: system architecture", font=font(BOLD, 34), fill="#111")
    box(80, 380, "Mobile app", "#dbeafe")
    box(80, 560, "Web client", "#dbeafe")
    box(420, 470, "API gateway", "#fde68a", "TLS, rate limits")
    box(780, 200, "Auth service", "#bbf7d0")
    box(780, 420, "Orders service", "#bbf7d0")
    box(780, 640, "Billing service", "#bbf7d0")
    box(1160, 300, "Message queue", "#fbcfe8")
    box(1160, 520, "Postgres 16", "#e9d5ff", "primary + replica")
    box(1160, 740, "Payment provider", "#fecaca", "external")
    arrow((320, 430), (420, 505))
    arrow((320, 610), (420, 540))
    arrow((660, 500), (780, 250))
    arrow((660, 520), (780, 470))
    arrow((660, 540), (780, 690))
    arrow((1020, 450), (1160, 360))
    arrow((1020, 480), (1160, 560))
    arrow((1020, 680), (1160, 590))
    arrow((1020, 710), (1160, 780))
    save(image, "diagrams/architecture-v2.png")


def wifi_card():
    card = Image.new("RGB", (1000, 640), "#fffdf7")
    draw = ImageDraw.Draw(card)
    draw.rectangle([20, 20, 980, 620], outline="#2f6f4f", width=6)
    draw.text((70, 70), "Welcome! Guest WiFi", font=font(BOLD, 52), fill="#2f6f4f")
    draw.text((70, 200), "Network:", font=font(SANS, 36), fill="#444")
    draw.text((300, 196), "BlueHeron-Guest", font=font(MONO_BOLD, 42), fill="#111")
    draw.text((70, 300), "Password:", font=font(SANS, 36), fill="#444")
    draw.text((300, 296), "maple-river-42", font=font(MONO_BOLD, 42), fill="#111")
    draw.text((70, 450), "Please ask if you need the printer.", font=font(SANS, 30), fill="#666")
    photo = Image.new("RGB", (1300, 1000), "#d9d4cc")
    noise = Image.effect_noise((1300, 1000), 25).convert("RGB")
    photo = Image.blend(photo, noise, 0.06)
    card = card.rotate(-4, expand=True, fillcolor="#d9d4cc")
    photo.paste(card, (130, 150))
    draw = ImageDraw.Draw(photo)
    draw.ellipse([560, 120, 640, 200], fill="#c0392b")
    photo = photo.filter(ImageFilter.GaussianBlur(0.8))
    save(photo, "photos/2024/IMG_5590.jpg", quality=86)


def boarding_pass():
    image = Image.new("RGB", (1500, 600), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 1500, 110], fill="#123a6b")
    draw.text((40, 30), "BOARDING PASS", font=font(BOLD, 44), fill="white")
    draw.text((1080, 38), "Economy", font=font(SANS, 34), fill="white")
    fields = [
        ("Passenger", "TRAVELER / ALEX", 40, 150), ("Flight", "838", 600, 150), ("Date", "14 APR 2025", 900, 150),
        ("From", "SAN FRANCISCO SFO", 40, 280), ("To", "TOKYO NARITA NRT", 600, 280),
        ("Gate", "G96", 40, 410), ("Boarding", "10:35", 300, 410), ("Departs", "11:20", 560, 410), ("Seat", "41K", 820, 410),
    ]
    for title, value, x, y in fields:
        draw.text((x, y), title, font=font(SANS, 24), fill="#666")
        draw.text((x, y + 34), value, font=font(BOLD, 38), fill="#111")
    random.seed(4)
    x = 1120
    while x < 1460:
        width = random.choice((2, 3, 5))
        draw.rectangle([x, 380, x + width, 560], fill="black")
        x += width + random.choice((2, 3, 4))
    draw.line([1080, 110, 1080, 600], fill="#999", width=2)
    save(image, "scans/boarding-pass.png")


for make in (terminal, chart, receipt, slide, diagram, wifi_card, boarding_pass):
    make()
