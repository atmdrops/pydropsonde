# import io
import subprocess
import qrcode

# from cairosvg import svg2png
from PIL import Image


def _main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="https://github.com/atmdrops/pydropsonde",
        help="URL behind the QRcode pointing to pydropsonde repo or docu",
    )
    parser.add_argument("--logo", default=None, help="Logo image to be embedded")
    parser.add_argument("--ofile", default="qrcode.png", help="output file name")
    args = parser.parse_args()

    img = qrcode.make(args.url)

    if args.logo:
        print(f"Embedding logo: {args.logo}")
        img = img.convert("RGBA")
        if args.logo.endswith(".svg"):
            logo_png = args.logo[:-3] + "png"
            subprocess.run(["magick", args.logo, logo_png])
            logo = Image.open(logo_png)
        else:
            logo = Image.open(args.logo)
        img.alpha_composite(
            logo, ((img.size[0] - logo.size[0]) // 2, (img.size[1] - logo.size[1]) // 2)
        )
        subprocess.run(["rm", logo_png])

    img.save(args.ofile, "PNG")


if __name__ == "__main__":
    _main()
