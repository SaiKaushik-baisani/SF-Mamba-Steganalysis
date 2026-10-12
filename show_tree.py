
import os

ROOT = "."
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__",
    "node_modules", ".pytest_cache"
}

IMAGE_EXTENSIONS = {
    ".pgm", ".png", ".jpg", ".jpeg", ".bmp",
    ".webp", ".tif", ".tiff", ".gif"
}

def show_tree(folder, prefix=""):
    entries = []

    for name in os.listdir(folder):
        path = os.path.join(folder, name)

        if os.path.isdir(path):
            if name not in SKIP_DIRS:
                entries.append((name, True))
        else:
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                entries.append((name, False))

    entries.sort(key=lambda x: (not x[1], x[0].lower()))

    for i, (name, is_dir) in enumerate(entries):
        last = i == len(entries) - 1
        print(prefix + ("└── " if last else "├── ")
              + name + ("/" if is_dir else ""))

        if is_dir:
            show_tree(
                os.path.join(folder, name),
                prefix + ("    " if last else "│   ")
            )

show_tree(ROOT)
