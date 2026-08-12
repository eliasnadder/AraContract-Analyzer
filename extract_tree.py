import os

def show_tree(path=".", prefix="", exclude={"venv", "env", "__pycache__", "node_modules", ".git", ".idea", "dist", "build", "*.pyc"}):
    items = [i for i in os.listdir(path) if i not in exclude and not i.endswith(".pyc")]
    items.sort()
    for i, item in enumerate(items):
        connector = "└── " if i == len(items)-1 else "├── "
        print(prefix + connector + item)
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            extension = "    " if i == len(items)-1 else "│   "
            show_tree(full_path, prefix + extension, exclude)

show_tree(".")