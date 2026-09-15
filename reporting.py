SECTION_WIDTH = 70


def print_header(title):
    print()
    print("=" * SECTION_WIDTH)
    print(title)
    print("=" * SECTION_WIDTH)


def print_subheader(title):
    print()
    print(title)
    print("-" * SECTION_WIDTH)


def print_row(label, value, label_width=32):
    print(f"{label:<{label_width}}{value}")


def print_saved(label, path):
    print(f"{label}: {path}")
