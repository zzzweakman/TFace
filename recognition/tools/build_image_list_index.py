import argparse
from pathlib import Path


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build an image-list index file from images/<class>/<image> directories.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Dataset root directory. The images directory is usually <data-root>/images.",
    )
    parser.add_argument(
        "--image-dir",
        default="images",
        help="Image directory relative to --data-root, or an absolute image directory path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output index file path. Each line will be: relative_path label",
    )
    parser.add_argument(
        "--label-mode",
        choices=["sorted_index", "folder_name"],
        default="sorted_index",
        help="Use sorted class order for contiguous labels, or use numeric folder names directly.",
    )
    return parser.parse_args()


def resolve_image_root(data_root, image_dir):
    image_root = Path(image_dir)
    if not image_root.is_absolute():
        image_root = Path(data_root) / image_dir
    return image_root.resolve()


def sorted_class_dirs(image_root):
    class_dirs = [path for path in image_root.iterdir() if path.is_dir()]
    return sorted(class_dirs, key=lambda path: int(path.name) if path.name.isdigit() else path.name)


def iter_images(class_dir):
    for image_path in sorted(class_dir.iterdir(), key=lambda path: path.name):
        if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
            yield image_path


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    image_root = resolve_image_root(args.data_root, args.image_dir)
    output_path = Path(args.output).resolve()

    if not image_root.is_dir():
        raise RuntimeError("Image root does not exist: {}".format(image_root))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted_class_dirs(image_root)
    if not class_dirs:
        raise RuntimeError("No class directories found under {}".format(image_root))

    sample_num = 0
    with output_path.open("w") as output_file:
        for sorted_label, class_dir in enumerate(class_dirs):
            if args.label_mode == "folder_name":
                if not class_dir.name.isdigit():
                    raise RuntimeError(
                        "folder_name label mode requires numeric class directories, got {}".format(class_dir.name)
                    )
                label = int(class_dir.name)
            else:
                label = sorted_label

            for image_path in iter_images(class_dir):
                relative_path = image_path.relative_to(data_root).as_posix()
                output_file.write("{} {}\n".format(relative_path, label))
                sample_num += 1

    print("image_root={}".format(image_root))
    print("class_num={}".format(len(class_dirs)))
    print("sample_num={}".format(sample_num))
    print("output={}".format(output_path))


if __name__ == "__main__":
    main()
