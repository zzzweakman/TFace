import cv2
import numpy as np


class IndexParser(object):
    """ Class for Line parser.
    """
    def __init__(self) -> None:
        self.sample_num = 0
        self.class_num = 0

    def __call__(self, line):
        line_s = self._split(line)
        if len(line_s) == 2:
            # Default line format
            img_path, label = line_s
            label = int(label)
            self.sample_num += 1
            self.class_num = max(self.class_num, label)
            return (img_path, label)
        elif len(line_s) == 4:
            # IndexTFRDataset line format
            tfr_name, tfr_index, tfr_offset, label = line_s
            label = int(label)
            tfr_file = "{0}/{0}-{1:05d}.tfrecord".format(tfr_name, int(tfr_index))
            tfr_offset = int(tfr_offset)
            self.sample_num += 1
            self.class_num = max(self.class_num, label)
            return (tfr_file, tfr_offset, label)
        else:
            raise RuntimeError("IndexParser line length %d not supported" % len(line_s))

    def _split(self, line):
        line = line.strip()
        if not line:
            raise RuntimeError("IndexParser received an empty line")
        if '\t' in line:
            fields = [item for item in line.split('\t') if item != ""]
        else:
            fields = line.split()
        return fields

    def reset(self):
        self.sample_num = 0
        self.class_num = 0


class ImgSampleParser(object):
    """ Class for Image Sample parser
    """
    def __init__(self, transform) -> None:
        self.transform = transform

    def __call__(self, path, label):
        image = cv2.imread(path)
        if image is None:
            raise RuntimeError("Failed to read image file: {}".format(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class TFRecordSampleParser(object):
    """ Class for TFRecord Sample parser
    """
    def __init__(self, transform) -> None:
        self.transform = transform
        self.file_readers = dict()
        self._db = None
        self._example_pb2 = None

    def _get_db_module(self):
        if self._db is None:
            import dareblopy as db
            self._db = db
        return self._db

    def _get_example_pb2_module(self):
        if self._example_pb2 is None:
            from . import example_pb2
            self._example_pb2 = example_pb2
        return self._example_pb2

    def __call__(self, record_path, offset, label):
        rr = self.file_readers.get(record_path, None)
        if rr is None:
            db = self._get_db_module()
            rr = db.RecordReader(record_path)
            self.file_readers[record_path] = rr
        pb_data = rr.read_record(offset)
        example_pb2 = self._get_example_pb2_module()
        example = example_pb2.Example()
        example.ParseFromString(pb_data)
        image_raw = example.features.feature['image'].bytes_list.value[0]
        image = cv2.imdecode(np.frombuffer(image_raw, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
