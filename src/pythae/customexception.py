class SizeMismatchError(Exception):
    pass


class LoadError(Exception):
    pass


class EncoderInputError(Exception):
    pass


class EncoderOutputError(Exception):
    pass


class DecoderInputError(Exception):
    pass


class DecoderOutputError(Exception):
    pass


class MetricInputError(Exception):
    pass


class MetricOutputError(Exception):
    pass


class BadInheritanceError(Exception):
    pass


class ModelError(Exception):
    pass


class DatasetError(Exception):
    pass
