class PrivretError(Exception):
    pass


class FormatError(PrivretError):
    pass


class DecryptError(PrivretError):
    pass


class RunMismatchError(PrivretError):
    pass


class MissingExtraError(PrivretError):
    pass
