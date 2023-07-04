class Error(object):
    """A class for representing errors that occur in the program.

    Attributes:
        _code (int): The error code.
        _msg_prefix (str): The prefix for the error message.
        _msg (str): The error message.
    """
    def __init__(self, code, msg_prefix):
        """Initialize a new Error instance.

        Args:
            code (int): The error code.
            msg_prefix (str): The prefix for the error message.
        """
        self._code = code
        self._msg_prefix = msg_prefix
        self._msg = ''

    @property
    def code(self):
        """int: The error code."""
        return self._code

    @code.setter
    def code(self, code):
        """Set the error code.

        Args:
            code (int): The new error code.
        """
        self._code = code

    @property
    def msg(self):
        """str: The full error message, including the prefix and message."""
        return self._msg_prefix + ' - ' + self._msg

    @msg.setter
    def msg(self, msg):
        """Set the error message.

        Args:
            msg (str): The new error message.
        """
        self._msg = msg


class JsonParseError(Error):
    """A class for representing JSON parse errors that occur in the program.

    Args:
        msg (str, optional): The error message. Defaults to an empty string.
    """
    def __init__(self, msg=''):
        """Initialize a new JsonParseError instance.

        Args:
            msg (str, optional): The error message. Defaults to an empty
             string.
        """
        super(JsonParseError, self).__init__(460, 'Json Parse Error')
        self.msg = msg


class InputFormatError(Error):
    """A class for representing input format errors that occur in the program.

    Args:
        msg (str, optional): The error message. Defaults to an empty string.
    """
    def __init__(self, msg=''):
        """Initialize a new InputFormatError instance.

        Args:
            msg (str, optional): The error message. Defaults to an empty
             string.
        """
        super(InputFormatError, self).__init__(460, 'Input Body Format Error')
        self.msg = msg


class ImageDecodeError(Error):
    """A class for representing image decode errors that occur in the program.

    Args:
        msg (str, optional): The error message. Defaults to an empty string.
    """
    def __init__(self, msg=''):
        """Initialize a new ImageDecodeError instance.

        Args:
            msg (str, optional): The error message. Defaults to an empty
             string.
        """
        super(ImageDecodeError, self).__init__(462, 'Image Decode Error')
        self.msg = msg


class UnExpectedServerError(Error):
    """A class for representing unexpected server errors that occur in
    the program.

    Args:
        msg (str, optional): The error message. Defaults to an empty string.
    """
    def __init__(self, msg=''):
        """Initialize a new UnExpectedServerError instance.

        Args:
            msg (str, optional): The error message. Defaults to an
             empty string.
        """
        super(UnExpectedServerError,
              self).__init__(469, 'Unexpected \
        Server Error')
        self.msg = msg
