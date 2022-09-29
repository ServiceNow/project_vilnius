"""
Utility functions

"""


def _capfirst(text):
    """
    Capitalize first letter of string

    """
    return text[0].upper() + text[1:]


def _enum(values, final="and"):
    """
    Generate an enumeration of words ending with some final word

    """
    values = list(values)
    if len(values) > 1:
        return f"{', '.join(values[: -1])} {final} {values[-1]}"
    else:
        return values[-1]
