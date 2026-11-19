from analysis.common.source import replace_source, insert_before_assignment
from analysis.feedback.balanced import common as c


def replace_once(text, old, new):
    try:
        return replace_source(text, old, new, 1)
    except ValueError as error:
        c.require(False, str(error))
