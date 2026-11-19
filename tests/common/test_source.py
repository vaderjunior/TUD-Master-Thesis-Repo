import pytest
from analysis.common.source import replace_source, insert_before_assignment


def test_replacement_accepts_whitespace_and_quote_variants():
    source = (
        "def f(value):" + chr(10) + "    return old( value, key = 'name' )" + chr(10)
    )
    result = replace_source(source, 'old(value, key="name")', 'new(value, key="name")')
    namespace = {
        "old": lambda value, **kw: -1,
        "new": lambda value, **kw: value + len(kw["key"]),
    }
    exec(result, namespace)
    assert namespace["f"](3) == 7


def test_replacement_preserves_literal_values_and_exact_count():
    with pytest.raises(ValueError):
        replace_source("value = old('n ame')", "old('name')", "new('name')")
    with pytest.raises(ValueError):
        replace_source("value = old( 1 ) + old( 1 )", "old(1)", "new(1)")


def test_multiline_replacement_keeps_function_scope():
    source = (
        "def f():"
        + chr(10)
        + "    values = []"
        + chr(10)
        + "    values.append( 'old' )"
        + chr(10)
        + "    return values"
        + chr(10)
    )
    result = replace_source(
        source,
        "values.append('old')",
        "values.append('first')" + chr(10) + "values.append('second')",
    )
    namespace = {}
    exec(result, namespace)
    assert namespace["f"]() == ["first", "second"]


def test_assignment_hook_does_not_need_a_comment_marker():
    source = (
        "def f():"
        + chr(10)
        + "    report = {}"
        + chr(10)
        + "    audit_folder = 'folder'"
        + chr(10)
        + "    return report"
        + chr(10)
    )
    result = insert_before_assignment(
        source, "audit_folder", "report['checked'] = True"
    )
    namespace = {}
    exec(result, namespace)
    assert namespace["f"]() == {"checked": True}
    with pytest.raises(ValueError):
        insert_before_assignment(source, "absent", "pass")
