import ast
import io
import textwrap
import tokenize


def _tokens(source):
    ignored = {
        tokenize.ENCODING,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.COMMENT,
        tokenize.ENDMARKER,
    }
    result = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in ignored:
            continue
        value = token.string
        if token.type == tokenize.STRING:
            try:
                value = (
                    type(ast.literal_eval(value)).__name__,
                    repr(ast.literal_eval(value)),
                )
            except (SyntaxError, ValueError):
                pass
        result.append((token, (token.type, value)))
    return result


def _fragment(source):
    value = textwrap.dedent(source)
    try:
        return ast.parse(value).body
    except (SyntaxError, IndentationError):
        lines = value.splitlines()
        indents = [len(line) - len(line.lstrip()) for line in lines[1:] if line.strip()]
        if lines and indents and min(indents) > 0:
            lines[0] = " " * min(indents) + lines[0]
            return ast.parse(textwrap.dedent(chr(10).join(lines))).body
        raise


def _ast_replace(source, before, after, count):
    wanted = _fragment(before)
    replacement = _fragment(after)
    tree = ast.parse(source)
    matches = []
    if (
        len(wanted) == 1
        and isinstance(wanted[0], ast.Expr)
        and len(replacement) == 1
        and isinstance(replacement[0], ast.Expr)
    ):
        signature = ast.dump(wanted[0].value, include_attributes=False)
        matches = [
            (node, node)
            for node in ast.walk(tree)
            if ast.dump(node, include_attributes=False) == signature
        ]
        rendered = ast.unparse(replacement[0].value)
    else:
        signature = [ast.dump(node, include_attributes=False) for node in wanted]
        for parent in ast.walk(tree):
            for _, value in ast.iter_fields(parent):
                if (
                    not isinstance(value, list)
                    or not value
                    or not all(isinstance(node, ast.stmt) for node in value)
                ):
                    continue
                values = [ast.dump(node, include_attributes=False) for node in value]
                for index in range(len(value) - len(wanted) + 1):
                    if values[index : index + len(wanted)] == signature:
                        matches.append((value[index], value[index + len(wanted) - 1]))
        rendered = ast.unparse(ast.Module(body=replacement, type_ignores=[]))
    if len(matches) != count:
        raise ValueError("AST replacement count differs")
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def position(line, column):
        return offsets[line - 1] + len(
            lines[line - 1].encode("utf-8")[:column].decode("utf-8")
        )

    for first, last in sorted(
        matches, key=lambda pair: (pair[0].lineno, pair[0].col_offset), reverse=True
    ):
        start = position(first.lineno, first.col_offset)
        end = position(last.end_lineno, last.end_col_offset)
        parts = rendered.splitlines()
        code = parts[0] + (
            chr(10) + chr(10).join(" " * first.col_offset + line for line in parts[1:])
            if len(parts) > 1
            else ""
        )
        source = source[:start] + code + source[end:]
    ast.parse(source)
    return source


def replace_source(source, before, after, count=1):
    if source.count(before) == count:
        return source.replace(before, after)
    try:
        return _ast_replace(source, before, after, count)
    except (SyntaxError, IndentationError, ValueError):
        pass
    wanted = [signature for _, signature in _tokens(textwrap.dedent(before))]
    tokens = _tokens(source)
    signatures = [signature for _, signature in tokens]
    hits = [
        i
        for i in range(len(tokens) - len(wanted) + 1)
        if signatures[i : i + len(wanted)] == wanted
    ]
    if not wanted or len(hits) != count:
        raise ValueError("Source replacement count differs: " + before[:100])
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    replacement = textwrap.dedent(after)
    for i in reversed(hits):
        first = tokens[i][0]
        last = tokens[i + len(wanted) - 1][0]
        start = offsets[first.start[0] - 1] + first.start[1]
        end = offsets[last.end[0] - 1] + last.end[1]
        parts = replacement.splitlines()
        if len(parts) > 1:
            padding = " " * first.start[1]
            rendered = (
                parts[0] + chr(10) + chr(10).join(padding + part for part in parts[1:])
            )
        else:
            rendered = replacement
        source = source[:start] + rendered + source[end:]
    ast.parse(source)
    return source


def insert_before_assignment(source, name, statement):
    tree = ast.parse(source)
    matches = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
    ]
    if len(matches) != 1:
        raise ValueError("Assignment insertion point differs: " + name)
    node = matches[0]
    lines = source.splitlines(keepends=True)
    lines.insert(node.lineno - 1, " " * node.col_offset + statement + chr(10))
    result = "".join(lines)
    ast.parse(result)
    return result
