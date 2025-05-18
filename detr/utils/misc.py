def add_tab_to_lines(input_string: str, column_name: str = "", tab: str = "\t") -> str:
    """Adds a tab to each line of a multiline string."""
    tabbed_lines = ""
    if column_name:
        tabbed_lines = f"{column_name}:\n"
    tabbed_lines += "\n".join(f"{tab}{line}" for line in input_string.splitlines())
    return tabbed_lines
