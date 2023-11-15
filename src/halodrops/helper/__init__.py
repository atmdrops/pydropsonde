def get_bool(s):
    if isinstance(s, bool):
        return s
    elif isinstance(s, int):
        return bool(s)
    elif isinstance(s, str):
        lower_s = s.lower()
        if lower_s == "true":
            return True
        elif lower_s == "false":
            return False
        elif lower_s in ["0", "1"]:
            return bool(int(lower_s))
        else:
            raise ValueError(f"Cannot convert {s} to boolean")
    else:
        raise ValueError(f"Cannot convert {s} to boolean")
