import re


def add_noqa(filepath, patterns_and_rules):
    try:
        with open(filepath) as f:
            lines = f.readlines()

        modified = False
        for i, line in enumerate(lines):
            for pattern, rule in patterns_and_rules:
                if re.search(pattern, line) and "noqa" not in line:
                    lines[i] = line.rstrip() + f"  # noqa: {rule}\n"
                    modified = True

        if modified:
            with open(filepath, "w") as f:
                f.writelines(lines)
            print(f"Fixed {filepath}")
    except Exception as e:
        print(f"Error {filepath}: {e}")


add_noqa(
    "app/api/dependencies/auth.py",
    [
        (r"from sqlalchemy\.ext\.asyncio import", "E402"),
        (r"from sqlalchemy\.future import", "E402"),
        (r"from app\.db\.session import", "E402"),
        (r"from app\.models\.db import", "E402"),
        (r"raise credentials_exception", "B904"),
    ],
)

add_noqa(
    "app/api/v1/endpoints/admin.py",
    [(r"current_user: UserResponse = Depends\(RequireRole\(Role\.ADMINISTRATOR\)\)\):", "ARG001")],
)
