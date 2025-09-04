"""Test file to trigger pre-commit hooks."""


def test_function_with_types(x: int) -> str:
    """Test function with type annotations."""
    return str(x * 2)
