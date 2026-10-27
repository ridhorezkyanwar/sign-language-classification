"""Labels used by the Sign-MNIST model's original 25-output layer."""

LABELS = {
    **{index: chr(ord("A") + index) for index in range(9)},
    **{index: chr(ord("A") + index) for index in range(10, 25)},
}
UNSUPPORTED_LABEL_INDEX = 9


def display_label(index: int) -> str:
    """Return a human-readable label without shifting the missing J class."""
    if index == UNSUPPORTED_LABEL_INDEX:
        return "Kelas J tidak tersedia"
    return LABELS.get(index, f"Kelas {index}")


def is_supported_label(index: int) -> bool:
    """Return whether the index corresponds to a supported static sign class."""
    return index in LABELS
