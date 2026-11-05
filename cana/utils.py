import copy
import warnings

from cana.cutils import binstate_to_statenum, flip_binstate_bit  # noqa: F401


def flip_bitset_in_strstates(strstates, idxs):
    """Flips the binary value for a set of bits in a binary state.

    Args:
        binstate (string) : The binary state to flip.
        idxs (int) : The indexes of the bits to flip.

    Returns:
        (list) : The flipped states

    Example:
        >>> flip_bit_in_strstates('000',[0, 2])
        ['100','001']
    """
    return [flip_binstate_bit(strstates, idx) for idx in idxs]


def flip_binstate_bit_set(binstate, idxs):
    """Flips the binary value for a set of bits in a binary state.

    Args:
        binstate (string) : The binary state to flip.
        idxs (int) : The indexes of the bits to flip.

    Returns:
        (list) : The flipped states
    """
    flipset = []
    if len(idxs) != 0:
        fb = idxs.pop()
        flipset.extend(flip_binstate_bit_set(binstate, copy.copy(idxs)))
        flipset.extend(
            flip_binstate_bit_set(flip_binstate_bit(binstate, fb), copy.copy(idxs))
        )
    else:
        flipset.append(binstate)
    return flipset


def negate_LUT_input(outputs, idx):
    """Negate one LUT input; deprecated alias of :meth:`BitLut.negate_input`.

    .. deprecated::
        Use :meth:`cana.function_outputs.bit_lut.BitLut.negate_input` instead.

    Args:
        outputs (list) : The output list defining the LUT.
        idx (int) : The index of the input to negate.

    Returns:
        (list) : The new output with input idx negated
    """
    warnings.warn(
        "cana.utils.negate_LUT_input is deprecated; use BitLut.negate_input instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from cana.function_outputs.bit_lut import BitLut

    if isinstance(outputs, BitLut):
        return outputs.negate_input(idx).to_list()
    return BitLut.from_sequence(outputs).negate_input(idx).to_list()


def print_logic_table(outputs):
    """Print a LUT; deprecated alias of :meth:`BitLut.lut_as_str`.

    .. deprecated::
        Use :meth:`cana.function_outputs.bit_lut.BitLut.lut_as_str` instead.

    Args:
        outputs (list) : The transition outputs of the function.

    Example:
        >>> print_logic_table([0,0,1,1])
        00 : 0
        01 : 0
        10 : 1
        11 : 1
    """
    warnings.warn(
        "cana.utils.print_logic_table is deprecated; use BitLut.lut_as_str instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from cana.function_outputs.bit_lut import BitLut

    if isinstance(outputs, BitLut):
        lut = outputs
    else:
        lut = BitLut.from_sequence(outputs)
    print(lut.lut_as_str())


def function_monotone(outputs, method="exact", nsamples=100, random_seed=None):
    """Deprecated alias of :func:`~cana.function_properties.unate.is_function_monotone_increasing`.

    .. deprecated::
        Use :func:`cana.function_properties.is_function_monotone_increasing` instead.

    ``method``, ``nsamples``, and ``random_seed`` are ignored (legacy signature).
    """
    warnings.warn(
        "cana.utils.function_monotone is deprecated; "
        "use cana.function_properties.is_function_monotone_increasing instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from cana.function_outputs.bit_lut import BitLut
    from cana.function_properties import is_function_monotone_increasing

    lut = outputs if isinstance(outputs, BitLut) else BitLut.from_sequence(outputs)
    return is_function_monotone_increasing(lut)


def input_monotone(outputs, input_idx, activation=1):
    """Deprecated alias of per-input positive/negative unate checks.

    .. deprecated::
        Use :func:`cana.function_properties.is_input_positive_unate` or
        :func:`cana.function_properties.is_input_negative_unate` instead.

    Example:
        >>> input_monotone([0,1,0,0], 0, activation=1) == False
        >>> input_monotone([0,1,0,0], 0, activation=-1) == True
    """
    warnings.warn(
        "cana.utils.input_monotone is deprecated; "
        "use cana.function_properties.is_input_positive_unate / "
        "is_input_negative_unate instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from cana.function_outputs.bit_lut import BitLut
    from cana.function_properties import (
        is_input_negative_unate,
        is_input_positive_unate,
    )

    lut = outputs if isinstance(outputs, BitLut) else BitLut.from_sequence(outputs)
    if activation == 1:
        return is_input_positive_unate(lut, input_idx)
    if activation == -1:
        return is_input_negative_unate(lut, input_idx)
    raise ValueError("activation must be 1 or -1; got {0!r}".format(activation))
