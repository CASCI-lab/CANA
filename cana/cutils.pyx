import random
from itertools import product, zip_longest


def recursive_map(f, d):
    """Normal python map, but recursive

    Args:
        f (function) : a function to be applied to every item of the iterable
        d (iterable) : the iterable to which f will be applied itemwise.
    """
    return [not hasattr(x, "__iter__") and f(x) or recursive_map(f, x) for x in d]


def binstate_to_statenum(binstate):
    """Converts from binary state to state number.

    Args:
        binstate (string) : The binary state.

    Returns:
        int : The state number.

    Example:

        .. code-block:: python

            '000' -> 0
            '001' -> 1
            '010' -> 2 ...

    See also:
        :attr:`statenum_to_binstate`, :attr:`statenum_to_density`
    """
    return int(binstate, 2)


def statenum_to_binstate(statenum, base):
    """Converts an interger into the binary string.

    Args:
        statenum (int) : The state number.
        base (int) : The binary base

    Returns:
        string : The binary state.

    Example:

        .. code-block:: python

            0 -> '00' (base 2)
            1 -> '01' (base 2)
            2 -> '10' (base 2)
            ...
            0 -> '000' (base 3)
            1 -> '001' (base 3)
            2 -> '010' (base 3)

    See also:
        :attr:`binstate_to_statenum`, :attr:`binstate_to_density`
    """

    return bin(statenum)[2:].zfill(base)


def binstate_pinned_to_binstate(binstate, pinned_binstate, pinned_var):
    """Combines two binstates based on the locations of pinned variables.

    Args:
        binstate (str) : the binary state of non-pinned variables
        pinned_binstate (str) : the binary states of the pinned variables
        pinned_var (list of int) : the list of pinned variables

    Returns:
        string : The combined binary state.

    See also:
        :attr:'statenum_to_binstate'
    """
    total_length = len(binstate) + len(pinned_binstate)
    new_binstate = list(statenum_to_binstate(0, base=total_length))
    ipin = 0
    ireg = 0
    for istate in range(total_length):
        if istate in pinned_var:
            new_binstate[pinned_var[ipin]] = pinned_binstate[ipin]
            ipin += 1
        else:
            new_binstate[istate] = binstate[ireg]
            ireg += 1
    return ''.join(new_binstate)


def statenum_to_output_list(statenum, base):
    """Converts an interger into a list of 0 and 1, thus can feed to BooleanNode.from_output_list()

    Args:
        statenum (int) : the state number
        base (int) : the length of output list

    Returns:
        list : a list of length base, consisting of 0 and 1

    See also:
        :attr:'statenum_to_binstate'
    """
    return [int(i) for i in statenum_to_binstate(statenum, base)]


def outputs_to_binstates_of_given_type(outputs, output, k):
    """Converts a node output list into a list of binary states given a specific output type.
    For instance, for the `AND` boolean function, with outputs `[0, 0, 0, 1]`
    it returns `['00', '01', '10']` when `output_type = 0` and `['11']` when `output_type = 1`.

    Args:
        outputs (list/str) : the list of outputs of a given node.
        output (int/str) : a certain output type. 0 or 1 in the case of Binary outputs.
        k (int) : input degree

    Returns:
        list : a list containing all the binstates of a certain output type

    """
    return [statenum_to_binstate(statenum, k) for statenum in range(2**k) if outputs[statenum] == output]


def flip_bit(bit):
    """Flips the binary value of a state.

    Args:
        bit (string/int/bool): The current bit position

    Returns:
        same as input: The flipped bit
    """
    if isinstance(bit, str):
        return '0' if (bit == '1') else '1'
    elif isinstance(bit, int) or isinstance(bit, bool):
        return 0 if (bit == 1) else 1
    else:
        raise TypeError("'bit' type format must be either 'string', 'int' or 'boolean'")


def flip_binstate_bit(binstate, idx):
    """Flips the binary value of a bit in a binary state.
        Args:
            binstate (string) : A string of binary states.
            idx (int) : The index of the bit to flip.
        Returns:
            (string) : New binary state.

        Example:

            .. code-block:: python
                flip_bit_in_strstates('000',1) -> '010'
        """
    if idx + 1 > len(binstate):
        raise TypeError("Binary state '{}' length and index position '{}' mismatch.".format(binstate, idx))
    return binstate[:idx] + flip_bit(binstate[idx]) + binstate[idx + 1:]


def statenum_to_density(statenum):
    """Converts from state number to density

    Args:
        statenum (int): The state number

    Returns:
        int: The density of ``1`` in that specific binary state number.

    Example:
        >>> statenum_to_binstate(14, base=2)
        >>> '1110'
        >>> statenum_to_density(14)
        >>> 3
    """
    return sum(map(int, bin(statenum)[2::]))


def binstate_to_density(binstate):
    """Converts from binary state to density

    Args:
        binstate (string) : The binary state

    Returns:
        int

    Example:
        >>> binstate_to_density('1110')
        >>> 3
    """
    return sum(map(int, binstate))


def binstate_to_constantbinstate(binstate, constant_template):
    """
    Todo:
        Documentation
    """
    # This function is being used in the boolean_network._update_trans_func
    constantbinstate = ''
    iorig = 0
    for value in constant_template:
        if value is None:
            constantbinstate += binstate[iorig]
            iorig += 1
        else:
            constantbinstate += str(value)

    return constantbinstate


def constantbinstate_to_statenum(constantbinstate, constant_template):
    """
    Todo:
        Documentation
    """
    binstate = ''.join([constantbinstate[ivar] for ivar in range(len(constant_template)) if constant_template[ivar] is None])
    return binstate_to_statenum(binstate)


def random_binstate(length, random_seed=None):
    """Generates a random binary state of a given length

    Args:
        length (int) : the length of the binary state

    Returns:
        binstate (str) : a random binary state
    """
    random.seed(random_seed)
    return "".join([random.choice(['0', '1']) for bit in range(length)])


def expand_logic_line(line):
    """This generator expands a logic line containing ``-`` (ie. ``00- 0`` or ``0-0 1``) to a series of logic lines containing only ``0`` and ``1``.

    Args:
        line (string) : The logic line. Format is <binary-state><space><output>.

    Returns:
        generator : a series of logic lines

    Example:
        >>> expand_logic_line('1-- 0')
        >>> 100 0
        >>> 101 0
        >>> 110 0
        >>> 111 0
    """
    # helper function for expand_logic_line
    def _insert_char(la, lb):
        lc = []
        for i in range(len(lb)):
            lc.append(la[i])
            lc.append(lb[i])
        lc.append(la[-1])
        return ''.join(lc)

    line1, line2 = line.split()
    chunks = line1.split('-')
    if len(chunks) > 1:
        for i in product(*[('0', '1')] * (len(chunks) - 1)):
            yield _insert_char(chunks, i) + ' ' + line2
    else:
        for i in [line]:
            yield i


def binstate_compare(binstate1, binstate2):
    """Compare each element in two binary states

    Args:
        binstate1, binstate2 : the two binary states to be compared

    Return:
        c (list, bool) : a list of comparisons
    """
    return [(b0 == b1) for b0, b1 in zip_longest(binstate1, binstate2)]


def hamming_distance(s1, s2):
    """Calculates the hamming distance between two configurations strings.

    Args:
        s1 (string): First string
        s2 (string): Second string

    Returns:
        float : The Hamming distance

    Example:
        >>> hamming_distance('001','101')
        >>> 1
    """
    assert len(s1) == len(s2), "The two strings must have the same length"
    return sum([(b0 != b1) for b0, b1 in zip_longest(s1, s2)])
