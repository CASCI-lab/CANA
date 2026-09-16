from setuptools import Extension, setup

extensions = [
    Extension("cana.cutils", ["cana/cutils.c"]),
    Extension(
        "cana.canalization.cboolean_canalization",
        ["cana/canalization/cboolean_canalization.c"],
    ),
]


setup(
    ext_modules=extensions,
)
