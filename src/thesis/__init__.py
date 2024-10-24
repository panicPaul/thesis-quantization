"""
Init file for the package. I need to explicitly define it in order to run beartype on
this package.
"""

import sys

from beartype import BeartypeConf, BeartypeStrategy, beartype
from beartype.claw import beartype_this_package

# disable beartype if the optimization flag is set
if not sys.flags.optimize:
    beartype_this_package()

# disable beartype for a specific function
nobeartype = beartype(conf=BeartypeConf(strategy=BeartypeStrategy.O0))
