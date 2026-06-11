# Local mirror of the real Perturb subnet's `perturbnet` package.
# These modules are copied/trimmed from the live repo so that flipper.py and
# flipper-base.py import the EXACT same names they would in production
# (perturbnet.model, perturbnet.image_io, perturbnet.constants). That means the
# engine code here is byte-for-byte copy-paste-able into the real miner.
