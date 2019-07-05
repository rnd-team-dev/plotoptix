"""
Install scripts for PlotOptiX extras.

Documentation: https://plotoptix.rnd.team
"""

import sys

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "denoiser":
            from plotoptix.install.denoiser import install_denoiser
            result = install_denoiser()
        else:
            print("Unknown package name %s." % sys.argv[1])

        if not result:
            print("Package not installed.")
        exit(0)

    print("Usage: python -m plotoptix.install [package].")

if __name__ == '__main__':
    main()
