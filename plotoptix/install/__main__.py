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
        elif sys.argv[1] == "examples":
            from plotoptix.install.project import install_project
            result = install_project("examples.zip", "1Bdq7SnvI3fA12_-LoaF31h-d5E67T_32")
        elif sys.argv[1] == "moon":
            from plotoptix.install.project import install_project
            result = install_project("moon.zip", "1yUZMskZzKiAQmjW7E3scGy-b_rUVdoa4")
        else:
            print("Unknown package name %s." % sys.argv[1])

        if not result:
            print("Package not installed.")
        exit(0)

    print("Usage: python -m plotoptix.install [package].")

if __name__ == '__main__':
    main()
