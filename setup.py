from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

import struct, os, platform, subprocess

def HandlePrerequisites(command_subclass):

    base_run = command_subclass.run

    def testPython64b(self):
        if struct.calcsize("P") * 8 != 64:
            print(80 * "*"); print(80 * "*")
            print("Python 64-bit is required.")
            print(80 * "*"); print(80 * "*")
            raise ValueError

    def findCuda(self, p, quiet):
        cuda_major = -1
        cuda_minor = -1
        try:
            try:
                outp = subprocess.check_output(["nvcc", "--version"]).decode("utf-8").split(" ")
            except FileNotFoundError:
                if p == "Linux": outp = subprocess.check_output(["/usr/local/cuda/bin/nvcc", "--version"]).decode("utf-8").split(" ")
                else: raise

            idx = outp.index("release")
            if idx + 1 < len(outp):
                rel = outp[idx + 1].strip(" ,")
                cuda_major = int(rel.split(".")[0])
                cuda_minor = int(rel.split(".")[1])
                print("OK: found CUDA %s" % rel)
            else:
                if not quiet: raise ValueError

        except FileNotFoundError:
            if not quiet:
                print(80 * "*"); print(80 * "*")
                print("Cannot access nvcc. Please check your CUDA installation and/or PATH variable.")
                print(80 * "*"); print(80 * "*")
                raise

        except ValueError:
            if not quiet:
                print(80 * "*"); print(80 * "*")
                print("CUDA release not recognized.")
                print(80 * "*"); print(80 * "*")
                raise

        except Exception as e:
            if not quiet:
                print("Cannot verify CUDA installation: " + str(e))
                raise

        return cuda_major, cuda_minor

    def prepareCudaLinux(self, cuda_major, cuda_minor, removing):
        if hasattr(self, 'install_lib') and self.install_lib is not None: lib_path = self.install_lib
        else: lib_path = os.getcwd()
        src = os.path.join(lib_path, "plotoptix", "bin", "librndSharpOptiX_%s_%s.so" % (cuda_major, cuda_minor))
        dst = os.path.join(lib_path, "plotoptix", "bin", "librndSharpOptiX.so")
        if os.path.isfile(dst): os.remove(dst)
        if not removing and os.path.isfile(src): os.symlink(src, dst)

    def subclass_run(self):

        #print(self.user_options)
        if hasattr(self, 'uninstall') and self.uninstall == 1: removing = True
        else: removing = False

        if hasattr(self, 'install_lib') and self.install_lib is not None: lib_path = self.install_lib
        else: lib_path = os.getcwd()

        testPython64b(self)

        p = platform.system()

        #cuda_major, cuda_minor = findCuda(self, p, removing)

        if p == "Windows":
            base_run(self)
        elif p == "Linux":
            base_run(self)
        else:
            raise NotImplementedError

        print("All correct.")

    command_subclass.run = subclass_run
    return command_subclass


@HandlePrerequisites
class HandlePrerequisitesDevelop(develop):
    pass

@HandlePrerequisites
class HandlePrerequisitesInstall(install):
    pass


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):

        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

        def get_tag(self):
            python, abi, plat = _bdist_wheel.get_tag(self)
            python, abi = 'py3', 'none'
            return python, abi, plat

except ImportError:
    bdist_wheel = None


setup(name='plotoptix',
      version='0.7.1',
      url='https://rnd.team/project/plotoptix',
      project_urls={
          'Documentation': 'https://plotoptix.rnd.team',
          'Examples': 'https://github.com/rnd-team-dev/plotoptix/tree/master/examples',
          'Source': 'https://github.com/rnd-team-dev/plotoptix',
          },
      author='Robert Sulej, R&D Team',
      author_email='dev@rnd.team',
      description='Data visualisation in Python based on NVIDIA OptiX ray tracing framework.',
      keywords="gpu nvidia optix ray-tracing path-tracing visualisation generative plot animation real-time",
      cmdclass={
          'bdist_wheel': bdist_wheel,
          'install': HandlePrerequisitesInstall,
          'develop': HandlePrerequisitesDevelop
      },
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Win32 (MS Windows)',
          'Environment :: X11 Applications',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3 :: Only',
          'Intended Audience :: Education',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Science/Research',
          'License :: Free for non-commercial use',
          'Natural Language :: English',
          'Topic :: Multimedia :: Graphics :: 3D Rendering',
          'Topic :: Scientific/Engineering :: Visualization',
          'Topic :: Artistic Software'
      ],
      packages=find_packages(exclude=['tests']),
      install_requires=[
          'enum34;python_version<="3.4"',
          'packaging>=18.0',
          'numpy>=1.0',
          'Pillow>=5.3',
          'python-dateutil>=2.7',
          'matplotlib>=2.0'
      ],
      long_description=open('README.rst').read(),
      include_package_data=True,
      exclude_package_data={
          '': [
              'README.rst',
              'README.md'
              ]
          },
      test_suite='nose2.collector.collector',
      tests_require=['nose2>=0.9'],
      zip_safe=False)

