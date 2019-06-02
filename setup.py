from setuptools import setup, find_packages

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
      version='0.2.2',
      url='https://plotoptix.rnd.team',
      project_urls={
          'Documentation': 'https://plotoptix.rnd.team',
          'Examples': 'https://github.com/rnd-team-dev/plotoptix/tree/master/examples',
          'Source': 'https://github.com/rnd-team-dev/plotoptix'
          },
      author='Robert Sulej, R&D Team',
      author_email='dev@rnd.team',
      description='Data visualisation in Python based on NVIDIA OptiX ray tracing framework.',
      keywords="gpu nvidia optix ray-tracing path-tracing visualisation generative plot animation real-time",
      cmdclass={'bdist_wheel': bdist_wheel},
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Win32 (MS Windows)',
          'Operating System :: Microsoft :: Windows',
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
          'enum34;python_version<"3.4"',
          'packaging>=18.0',
          'numpy>=1.0',
          'Pillow>=5.3',
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
