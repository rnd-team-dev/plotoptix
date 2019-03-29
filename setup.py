from setuptools import setup, find_packages

setup(name='plotoptix',
      version='0.1.1.1',
      url='https://github.com/robertsulej/plotoptix',
      author='Robert Sulej, R&D Team',
      author_email='contact@rnd.team',
      description='Dataset visualisation in Python based on NVIDIA OptiX raytracing framework.',
      keywords="gpu nvidia optix raytracing pathtracing visualisation generative plot",
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
          'logging>=0.4',
          'numpy>=1.0',
          'Pillow>=5.3'
      ],
      long_description=open('README.md').read(),
      include_package_data=True,
      exclude_package_data={'': ['README.md']},
      test_suite='nose.collector',
      tests_require=['nose>=1.0'],
      zip_safe=False)
