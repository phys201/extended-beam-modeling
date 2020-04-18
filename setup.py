from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='xbmodeling',
      version='0.1',
      description='',
      long_description=readme(),
      classifiers=[
        'License :: GNU GPL v3 License',
        'Programming Language :: Python :: Python 3',
        'Topic :: Statistics :: Bayesian Analysis',
      ],
      url='http://github.com/phys201/extended-beam-modeling',
      author='',
      author_email='',
      license='Gnu GPL v3',
      packages=['xbmodeling'],
      install_requires=[
          'numpy',
          'healpy',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)

