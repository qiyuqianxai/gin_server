from setuptools import find_packages, setup
setup(
    name='faceboard',
    packages=find_packages(),
    version='0.1',
    description='faceboard',
    author='Fei',
    license='MIT',
    install_requires=['docker', 'requests'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
    python_requires='>=2.6, !=3.0.*, !=3.1.*, !=3.2.*',
)
