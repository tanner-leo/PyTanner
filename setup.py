import PyTanner

with open("README.md","r",encoding="utf-8") as fh:
    long_description=fh.read()

PyTanner.setup(
    name='PyTanner',
    version='0.0.1',
    author='Tanner Leo',
    description='Various Electrochemistry Related Functions',
    long_description = long_description,
    long_description_content_type="text/markdown",
    packages=['PyTanner'],
    install_requires=['pandas','matplotlib','numpy']
)