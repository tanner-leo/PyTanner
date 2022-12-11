import setuptools

with open("README.md","r",encoding="utf-8") as fh:
    long_description=fh.read()

setuptools.setup(
    name='PyTanner',
    version='0.1.3',
    author='Tanner Leo',
    description='Various Electrochemistry Related Functions',
    long_description = long_description,
    long_description_content_type="text/markdown",
    packages=['PyTanner'],
    install_requires=['pandas','matplotlib','numpy','joblib','tqdm'],
    url='https://github.com/tanmann13/EchemTools.git'
)