# E-Chem Tools V1.0!!
*Author: Tanner Leo*
*Date: 12.5.22*

A collection of tools possibly of use for collecting, analyzing, and exporting data collected from EC-lab. 


## Installation:

Paste the code below into your jupyter notebook.

~~~
pip install git+https://github.com/tanmann13/first_package.git
~~~

Now, you can import to toobox into your notebook.

~~~
import PyTanner as pt
~~~



## Functions
### **Importing Functions**
---
**folder2files**

*arguments*: 

folder - folder to read

*Returns*: a list of files in the specified folder

---

**process**

*arguments*: 

files - list of files (.mpr files)

*Returns*: a dataframe containing metadata, and data

---

**importtxtPEIS**

*arguments*: 

paths - a list of file names (the output of folder2files)

filter='' - a string used to match against the names in paths. Includes the filer in import if filter is in paths

*returns*: A dataframe of imported files, and their corresponding names

---

**importtxtCA(paths, filt="", nfilt="")**

---

**importtxtCV(paths, debug=False, filt="", nfilt="")**

This function will automatically calculate the scan rate of the run.


### Data Types:

    pt.PEIS(dataframe, optional: name, description)

creates an object that stores a PEIS run

    pt.CP(dataframe, optional: name, description)

creates an object that stores a CP run


    pt.CA(dataframe, optional: name, description)

creates an object that stores a CA run

### Plotting Functions


    pt.CA.plot(*optional: timeunit='s', save=False, figpass = "", cycles=False*)
makes a standard plot 

    
    
    pt.CP.plot(*optional: timeunit='s', save=False, figpass = "", cycles=False*)

makes a standard plot

    pt.PEIS.nyquist()

creates a nyquist plot of PEIS object



