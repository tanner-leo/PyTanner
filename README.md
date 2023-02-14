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

## Other Information
### Inset Plots

Description of code to use to make an inset plot on any plot.
~~~
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from mpl_toolkits.axes_grid1.inset_locator import mark_inset

x1 = 500
x2 = 1600

y1 = 0.3e6
y2 = 0.5e6

axins = inset_axes(ax, 3,2, loc=1)
axins.plot(item.data.Time, item.data[c])

axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
plt.xticks(visible=False)
plt.yticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.draw()

~~~





