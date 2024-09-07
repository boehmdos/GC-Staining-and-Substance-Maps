# GC Staining and Substance Maps
See your GC data better! Adds color to gas chromatograms according to their mass spectra (GC Staining). Sorts mass spectra by similarity on a plane (Substance Map).

# Prerequisites
To work properly, the script needs the **FullColorWheel.png** in the same folder and **a map of mass spectra** (a SOM). An extensive SOM can be downloaded at https://doi.org/10.5281/zenodo.13710838. It is built from spectra available from the MassBank of North America (Mona, used under a CC-BY 4.0 license, https://mona.fiehnlab.ucdavis.edu) and covers a wide range of typical GC analytes.

Gas chromatograms with mass spectra must be saved as text before they can be processed by the script. We use OpenChrom to do this - https://www.openchrom.net/ 

The **expected structure** of the csv with GC data is:

		RT(milliseconds),RT(minutes),Retention Index (not used),m/z of detected masses (integer)

Labelling the columns explicitly in the header is not required.

For example, the first lines of an GC input might look like this (Openchrom adds the " - NOT USED BY IMPORT", this does not affect the script):

		RT(milliseconds),RT(minutes) - NOT USED BY IMPORT,RI,50,51,52, ... ,448,449,450
		
		125508,2.0918,0.0,0.0,0.0,0.0, ... ,0.0,0.0,0.0
		
		125787,2.09645,0.0,0.0,36.170532,36.22905, ... ,0.0,0.0,0.0
		
		126067,2.1011166666666665,0.0,245.98819,0.0,21.675735, ... ,0.0,0.0,0.0
  
  		...

		1080512,18.008533333333332,0.0,0.0,0.0,2.4189453, ... ,0.0,0.0,0.0





# Version Info
  
The script file is written on python 3.9.13

It uses following modules (tested on 05/09/2024):

- FreeSimpleGUI	5.1.1 

- colorsys

- math

- matplotlib	3.7.1

- numpy	1.24.3	

- os

- pandas	2.0.1	

- scipy	1.10.1	

- time

# What we are dreaming of

Speed. We are chemists and did our best - still this is a slow script. And it needs a lot of memory.

Batch processing and preparing the stain and map automatically right after an analysis is finished

Reading native files from GC manufacturers

Defining GC staining recipes for the first stain

Open output files to plot and continue working with them



# Main Code - What does it do?

The way it works is by first taking the raw data from a GC-MS, those need to be in a *.csv format saved from Openchrom.  The user has the option to choose between a few parameters:

1. *.csv file of choise, where you should know if the first row contains a string of data and the delimiter used.
2. 
3. m/z range to be analyzed within the range of [29,200].  The user can manually select a certain range within the limits or select "Auto".  In the last case, the programm will 
automatically fix the data to fit the range as best as possible.

	for example: GC-MS data are of range 50-450, if "Auto" is selected, then only the values 50-200 will be taken into account.

4. retention time in minutes.  The user can manually select a certain time range or select "All".  In the first case, the programm will assign a RT that is closed to the user's
input both for the start and the end.  In the last case, the programm will take into account the whole time range of the chromatogram.

	for example:  User's input is "From: 15" "To: 20".  The GC-MS data look something like that:

		RT[min]

		...
   
		14.99976635
   
		15.00918293
   
		...
   
		19.99468422
   
		20.00410080
   
	The time range that is going to be analyzed is [14.99976635, 19.99468422] in minutes.


!! It is advised to select a time range that excludes the column's decomposition !!

5. minimum TIC intensity, which zeros values of TIC below that and doesn't assign them a color (meaning they are white).

6. exponent of the power function which is used to adjust the luminance value to be more sensitive for small peaks.  For that reason only positive values less or equal to 1 can be used.

7. output folder for the processed data to be saved.

8. (optional) chromatogram and BMU map formatting to be saved directly.  The options of *.png and *.svg are given for the chromatogram and the option of *.png is given for the BMU map.

Basically the first few lines of code take care of the user's input, checking if correct and shaping raw data.

Normalizing the MS is done to get a vector, the size of the m/z range for each retention time, with values [0,1].

Some information on the SOM is required to better understand the code.  So the SOM is a 3D object.  The xy plane extents from 0 to 256 and is what the BMU map shows.  The z dimension
contains the vectors, called weights.  That means that we have 256*256 unique z_vectors.  Each vector has elements of value between [0,1] for each m/z in range [29,200].  To try to 
simplify that for the GC-MS data used in the previous examples, each weight will have 151 elements.

The SOM is loaded from a text file.  Now a SOM of 256x256x172 is beeing used.  The way that the text file is written has all 256x256 elements for m/z = 29, then 65536 elements for 
m/z = 30 etc.  If you wish to change the SOM used, then make sure the *.txt file has the same formatting.
For a SOM of different dimensions adjust the uppercase code or remove them entirely. 

  045:	[sg.Text("Input SOM dimensions:"), sg.Text("x:"), sg.Input('X DIMENSION', size=(8, 20)), sg.Text("y:"), sg.Input('Y DIMENSION', size=(8, 20))],
  
  057:    [sg.Text("Define new center:"), sg.Text("x:"), sg.Input('XCENTER(NUMBER)', size=(5, 15)), sg.Text("y:"), sg.Input('YCENTER(NUMBER)', size=(5, 15))],
  
  058:	[sg.Text("Define new radius from center (r<VALUE OF RMAX(NUMBER)):"), sg.Text("r:"), sg.Input('VALUE OF RMAX(NUMBER)', size=(5, 15))]

Next m/z values from the z_vectors are clipped off, according to selected m/z range.  This is done to get rid off unnecessary data and allocate less memory.

Finding the BMU for each MS follows the logic: Both weights and MS hold vectors of the same size.  So in order to find the BMU for each MS all that has to be done is to find the
weight with the least difference.  That's what the for loop does in lines 214-218, plus saving that minimum difference to a list and the index of the BMU in the z_matrix.

From the BMU index it's easy to determine the xy coordinates (BMU coordinates) with the help of modulus and intiger division.  

Determining the color according to BMU coordinates.  Using the Cartesian coordinate system, the BMU coordinates can be turned into vectors from point O(0,0) to each respective BMU (x,y)
point.  Same goes for the center point, which is 127.5, 127.5 and not 128.5, 128.5 because python has always an element in position 0. 

The radius is determined by fitting the square SOM in a cirle, meaning that the diameter will be the diagonal distance of the square.  The radius is half the diameter, so using 
Pythagorean theorem is given by sqrt(xcenter^2 + ycenter^2).  

Stain is the result of subtraction between each BMU's and center's vectors, that makes it an array of vectors all starting from the center.

Saturation is determined by finding the length of each stain, which is equivalent to a radius of a circle, and dividing by r_max to normalize values in [0,1].

Hue is given by the phase. The angle between two vectors is given by phi = arccos ( a•b / |a||b|).  Unit vectors are being used, so |a||b| = 1 and for reference the unit_north vector
is created.  Since the arccos() can not differentiate between left and right from the y axis, a way around that problem is to handle vectors according to their relevant position to 
the center. The phase is normalized [0,1] before passed to Hue.

Luminance is given by the total ion current (TIC), divided by the maximum TIC value to normalize in [0.5,1].  A baseline of Luminance=0.5 is set and the scale is set so high TIC get 
values close to 0.5 and low TIC close to 1.0, making them less visible when plotted in the chromatogram.  Also a power function is used to adjust the Luminance value to be more 
sensitive for small peaks.  It is especially usefull for dynamic chromatograms.  For that reason only positive values less or equal to 1 can be used.

Finally every MS is assigned a HSL triplet of values.

Converting HSL to RGB is done by colorsys.hls_to_rgb(), returning a list which later is converted to an array ( ,3)

Determining BMU intensity.  Similar MS may have the same BMU, that makes some BMU points to have multiplicity.  In order to quantify the effect of each BMU group of points, a total TIC
is calculated for each group.  The higher the sum of TIC, the more that group contributes in the chromatogram.  A grayscale gradient is used to show exactly that, as one can later see
on the BMU map.  All this allow us to have a BMU fingerprint of each sample, depending on the experimental protocol that it was used.

Getting the saving location for the exported *.csv file and the figures, if the user has selected the specific option.

The output file is created using pandas, since it was found easier to handle a *.csv file than numpy or csv.  It includes everything that the user selected at the GUI (input 
file, m/z range, time range, noise limit, exponent) plus the date that the output file was created, followed by the retention time in [ms] and [min], TIC and TIC denoised, the 
difference between each MS vector and its BMU, the BMU coordinates, the HSL values and the RGB values.

Last part of the main function of the code is the plotting of the graphs.  First the chromatogram is being plotted as y=f(TIC) using fill between for x, so each interval between 
measured retention times is colored.  Secondly the BMU map, which is just the BMU coordinates on the xy plane.  An image of RGB plane is set as background to easily correlate each 
point to the color of the peak.

## Staining Recipes


The user is asked if wished to continue on staining recipes, meaning analyzing only a small segment of the BMU map.

The new parameters that are requested are:

1. center point of the smallest circular area that is going to be further analyzed
 
2. radius of the circular area
 
3. rotation of the color plane.  Rotates anti-clockwise and the given value must be in degrees.

4. exponent of the power function which is used to adjust the luminance value to be more sensitive for small peaks.  For that reason only positive values less or equal to 1 can be used.

5. (optional) chromatogram and BMU stained map formatting to be saved.  The options of *.png and *.svg are given for the chromatogram and the option of *.png is given for the BMU 
stained map.

A preview of the area defined by the new parameters can be obtained by first clicking on "Check position".  The cirle inticates all the BMU points that are targated.  The circle can 
have a center outside of the colored plane and make a selection adjusting the radius.  By changing the paramaters and clicking on "Check position" the user can navigate on the BMU map
until they are satisfied with their selection.  Once that is done, they can continue by clicking on "Submit and stain".  
From there the same process is being done to assign a color, only this time to the selected data.

Next, a square is created with edge equal to the selected area's diameter.  This is done to make a first selection of the BMU points of interest, since it is easier to work with 
renctagulars, which resemble matrices, than cirles.  The indices for both inside and outside the staining area are saved to later give outside BMU points white color.

Applying the same method to determine color for each BMU.  The only difference is now on the Saturation.  BMU points ouside the cirle but inside the square will have a Saturation > 1.

This is the key to make our final selection of the BMU points of interest, by saving their index in the xy_user matrix.
Finally all outside points are colored white by making the Luminace = 1.

Follows the conversion from HSL to RGB, then saving also the staining recipe and finally the plotting of the final 2 graphs.

The final output *.csv file contains now also the center and radius of the stain, the rotation of the color plane, the exponent of the power function for the stain and the HSL and
RGB values of the stained area.
