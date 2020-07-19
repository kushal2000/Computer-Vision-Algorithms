## Assignment 4

### Execution of Code

* Execute using ``` python colors.py ```
* You will be first prompted to choose a diagonal on source image. Choose two points, top left to bottom right of the rectangle you want to specify.

** There will be 5 active windows at the end:
* Source -> Displays Source Image with ROI highlighted
* Target -> Displays Target Image with ROI highlighted
* Source Mask -> Displays Source Image Mask with dominant colour pixels marked white
* Target Mask -> Displays Source Image Mask with dominant colour pixels marked white
* Result Image -> Displays Target Image with dominant colour transfer from Source ROI to dominant colour extracted from Target ROI

### GUI and Region of Interest Selection

* This is done using MouseCallback and other standard OpenCV tools like line and rectangle in the functions chooseRectangle(), Choose_Point() and draw_point()

### Dominant Colour Extraction

* Dominant Colour is found out in the domColour() function
* A 2-D Histogram of the Hue/Saturation Space is plotted for the input image. As default, Hue is broken into 6 bins, Saturation is broken into 8 bins
* Max value of the histogram is chosen as dominant colour and a mask is suitably applied to input image
* Range of Hue/Saturation for dominant colour and the masked image is returned

### Transfer of Dominant Colour

* This is done in the transfer() function
* It takes input of the source dominant colour and range of target dominant colour
* All pixels within this range are found out and replaced


