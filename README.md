# microglia segmentation

Set of algorithms to segment microglia cells from mouse brain fixed in thin paraffin slises shots. 

Algorithm receives raw images as input and finds microglias borders and hull convex for each cell. 

All data is reported in csv files generated for each image. 
Like:

```
#, Size, HullSize, Perimeter
 1, 8298, 26190.5, 1658
 2, 3544, 8829.5,  958
 3, 3378, 19413.5, 1197
 4, 3130, 11644.5,  826
 5, 1899, 9405.5,  752
...
```

Images themselves look shomething like this:

### Unprocessed picture
<a href="https://imgbb.com/"><img src="https://image.ibb.co/fJKDep/demo_raw.png" alt="demo raw" border="0" /></a>


### Segmented picture
<a href="https://imgbb.com/"><img src="https://image.ibb.co/jTFas9/demo_processed.png" alt="demo processed" border="0" /></a>
