# microglia segmentation

Set of algorithms to segment microglia cells from mouse brain shots. 
(There is a ton of science behind how those pictures are taken. This repo has nothing to do with that.) 

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


### Publications 

This software was used to produce results presented on several scientific conferences.
Here is the list of some publications:

Glyavina M.M., Loginov P.A., Dudenkova V.V., Reunov D.G., Karpova A.O., Prodanets N.N., Korobkov N.A., Zhuchenko M.A., Schelchkova N.A., Mukhina I.V.   
Carbamylated darbepoetin (CdEPO) impact on microglia activation in a model of occlusion of the middle cerebral artery.   
*"XIV International Interdisciplinary Congress &quot;Neuroscience for Medicine and Psychology"*  
2018 May 30- June 10 p. 159  
  
Glyavina M.M., Loginov P.A., Dudenkova V.V., Shirokova O.M., Reunov D.G., Karpova A.O., Prodanets N.N., Korobkov N.A., Zhuchenko M.A., Mukhina I.V.   
Application of laser scanning confocal fluorescent microscopy for visualization microglia morphology in mouse local cerebral ischemia.   
*"26th International Conference on Advanced Laser Technologies"* [ALT&#39;18].  
2018 September 9-14: B-P-2.
  
Glyavina M.M., Loginov P.A., Dudenkova V.V., Shirokova O.M., Reunov D.G., Karpova A.O., Prodanets N.N., Korobkov N.A., Zhuchenko M.A., Mukhina I.V.  
Morphological analysis of microglia in early postischemic period in the mouse local cerebral ischemia.  
*"TERA — 2018, 3rd International Conference Terahertz and Microwave Radiation: Generation, Detection and Applications"*  
2018 October 22—25: S10, P.10.9
