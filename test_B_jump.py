## Calculate jump on B on the surface
## 1. Calculate the B total = B plasma + B external from VMEC output
## 2. Calculate the B external from the B total using virtual casing
## 3. B total outside - B total inside = (B plasma outside + B biot-savart on the surface)^2 - B vmec surface^2
##                                     = (B vmec surface - B external + B biot-savart on the surface)^2 - B vmec surface^2
##                                     = (Bbiosavart - Bexternal) (Bbiosavart - Bexternal + 2 Bvmecsurface)