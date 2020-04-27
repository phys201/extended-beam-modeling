### Importing BICEP/Keck Data

Here I show how to take a BICEP/Keck TOD and save it in a way that:

1. `io.py` can handle
2. makes the file less than 100MB so that it can be put on GitHub

First we simply load a calibrated TOD file.  Then we keep only the detector feedback values for the single pair that we want as well as the telescope pointing timestreams (both horizon and celestial coords).  We also want to tack on the auxillary info for the pair from the `p` struct given from `get_array_info`, as well as a time vector.  When saving we also want to bring along the scans struct `fs`.  And don't forget to save with the `-v7.3` flag!

Below is a MATLAB code snippet I used to save the first sample Keck TOD we used:

```
load('data/real/201505/20150531C01_dk068_tod.mat')
d0 = d;
clear d;
d.fb = d0.mce0.data.fb(:,[det_a, det_b]);
d.pointing = d0.pointing;
d.utcfast = d0.antenna0.time.utcfast;

% include abscal when getting array info
[p ind] = get_array_info(20150531,[],[],[],[],[],[],'ideal');
p = rmfield(p,'expt');
p = structcut(p, [det_a, det_b]);
d.p = p;

save(your_directory/20150531C01_dk068_tod_singlepair.mat','d','fs','-v7.3')
```

And that should do it! 
As you can see the changes are minimal since we want this code to work with real B/K data.  We just have to navigate some hoops to trim the file size down and get the detector info (like `r` and `theta`).

-TSG