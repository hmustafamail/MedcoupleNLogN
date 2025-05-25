# Medcouple N Log N

- Efficient (N log N) implementation of Medcouple (otherwise quadratic)

## History
- Guy Brys authored an R package for efficient medcouple, c. 2004 ([link](https://search.r-project.org/CRAN/refmans/robustbase/html/mc.html))
- Jordi Gutiérrez Hermoso used that as a reference for a Python 2 implementation, c. 2015 ([link](https://inversethought.com/hg/))
- There was a conversation about whether to include it in the Python Statsmodels project ([link](https://groups.google.com/g/pystatsmodels/c/6QWW4tynDW8))
- There were concerns due to the original reference implementation being licensed under GNU-GPL
- However, as mentioned in that thread, such code may be relicensed with author permission

## What I did
- Reached out to Guy on LinkedIn ([link to profile](https://www.linkedin.com/in/guy-brys-412a8a65/)) to ask for permission
  - He granted permission
  - [link to permission from guy brys.png](https://github.com/hmustafamail/MedcoupleNLogN/blob/main/permission%20from%20guy%20brys.png) in my repo
- Revised Jordi's code for Python 3
- Validated my revised code against the (quadratic) statsmodels implementation using Jordi's data
  - RMSE was 1.03e-4
- Posted the revised code on Github ([link to repo](https://github.com/hmustafamail/MedcoupleNLogN))
- Opened an issue to discuss on the Statsmodels Github page ([link to issue](https://github.com/statsmodels/statsmodels/issues/9570))
