# Unit tests

`Grasp` uses python's ``unittest`` framework. You can run each test file individually or let the framework discover and
run unit tests for you.


    python -m unittest


Alternatively, you can run tests one by one (or in batch):


    for file in test_*py; do echo $file; python $file; done
    
    
 
