### ELF

README under construction...

## Training

The file structure for the training data should look like this:

```
vision
│
└───data
    │
    └───train
    │   │
    │   └───label1
    │   │   │
    │   │   └───img1.jpeg
    │   │   │
    │   │   └───img2.jpeg
    │   │   │
    │   │   └───...
    │   │
    │   └───label2
    │   │
    │   └───...
    │
    └───test
        │
        └───label1
        │
        └───label2
        │
        └───...
```

Once that is set up all there is to do is to run `train.py` and wait for it to finish.
