# SODeep
Real-time deep learning framework for Slow oscillation (SO) detection.

Goal:
We want to build a system that detects different states of a SO in a real-time scenario. This might be interesting for phase-dependent stimulation.
Consequently, the network works on a windowed approach.

Some notes:
First ideas for a network architecture might be:

- 2s (at Fs=500) input --> this will cover more than 1 full SO
- Heavy downsampling and convolution as a first layer: Kernel 5 samples, stride 5 samples
  -> this downsamples the data to 200 features. Lets try maybe 16 filters
- Maybe multihead or single head attention layer after the CNN part
- 1 or 2 LSTM layers with few hidden units. Goal is to keep inference time low
- FC output with 3 classes: up-to-downstate, down-to-up state, and no event (0,1,2)
- for loss, use cross entropy


Notes on labeling the data:
- where should we define the up-to-down states? trough to zero? For now, zero makes sense


Notes on data preprocessing:
- lets use a heavy high cut at 0.5 Hz with order maybe 2. one pass only, not zero phase.
- Filter needs linear group delay over frequencies
- data needs to be normalized for a fixed range (e.g. -200 equals -1 and +200 mV equals +1)



Mohsen comments:
- use integral as feature to FC layer
- together with smooed derivative
- use previous(rolling) window for normalization

Notes on env:
Currently has pytorch
