# AI in Wearable devices:

## Digital sampling & Signal Processing :
### Signal :
- its a series of numbers , that represent a physical phenomenon  eg like voltage.signal processing helps us analys these series of numbers to undestand more about the represented physical phenomenon in our case it is voltage.A signal is simply a series of numbers (e.g. [3, 4, 6, 2, 4] is a signal!). Typically these numbers represent a voltage that changes with respect to some physical phenomenon. When the voltage signal changes in time, it’s called a time series signal. Signal processing helps us analyze this sequence of numbers so we can learn more about the physical phenomenon it represents. 

- This signal is also periodic, meaning that it repeats itself over and over again.
-  The period is the amount of time it takes to make one repetition, which in this case is half a second. 
-  The frequency is the inverse of the period, or the number of repetitions per second, which is 2 hertz (Hz).
-  The last property of a periodic signal is the phase shift, which is similar to a time shift.
-   If two signals are time shifted by one full period, there is no difference between them. For this reason, we express this shift by the fraction of the period that they are shifted.
-    If a signal is shifted by a quarter of a period, we can say the phase shift is 90 degrees (360 being a full period). 
-    You can also measure phase shift in radians and there are 2 pi radians in a full period. So this phase shift would be half pi radians.
###### Definations: 
- Period: The amount of time it takes to make one repetition.
- Frequency: The amount of repetitions in a given time period, usually 1 second is the time period.
- Hertz (Hz): The units of the sampling rate. 1Hz means 1 sample per second.
- Phase Shift: The shift between two similar periodic signals expressed in fractions of a period (multiplied by 2𝛑 radians or 360°).


## Sampling Analog Signals:
##### Digital sampling : 


The goal of digital sampling is to take an analog continuous-time signal, typically a voltage, and to quantize it and discretize it in time so that we can store it in a finite amount of memory and use the magic of computers to process it. The component that does this is called an analog-to-digital converter or an ADC, this is an example of a transducer.
  An ADC encodes a range of physical values to a set of discrete numbers. In this example, the analog signal varies over time between -3V and +3V and we are using a 4-bit ADC, which means that the ADC has 4 bits to encode the range from -3 to +3 (ie. the bit-depth of our sensor is 4). The 4 bits indicate that there are 16 discrete values and we can see the effect of this quantization in the digitized signal.

###### Quantization Noise:
But typically, ADCs have many more bits and you won’t see quantization noise because it will be overpowered by other noise sources.

- Latent thermal energy in the system.
- Electronic noise from within the sensor.
- Electronic noise from the surroundings and the building itself.
- All these types of noise contribute to what we call the noise floor. Even when the incoming signal is perfectly flat, you will see some noise in the output. If you ever see a - flat line at 0 in the output, it’s because your sensor is broken.

###### Additive Noise:
This noise is additive, so you’ll see it on top of whatever incoming signal you have.

###### Signal Clipping

ADCs have a fixed range on the input. So for this example, our ADC was limited to -3V and +3V. This is known as the dynamic range of our sensor. When the input signal exceeds the dynamic range of the sensor, in this case from -4 to +4, everything greater than 3 will be clipped and set to 3 and everything smaller than -3 will be clipped to -3. We call this effect clipping, oversaturation, or undersaturation.

###### definations
- Transducer: Part of a sensor that converts a physical phenomenon into an electrical one (e.g., voltage)
- Analog-to-Digital Convert (ADC): A device (usually embedded in the sensor) that converts an analog voltage into an array of bits.
- Bit depth: The number of bits an ADC uses to create a sample. A 16-bit ADC produces a 16-bit number for each sample.
- Noise floor: The total amount of noise in the sensor, including electrical interference from the environment and other parts of the device, thermal noise, and quantization noise.
- Dynamic range: The physical range of the sensor. Values outside of this range will show up as clipping in the digital signal.
- Sampling rate: The frequency at which a sensor measures a signal.


## Resampling - Interpolation

- interpolation which is a technique that allows us to work with multiple signals that are sampled differently in time. We saw 2 signals that are both 1 Hz sine waves, but the one that is sampled at 60 Hz has many more data points than the one sampled at 25 Hz. After plotting and verifying the lengths of the signals, it might appear that s2_interp and s1 are the same, but it is most certainly not! By plotting the original and the interpolated signal together we can see that linear interpolation estimates points in between existing points by using a weighted average of the original points.

###### Definations :
- Interpolation: A method for estimating new data points within a range of discrete known data points.
- Resampling: The process of changing the sampling rate of a discrete signal to obtain a new discrete representation of the underlying continuous signal.

## Fourier Transform:


