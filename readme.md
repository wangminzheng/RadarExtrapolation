# RadarExtrapolation

This project is aiming to extrapolate radar reflectivity for the future hours. The sample program is using 4 successive images to predict the next one.
The main algorithm is a 5-layer CNN with an adaptive final layer to reconstruct the image.

## Observation vs Prediction:
![featpnt](https://github.com/wangminzheng/RadarExtrapolation/blob/master/results/output.gif)

## Output: Time: 2018-05-03_18:42:00

![featpnt](https://github.com/wangminzheng/RadarExtrapolation/blob/master/results/201805031842.png)

## Input 1: Time: 2018-05-03_18:18:00

![featpnt](https://github.com/wangminzheng/RadarExtrapolation/blob/master/results/201805031818.png)

## Input 2: Time: 2018-05-03_18:24:00

![featpnt](https://github.com/wangminzheng/RadarExtrapolation/blob/master/results/201805031824.png)

## Input 3: Time: 2018-05-03_18:30:00

![featpnt](https://github.com/wangminzheng/RadarExtrapolation/blob/master/results/201805031830.png)

## Input 4: Time: 2018-05-03_18:36:00

![featpnt](https://github.com/wangminzheng/RadarExtrapolation/blob/master/results/201805031836.png)

