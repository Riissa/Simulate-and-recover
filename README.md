# Simulate-and-recover

The EZ diffusion model, a simplified version of the Drift Diffusion Model (DDM), allows us to directly compute estimated parameters that represent cognitive decision-making processes from observed summary statistics based on response time and accuracy data. These parameters include Drift Rate (ν), Boundary Separation (α) and Non-decision Time (τ). 

Drift rate represents the speed of accumulation of evidence. With a higher drift rate, decisions are faster and more accurate while a lower drift rate suggests slower and less accurate decisions. 
Boundary separation is the threshold of evidence for decision making. A larger boundary separation would indicate more conservativeness in decision-making and reflect a longer time to respond to assure the choice is accurate whereas a lower boundary separation would suggest more impulsive choices with quicker responses and less accuracy. 
Non-decision time is a parameter that accounts for any processes unrelated to decision making, such as the time it takes for encoding and for the motor response. 

The code for this model was organized into three main files to implement the EZ diffusion model efficiently which include generate, simulate and recover.
Generate.py contains the forward equations that provide the summary statistics in terms of parameters. So, provided the input parameters of drift rate, boundary separation and non-decision time, generate.py gives us the accuracy rate (R_pred), mean reaction time (M_pred) and variance of reaction time (V_pred). 
Then, simulate.py adds noise to these summary statistics to mimic real world data and bring in variability. These simulated observed values of the summary statistics are (R_obs), (M_obs) and (V_obs). 
Finally, recover.py takes these simulated values and applies the inverse EZ equations to estimate the original parameters. Recover.py plays an important role in testing whether the EZ diffusion model can reliably infer cognitive decision-making processes from behavioral data. 

To evaluate the accuracy of the EZ diffusion model overall, the full generate-simulate-recover process is repeated 1,000 times for three different sample sizes of 10, 40 and 4000. 

As shown in results.txt, the average bias of the parameters for all three sample sizes is close to 0. Additionally, as sample size increases (from 10 to 4000), average squared error decreases.

Bias measures the average difference between the true parameters and the estimated parameters while squared error measures how much each of the estimated parameters deviate from the true parameters. A bias of 0 suggests that the difference between the true parameters and the estimated parameters is small and the EZ diffusion model is unbiased, providing a reliable estimation of these parameters. Additionally, the results for average squared error suggest that the EZ diffusion model is valid as expected behavior of average squared error decreasing is demonstrated as sample size increases. It is important to note that for the smallest sample size (10), the larger squared error suggests more unreliability whereas for the larger sample size (4000), the EZ diffusion models estimates are closer to the true values. Overall, from these results we can determine that the EZ diffusion model is consistent and with larger sample sizes this model is more accurate further aligning with expectations that larger sample sizes reduce variability and overall improve estimation accuracy. 
