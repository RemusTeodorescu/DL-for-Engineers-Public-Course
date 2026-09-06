# DL for Engineers - Public Course

**Deep Learning for Engineering** - MSc course, AAU Energy, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant Noman Khan.

This repository holds the student-facing material: the lecture slides as PDF,
the lecture recordings as MP4 where available, and the exercise sets as
Colab-ready notebooks. It is generated from the course's development repository
and refreshed after every change, so it is always current.

## How to work with the exercises

Open any notebook and click the **Open in Colab** badge at its top. The first
code cell fetches the set's library files from this repository, so nothing needs
uploading. Work through the notebooks of a set in numerical order; each set ends
with a short report notebook. Running locally works just as well: clone the
repository and open the notebooks in Jupyter with the set folder as the working
directory.

## Lectures

Each lecture folder holds the slides (PDF) and, once recorded, the lecture video (MP4).
GitHub shows videos in the browser; use *Download raw file* on a video's page to keep a copy.

- `lectures/L00-course-intro/` - [slides](lectures/L00-course-intro/L0_Welcome_How_This_Course_Works.pdf)
- `lectures/L01.1-python-that-runs/` - [slides](lectures/L01.1-python-that-runs/L1.1_Python_That_Runs.pdf), [recording](lectures/L01.1-python-that-runs/L1.1_Python_That_Runs.mp4)
- `lectures/L01.2-numpy-and-arrays/` - [slides](lectures/L01.2-numpy-and-arrays/L1.2_NumPy_and_the_Shape_of_Everything.pdf), [recording](lectures/L01.2-numpy-and-arrays/L1.2_NumPy_and_the_Shape_of_Everything.mp4)
- `lectures/L02.1-scientific-python/` - [slides](lectures/L02.1-scientific-python/L2.1_Scientific_Python.pdf), [recording](lectures/L02.1-scientific-python/L2.1_Scientific_Python.mp4)
- `lectures/L02.2-pytorch-and-autograd/` - [slides](lectures/L02.2-pytorch-and-autograd/L2.2_PyTorch_as_an_Array_Library.pdf), [recording](lectures/L02.2-pytorch-and-autograd/L2.2_PyTorch_as_an_Array_Library.mp4)
- `lectures/L03.1-what-is-ai/` - [slides](lectures/L03.1-what-is-ai/L3.1_What_is_AI.pdf), [recording](lectures/L03.1-what-is-ai/L3.1_What_is_AI.mp4)
- `lectures/L03.2-ai-application-examples/` - [slides](lectures/L03.2-ai-application-examples/L3.2_AI_Application_Examples.pdf), [recording](lectures/L03.2-ai-application-examples/L3.2_AI_Application_Examples.mp4)
- `lectures/L04.1-perceptron-to-neural-network/` - [slides](lectures/L04.1-perceptron-to-neural-network/L4.1_From_Perceptron_to_Neural_Network.pdf), [recording](lectures/L04.1-perceptron-to-neural-network/L4.1_From_Perceptron_to_Neural_Network.mp4)
- `lectures/L04.2-deep-neural-networks/` - [slides](lectures/L04.2-deep-neural-networks/L4.2_Deep_Neural_Networks.pdf), [recording](lectures/L04.2-deep-neural-networks/L4.2_Deep_Neural_Networks.mp4)
- `lectures/L05.1-convolutional-and-graph-networks/` - [slides](lectures/L05.1-convolutional-and-graph-networks/L5.1_Convolutional_and_Graph_Networks.pdf)
- `lectures/L05.2-sequences-and-attention/` - [slides](lectures/L05.2-sequences-and-attention/L5.2_Sequences_and_Attention.pdf)
- `lectures/L06.1-loss-functions-and-gradients/` - [slides](lectures/L06.1-loss-functions-and-gradients/L6.1_Loss_Functions_and_Gradients.pdf)
- `lectures/L06.2-post-training/` - [slides](lectures/L06.2-post-training/L6.2_Post_Training_and_Reinforcement_Learning.pdf)
- `lectures/L07.1-fundamentals-of-pinns/` - [slides](lectures/L07.1-fundamentals-of-pinns/L7.1_Fundamentals_of_PINNs.pdf)
- `lectures/L07.2-fundamental-pdes/` - [slides](lectures/L07.2-fundamental-pdes/L7.2_Fundamental_PDEs.pdf)
- `lectures/L08.1-stationary-heat/` - [slides](lectures/L08.1-stationary-heat/L8.1_Stationary_Heat.pdf)
- `lectures/L08.2-dynamic-heat-transfer/` - [slides](lectures/L08.2-dynamic-heat-transfer/L8.2_Dynamic_Heat_Transfer.pdf)
- `lectures/L09.1-laminar-flow/` - [slides](lectures/L09.1-laminar-flow/L9.1_Laminar_Flow.pdf)
- `lectures/L09.2-turbulent-flow/` - [slides](lectures/L09.2-turbulent-flow/L9.2_Turbulent_Flow.pdf)
- `lectures/L10.1-ionic-diffusion/` - [slides](lectures/L10.1-ionic-diffusion/L10.1_Ionic_Diffusion_Charge_Conservation.pdf)
- `lectures/L10.2-solid-oxide-cells/` - [slides](lectures/L10.2-solid-oxide-cells/L10.2_Solid_Oxide_Cells_Optimisation.pdf)
- `lectures/L11.1-vision-based-navigation/` - [slides](lectures/L11.1-vision-based-navigation/L11.1_Vision_Based_Navigation.pdf)
- `lectures/L11.2-dynamics-energy/` - [slides](lectures/L11.2-dynamics-energy/L11.2_Dynamics_Energy_and_Efficient_Driving.pdf)
- `lectures/L12.1-power-grid-stability-estimation/` - [slides](lectures/L12.1-power-grid-stability-estimation/L12.1_Power_Grid_Stability_Estimation.pdf)
- `lectures/L12.2-power-grid-stability-prediction/` - [slides](lectures/L12.2-power-grid-stability-prediction/L12.2_Power_Grid_Stability_Prediction.pdf)
- `lectures/_template/` - [slides](lectures/_template/LXX_Template.pdf)

## Exercises

- `exercises/Ex01-python-fundamentals/`
- `exercises/Ex02-pytorch-and-autograd/`
- `exercises/Ex03-first-model/`
- `exercises/Ex04-perceptron-to-mlp/`
- `exercises/Ex05-cnn-and-gnn/`
- `exercises/Ex06-training-lab/`
- `exercises/Ex07.1-poisson/`
- `exercises/Ex07.2-heat-and-wave/`
- `exercises/Ex08.1-stationary-heat/`
- `exercises/Ex08.2-transient-heat/`
- `exercises/Ex09.1-burgers-2d/`
- `exercises/Ex09.2-channel-flow/`
- `exercises/Ex10.1-spm-and-thermal/`
- `exercises/Ex10.2-solid-oxide-cell/`
- `exercises/Ex11.1-camera-navigation/`
- `exercises/Ex11.2-energy-optimization/`
- `exercises/Ex12.1-power-grid-stability-estimation/`
- `exercises/Ex12.2-power-grid-stability-prediction/`

## Licence

(c) 2026 Remus Teodorescu, Aalborg University. A licence will be added shortly;
until then the material is shared for the course's students.

Synced from the development repository; 26 lecture PDFs, 8 recordings and 18 exercise sets.
